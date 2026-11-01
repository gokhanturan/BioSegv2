"""
girdi_bicimi_kontrolu.py
========================
Deney 2: Perch 2.0'a 3 sn'lik segmentin 5 sn'lik pencerede nasıl verildiği sonucu etkiliyor mu?
Aynı segmentler, aynı kayıt-gruplu bölmeler ve aynı sınıflandırıcı (LR) ile karşılaştırılır:
  Perch_sifir  : segment + sıfır dolgu (uygulamadaki mevcut biçim)
  Perch_tekrar : segment 5 sn dolana kadar tekrarlanır
  BirdNET      : 3 sn doğal girdi (isteğe bağlı, --birdnet)
Önceden tanımlı birincil karşılaştırma: Perch_tekrar − Perch_sifir (makro F1, Nadeau-Bengio düzeltmeli p).
Sonuç ne çıkarsa çıksın tüm kipler raporlanır.

Kullanım (proje klasöründe, .venv etkinken):
    python girdi_bicimi_kontrolu.py --birdnet
    python girdi_bicimi_kontrolu.py --birdnet --kat 5 --tekrar 5

Mevcut embedding_onbellek.npz (sıfır dolgu) yeniden kullanılır; tekrar kipi perch_tekrar_onbellek.npz'ye yazılır.
Çıktılar: girdi_fold.csv, girdi_ozet.csv, girdi_eslestirilmis.csv
"""
import argparse
import csv

from src.ml.pipeline import load_or_extract_embeddings
from src.ml import embedding_comparison as ec
from src.ml import input_format_control as ifc


def _yaz(name, rows):
    if not rows:
        return
    with open(name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kat", type=int, default=5)
    ap.add_argument("--tekrar", type=int, default=3)
    ap.add_argument("--min-segment", type=int, default=30)
    ap.add_argument("--min-kayit", type=int, default=2)
    ap.add_argument("--birdnet", action="store_true", help="BirdNET 2.4 gömmelerini ekle")
    ap.add_argument("--birdnet-kutuphane", default="litert", choices=["litert", "tflite"])
    ap.add_argument("--agirliksiz", action="store_true", help="sınıf ağırlığını kapat")
    args = ap.parse_args()

    X_sifir, kept = load_or_extract_embeddings(
        min_samples_per_class=args.min_segment, min_recordings_per_class=args.min_kayit,
        cache_path="embedding_onbellek.npz", backend="perch",
        progress_callback=lambda st, c, t: print(f"\r{st}: {c}/{t}", end="", flush=True))
    print(f"\nÖrnek: {len(kept)} segment, {len({s.recording_id for s in kept})} kayıt, "
          f"{len({s.species for s in kept})} tür")

    from src.detection.detector import get_shared_perch_classifier
    clf = get_shared_perch_classifier()
    clf.ensure_loaded()
    print("Perch (tekrar kipi) gömmeleri hazırlanıyor...")
    X_tekrar = ifc.extract_perch_embeddings_mode(
        clf, kept, "tekrar", cache_path="perch_tekrar_onbellek.npz",
        progress_callback=lambda c, t: print(f"\rPerch tekrar: {c}/{t}", end="", flush=True))
    print()

    sets = {"Perch_sifir": X_sifir, "Perch_tekrar": X_tekrar}
    if args.birdnet:
        print("BirdNET 2.4 gömmeleri hazırlanıyor...")
        sets["BirdNET"] = ec.extract_birdnet_embeddings(
            kept, cache_path="birdnet_onbellek.npz", library=args.birdnet_kutuphane,
            progress_callback=lambda c, t: print(f"\rBirdNET: {c}/{t}", end="", flush=True))
        print()

    rows, oof, n_classes = ec.run_embedding_comparison(
        sets, kept, n_splits=args.kat, n_repeats=args.tekrar, class_weight=not args.agirliksiz,
        add_concat=False,
        progress_callback=lambda d, t: print(f"\rDeğerlendirme: {d}/{t}", end="", flush=True))
    print()
    ozet = ec.summarize_comparison(rows, oof, n_classes)
    for o in ozet:
        ga = (f" | GA95 [{o['makro_f1_ga_alt']:.3f}, {o['makro_f1_ga_ust']:.3f}]" if "makro_f1_ga_alt" in o else "")
        print(f"{o['yontem']:<22} doğruluk {o['dogruluk_ort']:.3f} ± {o['dogruluk_ss']:.3f} "
              f"| makro F1 {o['makro_f1_ort']:.3f} ± {o['makro_f1_ss']:.3f}{ga}")
    fark = ifc.padding_effect_rows(rows, "LR_Perch_sifir", "LR_Perch_tekrar", "LR_BirdNET", "makro_f1")
    fark_dog = [dict(f, karsilastirma=f["karsilastirma"] + " (doğruluk)")
                for f in ifc.padding_effect_rows(rows, "LR_Perch_sifir", "LR_Perch_tekrar", "LR_BirdNET", "dogruluk")]
    print("\nEşleştirilmiş farklar (makro F1; birincil: ilk satır):")
    for f in fark:
        print(f"  {f['karsilastirma']:<28} fark {f['fark_ort']:+.3f} ± {f['fark_ss']:.3f} | "
              f"üstün kat {f['a_kazanilan_kat']}/{f['n_kat']} | p(düzeltilmiş) {f['p_duzeltilmis']:.3f}")
    _yaz("girdi_fold.csv", rows)
    _yaz("girdi_ozet.csv", ozet)
    _yaz("girdi_eslestirilmis.csv", fark + fark_dog)
    print("Yazıldı: girdi_fold.csv, girdi_ozet.csv, girdi_eslestirilmis.csv")


if __name__ == "__main__":
    main()
