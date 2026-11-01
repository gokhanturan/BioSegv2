"""
gomme_karsilastirma.py
======================
Perch 2.0 ile alternatif ses gömmelerini (BirdNET 2.4, harici .npz) AYNI kayıt-gruplu bölmelerde
karşılaştırır: kat ortalaması ± s.s., kayıt düzeyinde bootstrap %95 GA ve Perch'e göre
eşleştirilmiş fark (Nadeau-Bengio düzeltmeli p).

Kullanım (proje klasöründe, .venv etkinken):
    python gomme_karsilastirma.py                          # yalnızca Perch (kontrol)
    python gomme_karsilastirma.py --birdnet                # + BirdNET 2.4 ve birleşik (pip install birdnet)
    python gomme_karsilastirma.py --birdnet --harici AVES=aves.npz --harici BEATs=beats.npz
    python gomme_karsilastirma.py --birdnet --kat 5 --tekrar 5

Harici .npz biçimi: `X` (M, d) ve `keys` (M,) — keys, embedding_onbellek.npz'deki anahtarlarla aynı
("dosya||tür||kayıt_id"). Veri setine ve eğitilmiş modele DOKUNMAZ; yalnızca CSV çıktıları yazar.
Çıktılar: gomme_fold.csv, gomme_ozet.csv, gomme_eslestirilmis.csv
"""
import argparse
import csv

from src.ml.pipeline import load_or_extract_embeddings
from src.ml import embedding_comparison as ec


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
    ap.add_argument("--birdnet-kutuphane", default="litert", choices=["litert", "tflite"],
                    help="birdnet.load(..., library=...) (varsayılan litert: TensorFlow gerekmez)")
    ap.add_argument("--harici", action="append", default=[], metavar="AD=YOL.npz",
                    help="harici gömme (birden çok kez verilebilir)")
    ap.add_argument("--agirliksiz", action="store_true", help="sınıf ağırlığını kapat")
    ap.add_argument("--referans", default="LR_Perch")
    args = ap.parse_args()

    X, kept = load_or_extract_embeddings(
        min_samples_per_class=args.min_segment, min_recordings_per_class=args.min_kayit,
        cache_path="embedding_onbellek.npz", backend="perch",
        progress_callback=lambda st, c, t: print(f"\r{st}: {c}/{t}", end="", flush=True))
    print(f"\nÖrnek: {len(kept)} segment, {len({s.recording_id for s in kept})} kayıt, "
          f"{len({s.species for s in kept})} tür")

    sets = {"Perch": X}
    if args.birdnet:
        print("BirdNET 2.4 gömmeleri hazırlanıyor (ilk çalıştırmada model iner)...")
        sets["BirdNET"] = ec.extract_birdnet_embeddings(
            kept, cache_path="birdnet_onbellek.npz", library=args.birdnet_kutuphane,
            progress_callback=lambda c, t: print(f"\rBirdNET: {c}/{t}", end="", flush=True))
        print()
    for item in args.harici:
        ad, _, yol = item.partition("=")
        if not ad or not yol:
            raise SystemExit(f"--harici AD=YOL.npz biçiminde olmalı: {item}")
        sets[ad] = ec.load_external_embeddings(yol, kept)

    rows, oof, n_classes = ec.run_embedding_comparison(
        sets, kept, n_splits=args.kat, n_repeats=args.tekrar, class_weight=not args.agirliksiz,
        progress_callback=lambda d, t: print(f"\rDeğerlendirme: {d}/{t}", end="", flush=True))
    print()
    ozet = ec.summarize_comparison(rows, oof, n_classes)
    for o in ozet:
        ga = (f" | GA95 [{o['makro_f1_ga_alt']:.3f}, {o['makro_f1_ga_ust']:.3f}]" if "makro_f1_ga_alt" in o else "")
        vf = (f" | tür varlığı F1 (≥1 seg) {o['varlik_f1_min1']:.3f}, (≥2 seg) {o['varlik_f1_min2']:.3f}"
              if "varlik_f1_min1" in o else "")
        print(f"{o['yontem']:<28} doğruluk {o['dogruluk_ort']:.3f} ± {o['dogruluk_ss']:.3f} "
              f"| makro F1 {o['makro_f1_ort']:.3f} ± {o['makro_f1_ss']:.3f}{ga}{vf}")
    fark = ec.pairwise_table(rows, args.referans)
    if fark:
        print(f"\nEşleştirilmiş fark (yöntem − {args.referans}, makro F1):")
        for f in fark:
            print(f"  {f['karsilastirma']:<34} fark {f['fark_ort']:+.3f} ± {f['fark_ss']:.3f} | "
                  f"üstün kat {f['a_kazanilan_kat']}/{f['n_kat']} | p(düzeltilmiş) {f['p_duzeltilmis']:.3f}")
    _yaz("gomme_fold.csv", rows)
    _yaz("gomme_ozet.csv", ozet)
    _yaz("gomme_eslestirilmis.csv", fark)
    print("Yazıldı: gomme_fold.csv, gomme_ozet.csv, gomme_eslestirilmis.csv")


if __name__ == "__main__":
    main()
