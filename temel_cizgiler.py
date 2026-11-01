"""
temel_cizgiler.py
=================
Deney 4: Yerel veriyle transfer öğrenme, hazır/ön eğitimsiz temel çizgilere göre ne kazandırıyor?
Kollar (önceden tanımlı, hepsi raporlanır; hepsi aynı kayıt-gruplu bölmelerde):
  LR_BirdNET            : BirdNET gömmeleri + lojistik regresyon (yerel veriyle eğitilir)
  LR_MFCC               : MFCC özetleri + lojistik regresyon (ön eğitimsiz)
  SifirAtis_BirdNET_kisitli : BirdNET'in hazır tahmini, yalnızca veri setindeki sınıflar arasından
  SifirAtis_BirdNET_acik    : BirdNET'in hazır tahmini, tüm BirdNET türleri arasından
Önceden tanımlı birincil karşılaştırma: LR_BirdNET - SifirAtis_BirdNET_kisitli (makro F1, Nadeau-Bengio düzeltmeli p).
İkincil: LR_BirdNET - LR_MFCC ve LR_BirdNET - SifirAtis_BirdNET_acik.

Kullanım (proje klasöründe, .venv etkinken):
    python temel_cizgiler.py
    python temel_cizgiler.py --kat 5 --tekrar 5

Mevcut birdnet_onbellek.npz yeniden kullanılır. Çıktılar: temel_fold.csv, temel_ozet.csv,
temel_eslestirilmis.csv, temel_kapsam.csv
"""
import argparse
import csv

import numpy as np

from src.ml import baselines
from src.ml import birdnet_zero_shot as bzs
from src.ml import embedding_comparison as ec
from src.ml.dataset_builder import build_training_samples


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
    ap.add_argument("--birdnet-kutuphane", default="litert", choices=["litert", "tflite"])
    ap.add_argument("--agirliksiz", action="store_true", help="sınıf ağırlığını kapat")
    args = ap.parse_args()

    kept, _, _ = build_training_samples(
        min_samples_per_class=args.min_segment, min_recordings_per_class=args.min_kayit)
    class_names = sorted({s.species for s in kept})
    print(f"Örnek: {len(kept)} segment, {len({s.recording_id for s in kept})} kayıt, {len(class_names)} tür")

    print("BirdNET gömmeleri hazırlanıyor...")
    X = ec.extract_birdnet_embeddings(
        kept, cache_path="birdnet_onbellek.npz", library=args.birdnet_kutuphane,
        progress_callback=lambda c, t: print(f"\rBirdNET: {c}/{t}", end="", flush=True))
    print("\nMFCC öznitelikleri hazırlanıyor...")
    mfcc = baselines.load_or_extract_mfcc(
        kept, cache_path="mfcc_onbellek.npz",
        progress_callback=lambda c, t: print(f"\rMFCC: {c}/{t}", end="", flush=True))
    print("\nBirdNET hazır tahminleri alınıyor...")
    kis, aci, match = bzs.zero_shot_predictions(
        kept, class_names, library=args.birdnet_kutuphane,
        progress_callback=lambda c, t: print(f"\rHazır tahmin: {c}/{t}", end="", flush=True))
    print()
    kapsam = [{"sinif": c, "birdnet_etiketi": match[c] or "(eşleşme yok)"} for c in class_names]
    for k in kapsam:
        print(f"  {k['sinif']:<28} -> {k['birdnet_etiketi']}")
    print(f"Eşleşen sınıf: {sum(1 for c in class_names if match[c])}/{len(class_names)}")

    rows, oof, n_classes = ec.run_embedding_comparison(
        {"BirdNET": X, "MFCC": mfcc}, kept, n_splits=args.kat, n_repeats=args.tekrar,
        class_weight=not args.agirliksiz, add_concat=False,
        progress_callback=lambda d, t: print(f"\rDeğerlendirme: {d}/{t}", end="", flush=True))
    print()
    rows += bzs.fold_rows(kis, kept, "SifirAtis_BirdNET_kisitli", args.kat, args.tekrar)
    rows += bzs.fold_rows(aci, kept, "SifirAtis_BirdNET_acik", args.kat, args.tekrar)

    ozet = []
    for name in sorted({r["yontem"] for r in rows}):
        sel = [r for r in rows if r["yontem"] == name]
        d = {"yontem": name, "n_kat": len(sel)}
        for m in ("dogruluk", "makro_f1"):
            v = np.array([r[m] for r in sel])
            d[m + "_ort"] = float(v.mean())
            d[m + "_ss"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        ozet.append(d)
    ozet.sort(key=lambda r: -r["makro_f1_ort"])
    for o in ozet:
        print(f"{o['yontem']:<28} doğruluk {o['dogruluk_ort']:.3f} ± {o['dogruluk_ss']:.3f} "
              f"| makro F1 {o['makro_f1_ort']:.3f} ± {o['makro_f1_ss']:.3f}")

    pairs = (("LR_BirdNET", "SifirAtis_BirdNET_kisitli", "LR_BirdNET − hazır BirdNET (kısıtlı) (birincil)"),
             ("LR_BirdNET", "LR_MFCC", "LR_BirdNET − LR_MFCC"),
             ("LR_BirdNET", "SifirAtis_BirdNET_acik", "LR_BirdNET − hazır BirdNET (açık)"))
    fark = []
    for metric in ("makro_f1", "dogruluk"):
        for a, b, label in pairs:
            d = dict(ec.paired_difference(rows, a, b, metric))
            d["karsilastirma"] = label + ("" if metric == "makro_f1" else " (doğruluk)")
            fark.append(d)
    print("\nEşleştirilmiş farklar:")
    for f in fark:
        print(f"  {f['karsilastirma']:<52} fark {f['fark_ort']:+.3f} ± {f['fark_ss']:.3f} | "
              f"üstün kat {f['a_kazanilan_kat']}/{f['n_kat']} | p(düzeltilmiş) {f['p_duzeltilmis']:.3f}")
    _yaz("temel_fold.csv", rows)
    _yaz("temel_ozet.csv", ozet)
    _yaz("temel_eslestirilmis.csv", fark)
    _yaz("temel_kapsam.csv", kapsam)
    print("Yazıldı: temel_fold.csv, temel_ozet.csv, temel_eslestirilmis.csv, temel_kapsam.csv")


if __name__ == "__main__":
    main()
