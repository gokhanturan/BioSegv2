"""
seviye_kontrolu.py
==================
Deney 3: Ses seviyesi bir kısayol mu? Kollar (önceden tanımlı, hepsi raporlanır):
  BirdNET       : mevcut gömmeler
  BirdNET_rms   : segmentler veri setinin medyan RMS seviyesine ölçeklenmiş
  BirdNET_tepe  : segmentler tepe -3 dBFS'e ölçeklenmiş
  Seviye        : yalnızca [RMS dBFS, tepe dBFS] ile sınıflandırma
  Sabit         : özelliksiz taban çizgisi
Önceden tanımlı birincil karşılaştırma: BirdNET_rms - BirdNET (makro F1, Nadeau-Bengio düzeltmeli p).
İkincil: BirdNET_tepe - BirdNET ve Seviye - Sabit. Sonuç ne çıkarsa çıksın tüm kollar raporlanır.

Kullanım (proje klasöründe, .venv etkinken):
    python seviye_kontrolu.py
    python seviye_kontrolu.py --kat 5 --tekrar 5

Mevcut birdnet_onbellek.npz yeniden kullanılır; normalize kolların gömmeleri
birdnet_rms_onbellek.npz ve birdnet_tepe_onbellek.npz dosyalarına yazılır.
Çıktılar: seviye_fold.csv, seviye_ozet.csv, seviye_eslestirilmis.csv
"""
import argparse
import csv

import numpy as np

from src.ml.dataset_builder import build_training_samples
from src.ml import embedding_comparison as ec
from src.ml import level_control as lc


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
    print(f"Örnek: {len(kept)} segment, {len({s.recording_id for s in kept})} kayıt, "
          f"{len({s.species for s in kept})} tür")

    sets = {}
    print("BirdNET gömmeleri (özgün seviye) hazırlanıyor...")
    sets["BirdNET"] = ec.extract_birdnet_embeddings(
        kept, cache_path="birdnet_onbellek.npz", library=args.birdnet_kutuphane,
        progress_callback=lambda c, t: print(f"\rBirdNET: {c}/{t}", end="", flush=True))
    print()
    hedef = lc.median_rms_dbfs(kept)
    print(f"RMS kipi hedef seviyesi (veri setinin medyanı): {hedef:.2f} dBFS")
    for mod, ad in (("rms", "BirdNET_rms"), ("tepe", "BirdNET_tepe")):
        print(f"{ad} gömmeleri hazırlanıyor...")
        sets[ad] = lc.embeddings_for_mode(
            kept, mod, cache_path=f"birdnet_{mod}_onbellek.npz", library=args.birdnet_kutuphane,
            target_rms_dbfs=hedef if mod == "rms" else None,
            progress_callback=lambda c, t, a=ad: print(f"\r{a}: {c}/{t}", end="", flush=True))
        print()
    sets["Seviye"] = lc.level_features(kept)
    sets["Sabit"] = np.zeros((len(kept), 1))

    rows, oof, n_classes = ec.run_embedding_comparison(
        sets, kept, n_splits=args.kat, n_repeats=args.tekrar, class_weight=not args.agirliksiz,
        add_concat=False,
        progress_callback=lambda d, t: print(f"\rDeğerlendirme: {d}/{t}", end="", flush=True))
    print()
    ozet = ec.summarize_comparison(rows, oof, n_classes)
    for o in ozet:
        ga = (f" | GA95 [{o['makro_f1_ga_alt']:.3f}, {o['makro_f1_ga_ust']:.3f}]" if "makro_f1_ga_alt" in o else "")
        print(f"{o['yontem']:<18} doğruluk {o['dogruluk_ort']:.3f} ± {o['dogruluk_ss']:.3f} "
              f"| makro F1 {o['makro_f1_ort']:.3f} ± {o['makro_f1_ss']:.3f}{ga}")

    pairs = (("LR_BirdNET_rms", "LR_BirdNET", "BirdNET_rms − BirdNET (birincil)"),
             ("LR_BirdNET_tepe", "LR_BirdNET", "BirdNET_tepe − BirdNET"),
             ("LR_Seviye", "LR_Sabit", "Seviye − Sabit"))
    fark = []
    for metric in ("makro_f1", "dogruluk"):
        for a, b, label in pairs:
            d = dict(ec.paired_difference(rows, a, b, metric))
            d["karsilastirma"] = label + ("" if metric == "makro_f1" else " (doğruluk)")
            fark.append(d)
    print("\nEşleştirilmiş farklar:")
    for f in fark:
        print(f"  {f['karsilastirma']:<42} fark {f['fark_ort']:+.3f} ± {f['fark_ss']:.3f} | "
              f"üstün kat {f['a_kazanilan_kat']}/{f['n_kat']} | p(düzeltilmiş) {f['p_duzeltilmis']:.3f}")
    _yaz("seviye_fold.csv", rows)
    _yaz("seviye_ozet.csv", ozet)
    _yaz("seviye_eslestirilmis.csv", fark)
    print("Yazıldı: seviye_fold.csv, seviye_ozet.csv, seviye_eslestirilmis.csv")


if __name__ == "__main__":
    main()
