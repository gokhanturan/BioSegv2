"""
capraz_dogrulama.py
===================
Tekrarlı, KAYIT bazlı (grup-farkında) k-katlı çapraz doğrulama çalıştırır ve sınıf
ağırlıklarının (var / yok) etkisini AYNI bölmelerde (eşleştirilmiş) karşılaştırır.

Kullanım (proje klasöründe, .venv etkinken):
    python capraz_dogrulama.py                      # 5 kat x 3 tekrar
    python capraz_dogrulama.py --kat 5 --tekrar 5   # daha fazla tekrar
    python capraz_dogrulama.py --kuş-filtresi       # yalnızca bilinen kuş türleri (518 tür)
    python capraz_dogrulama.py --temel-cizgi        # + LR(BirdNET) ve MFCC+LR temel çizgileri (aynı bölmeler)

Embedding'ler ilk çalıştırmada BirdNET 2.4 ile çıkarılıp `embedding_onbellek.npz` dosyasına
kaydedilir; sonraki çalıştırmalarda yeniden çıkarılmaz (veri değişirse otomatik yenilenir).
Çıktılar: cv_sonuclari.csv (kat bazlı) ve cv_tur_bazli.csv (birleştirilmiş tür bazlı);
--temel-cizgi ile ayrıca cv_temel_cizgi.csv (kat bazlı) ve cv_yontem_ozeti.csv (yöntem karşılaştırması).
Mevcut modele ve CSV veri setine DOKUNMAZ (salt-okunur; yalnızca bu iki çıktı dosyasını yazar).
"""
import argparse
import csv
import os

from src.ml.pipeline import load_or_extract_embeddings, run_cross_validation


def _yaz(name, rows):
    with open(name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def _temel_cizgiler(args, X, kept, res):
    import numpy as np
    from src.ml import baselines

    print("MFCC öznitelikleri hazırlanıyor...")
    mfcc = baselines.load_or_extract_mfcc(
        kept, cache_path="mfcc_onbellek.npz",
        progress_callback=lambda c, t: print(f"\rMFCC: {c}/{t}", end="", flush=True))
    print()
    zs = None
    rows = baselines.run_baselines(
        X, kept, mfcc=mfcc, zero_shot_scores=zs, n_splits=args.kat, n_repeats=args.tekrar,
        progress_callback=lambda d, t: print(f"\rTemel çizgi: {d}/{t}", end="", flush=True))
    print()
    ozet = baselines.summarize(rows)
    for opt in (False, True):
        if opt in res.summary:
            a, f = res.summary[opt]["dogruluk"], res.summary[opt]["makro_f1"]
            ozet.append({"yontem": "MLP_BirdNET", "sinif_agirligi": opt, "n_kat": args.kat * args.tekrar,
                         "dogruluk_ort": a[0], "dogruluk_ss": a[1], "makro_f1_ort": f[0], "makro_f1_ss": f[1]})
    for o in ozet:
        print(f"{o['yontem']:<16} ağırlık={str(o['sinif_agirligi']):<5} doğruluk {o['dogruluk_ort']:.3f} ± {o['dogruluk_ss']:.3f} "
              f"| makro F1 {o['makro_f1_ort']:.3f} ± {o['makro_f1_ss']:.3f}")
    # Eşleştirilmiş farklar (aynı bölmeler; Nadeau-Bengio düzeltmeli p)
    fark = []
    for ad, cw in (("LR_BirdNET", True), ("LR_BirdNET", False), ("SifirAtis_Perch", None), ("LR_MFCC", True)):
        for metrik in ("dogruluk", "makro_f1"):
            try:
                d = baselines.paired_comparison(res.fold_rows, rows, ad, cw, metric=metrik)
            except ValueError:
                continue
            if cw is not None:
                d["karsilastirma"] = f"{d['karsilastirma']} ({'ağırlıklı' if cw else 'ağırlıksız'})"
            fark.append(d)
    print("\nEşleştirilmiş fark (MLP ağırlıklı − temel çizgi):")
    for f in fark:
        print(f"  {f['karsilastirma']:<40} {f['metrik']:<9} fark {f['fark_ort']:+.3f} ± {f['fark_ss']:.3f} | "
              f"MLP üstün kat {f['mlp_kazanilan_kat']}/{f['n_kat']} | p(naif) {f['p_naif']:.4f} | p(düzeltilmiş) {f['p_duzeltilmis']:.3f}")
    if fark:
        _yaz("cv_eslestirilmis_fark.csv", fark)
    _yaz("cv_temel_cizgi.csv", rows)
    _yaz("cv_yontem_ozeti.csv", ozet)
    print("Yazıldı: cv_temel_cizgi.csv ve cv_yontem_ozeti.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kat", type=int, default=5)
    ap.add_argument("--tekrar", type=int, default=3)
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--min-segment", type=int, default=30)
    ap.add_argument("--min-kayit", type=int, default=2)
    ap.add_argument("--kuş-filtresi", dest="kus_filtresi", action="store_true")
    ap.add_argument("--temel-cizgi", dest="temel_cizgi", action="store_true",
                    help="LR(BirdNET) ve MFCC+LR temel çizgilerini AYNI bölmelerde de değerlendir")
    args = ap.parse_args()

    X, kept = load_or_extract_embeddings(
        min_samples_per_class=args.min_segment, min_recordings_per_class=args.min_kayit,
        restrict_to_known_bird_species=args.kus_filtresi, cache_path="embedding_onbellek.npz",
        progress_callback=lambda st, c, t: print(f"\r{st}: {c}/{t}", end="", flush=True),
    )
    print(f"\nÖrnek: {len(kept)} segment, {len({s.recording_id for s in kept})} kayıt, "
          f"{len({s.species for s in kept})} tür")

    res = run_cross_validation(
        X, kept, n_splits=args.kat, n_repeats=args.tekrar, max_epochs=args.epoch,
        progress_callback=lambda d, t: print(f"\rEğitim: {d}/{t}", end="", flush=True),
    )
    print()
    for opt, label in ((False, "Sınıf ağırlığı YOK"), (True, "Sınıf ağırlığı VAR")):
        if opt in res.summary:
            a, f = res.summary[opt]["dogruluk"], res.summary[opt]["makro_f1"]
            print(f"{label}: doğruluk {a[0]:.3f} ± {a[1]:.3f} | makro F1 {f[0]:.3f} ± {f[1]:.3f}")
    d = res.summary.get("fark_agirlikli_eksi_agirliksiz")
    if d:
        print(f"Eşleştirilmiş fark (ağırlıklı − ağırlıksız): doğruluk {d['dogruluk'][0]:+.3f} ± {d['dogruluk'][1]:.3f}, "
              f"makro F1 {d['makro_f1'][0]:+.3f} ± {d['makro_f1'][1]:.3f}  (± = katlar arası std)")

    for name, rows in (("cv_sonuclari.csv", res.fold_rows), ("cv_tur_bazli.csv", res.per_class_rows)):
        with open(name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(rows)
    print("Yazıldı:", os.path.abspath("cv_sonuclari.csv"), "ve", os.path.abspath("cv_tur_bazli.csv"))

    if args.temel_cizgi:
        _temel_cizgiler(args, X, kept, res)


if __name__ == "__main__":
    main()
