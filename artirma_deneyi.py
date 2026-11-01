"""
artirma_deneyi.py
=================
Eğitim kümesine ses düzeyinde veri artırma (ses seviyesi, zaman kaydırma, pembe gürültü) uygulamanın
sınıflandırma başarısını DEĞİŞTİRİP DEĞİŞTİRMEDİĞİNİ ölçer. Aynı kayıt-gruplu bölmelerde "artırmasız" ile
karşılaştırılır; artırma yalnızca eğitim katlarına uygulanır, test katı özgün segmentlerdir.

Kullanım (proje klasöründe, .venv etkinken; Perch 2.0 / TensorFlow gerekir):
    python artirma_deneyi.py
    python artirma_deneyi.py --kat 5 --tekrar 5
    python artirma_deneyi.py --turler gain,noise

İlk çalıştırmada her segment için her dönüşümün Perch gömmesi çıkarılır (≈ 3 × segment sayısı; dakikalar
sürebilir) ve artirma_onbellek.npz'ye yazılır. Veri setine ve eğitilmiş modele DOKUNMAZ.
Çıktılar: artirma_fold.csv, artirma_ozet.csv, artirma_eslestirilmis.csv
"""
import argparse
import csv

from src.ml import augmentation as aug
from src.ml import embedding_comparison as ec
from src.ml.pipeline import load_or_extract_embeddings


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
    ap.add_argument("--turler", default=",".join(aug.DEFAULT_KINDS),
                    help="virgülle ayrılmış dönüşümler: gain, shift, noise")
    ap.add_argument("--tohum", type=int, default=0, help="kopyaların rastgelelik tohumu")
    ap.add_argument("--agirliksiz", action="store_true", help="sınıf ağırlığını kapat")
    args = ap.parse_args()
    kinds = [k.strip() for k in args.turler.split(",") if k.strip()]

    X, kept = load_or_extract_embeddings(
        min_samples_per_class=args.min_segment, min_recordings_per_class=args.min_kayit,
        cache_path="embedding_onbellek.npz", backend="perch",
        progress_callback=lambda st, c, t: print(f"\r{st}: {c}/{t}", end="", flush=True))
    print(f"\nÖrnek: {len(kept)} segment, {len({s.recording_id for s in kept})} kayıt, "
          f"{len({s.species for s in kept})} tür")

    print(f"Artırılmış gömmeler hazırlanıyor ({', '.join(kinds)})...")
    X_aug = aug.extract_augmented_embeddings(
        kept, kinds=kinds, seed=args.tohum, cache_path="artirma_onbellek.npz",
        progress_callback=lambda c, t: print(f"\rArtırma: {c}/{t}", end="", flush=True))
    print()

    rows, oof, n_classes = aug.run_augmentation_comparison(
        X, X_aug, kinds, kept, n_splits=args.kat, n_repeats=args.tekrar,
        class_weight=not args.agirliksiz,
        progress_callback=lambda d, t: print(f"\rDeğerlendirme: {d}/{t}", end="", flush=True))
    print()
    ozet = ec.summarize_comparison(rows, oof, n_classes)
    for o in ozet:
        print(f"{o['yontem']:<28} doğruluk {o['dogruluk_ort']:.3f} ± {o['dogruluk_ss']:.3f} "
              f"| makro F1 {o['makro_f1_ort']:.3f} ± {o['makro_f1_ss']:.3f} "
              f"| GA95 [{o['makro_f1_ga_alt']:.3f}, {o['makro_f1_ga_ust']:.3f}]")
    fark = ec.pairwise_table(rows, "LR_Artirmasiz")
    if fark:
        print("\nEşleştirilmiş fark (yöntem − artırmasız, makro F1):")
        for f in fark:
            print(f"  {f['karsilastirma']:<40} fark {f['fark_ort']:+.3f} ± {f['fark_ss']:.3f} | "
                  f"üstün kat {f['a_kazanilan_kat']}/{f['n_kat']} | p(düzeltilmiş) {f['p_duzeltilmis']:.3f}")
    print("\nNot: asıl karşılaştırma 'Artirmali(tum)' − 'Artirmasiz'; tek tek dönüşümler ikincil/keşifseldir.")
    _yaz("artirma_fold.csv", rows)
    _yaz("artirma_ozet.csv", ozet)
    _yaz("artirma_eslestirilmis.csv", fark)
    print("Yazıldı: artirma_fold.csv, artirma_ozet.csv, artirma_eslestirilmis.csv")


if __name__ == "__main__":
    main()
