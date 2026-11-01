"""
zenginlik_dogrulama.py
======================
Yapay zekâ (model) çıktılarından elde edilen tür zenginliğini, uzman etiketlerinden
elde edilen zenginlikle KARŞILAŞTIRIR (nokta düzeyinde).

Kapsam: girdi uzmanın ayırdığı segmentlerdir; ölçülen, sınıflandırma hatalarının tür
zenginliğine yansımasıdır. Her segmentin sınıflandırması, o kaydı eğitimde görmemiş bir modelden
gelir (kayıt bazlı çapraz doğrulama). Ayrıntı: src/ml/richness_validation.py

Kullanım (proje klasöründe, .venv etkinken):
    python zenginlik_dogrulama.py                      # Perch + MLP, 5 kat x 3 tekrar
    python zenginlik_dogrulama.py --yontem lr          # lojistik regresyon (hızlı)
    python zenginlik_dogrulama.py --yontem zeroshot    # Perch hazır çıktısı (TF + Perch gerekir)
    python zenginlik_dogrulama.py --esikler 0 0.5 0.7 0.9

Çıktılar: zenginlik_karsilastirma.csv (tekrar x eşik), zenginlik_ozet.csv (eşik başına ort ± s.s.),
          zenginlik_nokta.csv (nokta bazlı tür sayısı: uzman / model; ilk tekrar, eşik 0)
Veri setine ve modele DOKUNMAZ (salt-okunur).
"""
import argparse
import csv
import os

import pandas as pd

from src.ml.pipeline import load_or_extract_embeddings
from src.ml import richness_validation as RV


def _yaz(name, rows):
    with open(name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yontem", choices=["mlp", "lr", "zeroshot"], default="mlp")
    ap.add_argument("--kat", type=int, default=5)
    ap.add_argument("--tekrar", type=int, default=3)
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--min-segment", type=int, default=30)
    ap.add_argument("--min-kayit", type=int, default=2)
    ap.add_argument("--esikler", type=float, nargs="+", default=[0.0, 0.5, 0.7, 0.9])
    ap.add_argument("--csv", default=None, help="veri seti CSV (varsayılan: proje CSV'si)")
    args = ap.parse_args()

    X, kept = load_or_extract_embeddings(
        min_samples_per_class=args.min_segment, min_recordings_per_class=args.min_kayit,
        csv_path=args.csv, cache_path="embedding_onbellek.npz",
        progress_callback=lambda st, c, t: print(f"\r{st}: {c}/{t}", end="", flush=True))
    print(f"\nÖrnek: {len(kept)} segment, {len({s.recording_id for s in kept})} kayıt, "
          f"{len({s.species for s in kept})} tür")

    csv_path = args.csv
    if csv_path is None:
        from src.ml.dataset_builder import get_csv_root
        csv_path = os.path.join(get_csv_root(), "bird_sound_dataset.csv")
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    point_of_recording = {str(k).strip(): str(v).strip() for k, v in
                          df.dropna(subset=["Original_Recording_Name", "Habitat_ID"])
                            .drop_duplicates("Original_Recording_Name")
                            [["Original_Recording_Name", "Habitat_ID"]].itertuples(index=False)}
    n_miss = len({s.recording_id for s in kept} - set(point_of_recording))
    if n_miss:
        print(f"⚠ {n_miss} kaydın Habitat_ID değeri yok; bu kayıtlar karşılaştırmaya girmez.")

    zs = None
    class_names = sorted({s.species for s in kept})
    if args.yontem == "zeroshot":
        from src.ml import baselines
        zs = baselines.compute_zero_shot_scores(kept, class_names, cache_path="sifir_atis_onbellek.npz",
                                                progress_callback=lambda c, t: print(f"\rSıfır-atış: {c}/{t}", end="", flush=True))
        print()

    classes, preds = RV.out_of_fold_predictions(
        X, kept, method=args.yontem, n_splits=args.kat, n_repeats=args.tekrar,
        max_epochs=args.epoch, zero_shot_scores=zs,
        progress_callback=lambda d, t: print(f"\rSınıflandırma: {d}/{t}", end="", flush=True))
    print()
    rows = RV.evaluate_richness_agreement(kept, point_of_recording, classes, preds, thresholds=args.esikler)
    ozet = RV.summarize_by_threshold(rows)

    print(f"\nYöntem: {args.yontem} | nokta sayısı: {rows[0]['nokta_sayisi']}")
    for o in ozet:
        print(f"eşik {o['guven_esigi']:.2f}: tespit oranı {o['tespit_orani_ort']:.2f} | "
              f"tür uzman {o['tur_uzman_ort']:.0f} / model {o['tur_model_ort']:.1f} | "
              f"nokta başına tür Spearman {o['nokta_basina_tur_spearman_ort']:.2f} | "
              f"Jaccard {o['var_yok_jaccard_ort']:.2f} (kesinlik {o['var_yok_kesinlik_ort']:.2f}, "
              f"duyarlılık {o['var_yok_duyarlilik_ort']:.2f})")

    _yaz("zenginlik_karsilastirma.csv", rows)
    _yaz("zenginlik_ozet.csv", ozet)
    _yaz("zenginlik_nokta.csv", RV.per_point_table(kept, point_of_recording, classes, preds[0], 0.0))
    print("Yazıldı: zenginlik_karsilastirma.csv, zenginlik_ozet.csv, zenginlik_nokta.csv")


if __name__ == "__main__":
    main()
