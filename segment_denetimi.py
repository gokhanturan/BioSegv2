"""
segment_denetimi.py
===================
Veri setindeki segment WAV dosyalarının teknik özelliklerini ölçer (örnekleme hızı, kanal, süre,
ses seviyesi, spektral üst sınır). Hiçbir şeyi değiştirmez, hiçbir eşik uygulamaz; yalnızca dağılımı raporlar.

Kullanım (proje klasöründe, .venv etkinken):
    python segment_denetimi.py

Çıktılar: segment_denetim.csv (dosya başına), segment_denetim_ozet.csv (tür başına ve TÜMÜ)
"""
import argparse
import csv

from src.ml.dataset_builder import build_training_samples
from src.ml import segment_audit as sa


def _yaz(name, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter=";")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--segment-klasoru", default=None)
    args = ap.parse_args()

    samples, _, _ = build_training_samples(csv_path=args.csv, classification_dir=args.segment_klasoru,
                                           min_samples_per_class=1, min_recordings_per_class=1)
    print(f"{len(samples)} segment dosyası bulundu; ölçülüyor...")
    species_of = {s.filepath: s.species for s in samples}
    rows = []
    for i, s in enumerate(samples, 1):
        rows.append(sa.audit_file(s.filepath))
        if i % 100 == 0:
            print(f"  {i}/{len(samples)}")
    for r in rows:
        r["tur"] = species_of.get(r["dosya"], "?")
    ozet = sa.summarize(rows, species_of)
    _yaz("segment_denetim.csv", rows)
    _yaz("segment_denetim_ozet.csv", ozet)

    t = ozet[0]
    print("\n=== TÜMÜ ===")
    for k, v in t.items():
        print(f"  {k}: {v}")
    print("\nDosya başına: segment_denetim.csv · tür başına özet: segment_denetim_ozet.csv")


if __name__ == "__main__":
    main()
