"""Kapsam raporu ve etiketleme iş listesi.

Kullanım (proje kökünden):
    python kapsam_raporu.py [--csv bird_sound_dataset.csv] [--cikti etiketlenmemis_noktalar.csv]

Tasarım (örnekleme noktaları) nokta kataloğundan (src/habitat_data.json) veya
zenginlik_ayarlar.json içindeki planned_points_csv dosyasından okunur. Başka bir
çalışmaya geçmek için bu dosyayı değiştirmek yeterlidir; kodda sabit değer yoktur.
"""
import argparse
import os
import sys

import pandas as pd

from src.data import coverage_report as cov
from src.data.csv_handler import get_project_root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(get_project_root(), "bird_sound_dataset.csv"))
    ap.add_argument("--cikti", default="etiketlenmemis_noktalar.csv")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.csv):
        print(f"CSV bulunamadı: {a.csv}")
        return 1
    df = pd.read_csv(a.csv, sep=";", encoding="utf-8-sig")
    cfg = cov.load_project_config()
    if cfg.get("habitat_groups"):
        df = cov.apply_habitat_groups(df, cfg["habitat_groups"])
    print(cov.build_coverage_report(df, cfg).format())
    todo = cov.unlabeled_points(df, cfg)
    todo.to_csv(a.cikti, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nEtiketlenmemiş {len(todo)} nokta -> {a.cikti}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
