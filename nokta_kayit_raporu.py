"""Nokta başına kayıt süresi raporu (salt okunur).

Kullanım (proje kökünden):
    python nokta_kayit_raporu.py [--csv bird_sound_dataset.csv] [--cikti-klasoru .]

Çıktılar: nokta_kayit_suresi.csv (nokta bazlı), nokta_kayit_ayrintisi.csv (kayıt bazlı),
nokta_kayit_raporu.txt (aynı metin). Veri setine ve modellere DOKUNMAZ.
"""
import argparse
import os
import sys

import pandas as pd

from src.data import point_effort_report as per
from src.data.csv_handler import get_project_root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(get_project_root(), "bird_sound_dataset.csv"))
    ap.add_argument("--cikti-klasoru", default=".")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.csv):
        print(f"CSV bulunamadı: {a.csv}")
        return 1
    df = pd.read_csv(a.csv, sep=";", encoding="utf-8-sig", dtype=str)
    rep = per.build_effort_report(df)
    text = rep.format()
    print(text)
    os.makedirs(a.cikti_klasoru, exist_ok=True)
    files = per.write_csvs(a.cikti_klasoru, rep)
    txt = os.path.join(a.cikti_klasoru, "nokta_kayit_raporu.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    files.append("nokta_kayit_raporu.txt")
    print("\nYazılan dosyalar:", ", ".join(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
