"""L/M kanal çiftlerinin segmentleri zaman olarak çakışıyor mu? (salt okunur)

Kullanım (proje kökünden):  python lm_cakisma_kontrolu.py [--csv bird_sound_dataset.csv]
Çıktı ekrana yazılır ve lm_cakisma.csv olarak kaydedilir. Veriye dokunmaz.
"""
import argparse
import os
import sys

import pandas as pd

from src.data import recording_groups as rg
from src.data.csv_handler import get_project_root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=os.path.join(get_project_root(), "bird_sound_dataset.csv"))
    a = ap.parse_args(argv)
    if not os.path.isfile(a.csv):
        print(f"CSV bulunamadı: {a.csv}")
        return 1
    df = pd.read_csv(a.csv, sep=";", encoding="utf-8-sig", dtype=str)
    rep = rg.lm_overlap_report(df)
    if rep.empty:
        print("Birleştirilen L/M çifti yok (ya da Start_Time/End_Time sütunları eksik).")
        return 0
    print(rep.to_string(index=False))
    print(f"\nToplam: L={rep.L_segment.sum()}, M={rep.M_segment.sum()}, zamanı çakışan çift={rep.Cakisan_cift.sum()}, "
          f"aynı türde çakışan={rep.Cakisan_ayni_tur.sum()}, aynı pencere={rep.Ayni_pencere.sum()}")
    rep.to_csv("lm_cakisma.csv", sep=";", index=False, encoding="utf-8-sig")
    print("Yazıldı: lm_cakisma.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
