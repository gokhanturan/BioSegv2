"""CSV'deki 'Project' sütununu Türkçe proje adına çevirir. Başka hiçbir sütuna/satıra dokunmaz.

Kullanım (proje klasöründe, program KAPALIYKEN):
    python csv_proje_adi_duzelt.py                 -> yalnızca rapor (dosyaya yazmaz)
    python csv_proje_adi_duzelt.py --uygula        -> yedek alır, sonra yazar
    python csv_proje_adi_duzelt.py --csv yol\\dosya.csv --uygula
Yedek: bird_sound_dataset_YYYYMMDD_HHMMSS_proje_oncesi.csv (aynı klasörde).
"""
import argparse
import csv
import os
import shutil
import sys
from collections import Counter
from datetime import datetime

YENI = "TÜBİTAK 1001 - Proje No: 123O927 - Kuş Türü Fonksiyonel Çeşitliliğinin Hesaplanması, Modellenmesi ve Haritalanması"
SUTUN = "Project"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="bird_sound_dataset.csv")
    ap.add_argument("--uygula", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(a.csv):
        sys.exit(f"CSV bulunamadı: {os.path.abspath(a.csv)}")
    with open(a.csv, "rb") as f:
        ham = f.read()
    bom = ham.startswith(b"\xef\xbb\xbf")
    crlf = b"\r\n" in ham
    metin = ham.decode("utf-8-sig")
    satirlar = list(csv.reader(metin.splitlines(keepends=False), delimiter=";")) if False else None
    import io
    rows = list(csv.reader(io.StringIO(metin, newline=""), delimiter=";"))
    if not rows or SUTUN not in rows[0]:
        sys.exit(f"'{SUTUN}' sütunu bulunamadı.")
    k = rows[0].index(SUTUN)
    mevcut = Counter(r[k] for r in rows[1:] if len(r) > k)
    print(f"Dosya: {os.path.abspath(a.csv)}  ({len(rows) - 1} satır)")
    print("Şu anki 'Project' değerleri:")
    for v, n in mevcut.most_common():
        print(f"  {n:>5} x  {v!r}")
    degisecek = sum(n for v, n in mevcut.items() if v != YENI)
    print(f"\nDeğişecek satır: {degisecek}  |  Zaten doğru: {len(rows) - 1 - degisecek}")
    if not a.uygula:
        print("\n(Rapor modu: dosyaya yazılmadı. Uygulamak için --uygula ekleyin.)")
        return
    if degisecek == 0:
        print("Yapılacak bir şey yok.")
        return
    yedek = os.path.splitext(a.csv)[0] + datetime.now().strftime("_%Y%m%d_%H%M%S") + "_proje_oncesi.csv"
    shutil.copy2(a.csv, yedek)
    for r in rows[1:]:
        if len(r) > k:
            r[k] = YENI
    tmp = a.csv + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
        csv.writer(f, delimiter=";", lineterminator="\r\n" if crlf else "\n").writerows(rows)
    # Doğrulama: yalnızca Project sütunu farklı olmalı
    with open(tmp, "r", encoding="utf-8-sig", newline="") as f:
        yeni_rows = list(csv.reader(f, delimiter=";"))
    assert len(yeni_rows) == len(rows)
    for eski, yeni in zip(csv.reader(io.StringIO(metin, newline=""), delimiter=";"), yeni_rows):
        for i, (x, y) in enumerate(zip(eski, yeni)):
            assert i == k or x == y, "Project dışındaki bir hücre değişti; işlem iptal."
    os.replace(tmp, a.csv)
    print(f"\nTamam. {degisecek} satır güncellendi.\nYedek: {os.path.abspath(yedek)}")


if __name__ == "__main__":
    main()
