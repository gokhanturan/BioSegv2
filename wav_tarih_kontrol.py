"""
WAV dosyalarındaki kayıt tarihi bilgisini kontrol eder.

Kullanım (proje klasöründe):
    python wav_tarih_kontrol.py "C:\\yol\\dosya1.wav" "C:\\yol\\dosya2.wav"

Her dosya için: WAV içindeki bölüm (chunk) adları, WAV içinden okunabilen kayıt
tarihi (varsa) ve dosya sisteminin oluşturma/değiştirme zamanları yazdırılır.
"""
import os
import struct
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.audio.utils import read_wav_recording_datetime


def list_chunks(path):
    ids = []
    with open(path, "rb") as f:
        if f.read(4) not in (b"RIFF", b"RF64"):
            return ["(RIFF değil)"]
        f.read(8)
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            cid, size = struct.unpack("<4sI", h)
            ids.append(cid.decode("ascii", "replace"))
            if size == 0xFFFFFFFF:
                break
            f.seek(size + (size & 1), 1)
    return ids


for p in sys.argv[1:]:
    print("=" * 70)
    print(p)
    try:
        print("  WAV bölümleri      :", ", ".join(list_chunks(p)))
    except Exception as e:  # noqa: BLE001
        print("  WAV bölümleri      : okunamadı:", e)
    dt, kaynak = read_wav_recording_datetime(p)
    print("  WAV içindeki tarih :", dt if dt else "BULUNAMADI", f"({kaynak})" if kaynak else "")
    st = os.stat(p)
    print("  Oluşturma (ctime)  :", datetime.fromtimestamp(st.st_ctime))
    print("  Değiştirme (mtime) :", datetime.fromtimestamp(st.st_mtime))
