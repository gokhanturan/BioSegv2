"""Ses G/Ç (okuma/yazma) yardımcı fonksiyonları."""

import os
import re
import struct
import wave
import numpy as np
from datetime import datetime, timedelta
from scipy.io.wavfile import write as wavwrite


_DATETIME_PATTERNS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M",
    "%Y/%m/%d %H:%M:%S", "%Y.%m.%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
)


def _parse_datetime_text(text):
    """Metinden (örn. '2026-08-24' ya da '2026-08-24T07:15:00') datetime üretir; olmazsa None."""
    text = (text or "").replace("\x00", "").strip()
    if not text:
        return None
    for pattern in _DATETIME_PATTERNS:
        try:
            return datetime.strptime(text[:len(datetime.now().strftime(pattern))], pattern)
        except ValueError:
            continue
    return None


def _plausible(dt):
    """Makul bir kayıt tarihi mi? (2000 sonrası, yarından önce)"""
    return dt is not None and datetime(2000, 1, 1) <= dt <= datetime.now() + timedelta(days=1)


def read_wav_recording_datetime(file_path):
    """
    WAV dosyasının İÇİNE kaydedici tarafından yazılmış kayıt tarih/saatini okur.

    Sırayla şunlara bakar:
      1. BWF 'bext' bölümü (OriginationDate + OriginationTime; Zoom gibi saha
         kaydedicilerinin çoğu bunu yazar),
      2. 'LIST/INFO' bölümündeki ICRD (oluşturulma tarihi) alanı,
      3. 'LIST/INFO' ICMT (yorum) alanındaki AudioMoth biçimi
         ('Recorded at 07:15:00 24/08/2026 (UTC+3) ...').

    Dönüş: (datetime, kaynak_adı) ya da bulunamazsa (None, None).
    Asla hata fırlatmaz; okunamayan/bozuk başlıkta (None, None) döner.
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)
            if len(header) < 12 or header[:4] not in (b"RIFF", b"RF64") or header[8:12] != b"WAVE":
                return None, None
            bext_dt = None
            icrd_dt = None
            icmt_dt = None
            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break
                chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
                if chunk_id == b"data":
                    # Ses verisini atla (RF64'te boyut 0xFFFFFFFF olabilir, bu durumda dur).
                    if chunk_size == 0xFFFFFFFF:
                        break
                    f.seek(chunk_size + (chunk_size & 1), 1)
                    continue
                if chunk_size > 16 * 1024 * 1024:   # beklenmedik büyük bölüm: güvenle atla
                    f.seek(chunk_size + (chunk_size & 1), 1)
                    continue
                body = f.read(chunk_size)
                if chunk_size & 1:
                    f.read(1)

                if chunk_id == b"bext" and len(body) >= 338:
                    # Description(256) Originator(32) OriginatorReference(32) -> tarih 10, saat 8
                    date_txt = body[320:330].decode("ascii", "ignore")
                    time_txt = body[330:338].decode("ascii", "ignore")
                    bext_dt = _parse_datetime_text(f"{date_txt} {time_txt}") or _parse_datetime_text(date_txt)
                elif chunk_id == b"LIST" and body[:4] == b"INFO":
                    pos = 4
                    while pos + 8 <= len(body):
                        sub_id = body[pos:pos + 4]
                        sub_size = struct.unpack("<I", body[pos + 4:pos + 8])[0]
                        value = body[pos + 8:pos + 8 + sub_size].decode("utf-8", "ignore")
                        if sub_id == b"ICRD":
                            icrd_dt = _parse_datetime_text(value)
                        elif sub_id == b"ICMT":
                            m = re.search(r"(\d{2}:\d{2}:\d{2})\s+(\d{2})/(\d{2})/(\d{4})", value)
                            if m:
                                icmt_dt = _parse_datetime_text(f"{m.group(4)}-{m.group(3)}-{m.group(2)} {m.group(1)}")
                        pos += 8 + sub_size + (sub_size & 1)

            for dt, source in ((bext_dt, "wav_bext"), (icrd_dt, "wav_info_icrd"), (icmt_dt, "wav_info_icmt")):
                if _plausible(dt):
                    return dt, source
    except Exception:  # noqa: BLE001
        return None, None
    return None, None


def get_audio_file_info(file_path):
    """Bir WAV dosyasından üstveriyi (boyut, süre, örnekleme hızı, kanal vb.) çıkarır."""
    try:
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        file_size_mb = file_size / (1024 * 1024)

        # KAYIT TARİHİ: Önce ses dosyasının İÇİNDE kaydedicinin yazdığı tarihe bakılır.
        # Bulunamazsa dosyanın "değiştirme" zamanına (st_mtime) düşülür: Windows'ta dosya
        # kopyalanırken genellikle korunur (kayıt cihazından kopyalanan dosyada kaydın bittiği
        # anı yansıtır). st_ctime KULLANILMAZ: Windows'ta bu, dosyanın bilgisayara
        # KOPYALANDIĞI zamandır ve gerçek kayıt tarihi değildir.
        creation_time, date_source = read_wav_recording_datetime(file_path)
        if creation_time is None:
            creation_time = datetime.fromtimestamp(file_stats.st_mtime)
            date_source = "file_mtime"

        with wave.open(file_path, 'rb') as wav_file:
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            duration = num_frames / float(frame_rate)

            bit_depth_map = {1: "8 bit", 2: "16 bit", 3: "24 bit", 4: "32 bit"}
            bit_depth = bit_depth_map.get(sample_width, f"{sample_width * 8} bit")
            channels = "Mono" if num_channels == 1 else "Stereo" if num_channels == 2 else f"{num_channels} Channels"
            file_format = os.path.splitext(file_path)[1].upper().replace('.', '')

        return {
            'file_size': file_size,
            'file_size_mb': file_size_mb,
            'channels': channels,
            'num_channels': num_channels,
            'sample_rate': frame_rate,
            'bit_depth': bit_depth,
            'sample_width': sample_width,
            'duration': duration,
            'num_frames': num_frames,
            'creation_time': creation_time,
            'file_path': file_path,
            'file_format': file_format,
            'recording_date': creation_time.strftime('%Y-%m-%d'),
            'recording_time': creation_time.strftime('%H:%M:%S'),
            'recording_datetime_source': date_source,
        }
    except Exception as e:
        print(f"Dosya bilgisi alınamadı: {e}")
        return None


def write_wav_safe(path: str, y: np.ndarray, sr: int):
    """Bir WAV dosyasını güvenli biçimde 16-bit PCM olarak yazar."""
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    # Kayan noktalı sesi int16'ya dönüştür
    if np.issubdtype(y.dtype, np.floating):
        y = np.clip(y, -1.0, 1.0)
        y_int16 = (y * 32767.0).astype(np.int16)

    # Tamsayı sesi int16'ya dönüştür
    elif np.issubdtype(y.dtype, np.integer):
        if y.dtype != np.int16:
            max_val = np.iinfo(y.dtype).max
            y_float = y.astype(np.float32) / max_val
            y_int16 = (y_float * 32767.0).astype(np.int16)
        else:
            y_int16 = y.astype(np.int16)

    # Diğer türler için yedek yol
    else:
        y = y.astype(np.float32)
        y = np.clip(y, -1.0, 1.0)
        y_int16 = (y * 32767.0).astype(np.int16)

    wavwrite(path, sr, y_int16)