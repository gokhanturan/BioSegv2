
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from .constants import TARGET_LOUDNESS_LUFS, NOISE_REDUCTION_N_FFT, NOISE_REDUCTION_WIN_LENGTH

# Yedekli (isteğe bağlı) kütüphaneler
try:
    import noisereduce as nr
except ImportError:
    nr = None

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None


class AudioProcessThread(QThread):
    """Ses işlemeyi (gürültü azaltma, LUFS normalizasyonu) arka planda yürüten QThread sınıfı."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, y, sr, t0, t1, do_denoise, do_lufs, target_lufs=TARGET_LOUDNESS_LUFS, parent=None):
        super().__init__(parent)
        self.y = y
        self.sr = sr
        self.t0 = max(0.0, float(t0))
        self.t1 = max(self.t0, float(t1))
        self.do_denoise = do_denoise
        self.do_lufs = do_lufs
        self.target_lufs = float(target_lufs)

    def run(self):
        """Ses bölümüne gürültü azaltma ve/veya LUFS normalizasyonu uygular."""
        try:
            # Girdiyi doğrula
            if self.y is None or self.sr is None:
                raise RuntimeError("Önce bir ses dosyası yükleyin.")

            # Başlangıç ve bitiş örnek (sample) indekslerini hesapla
            start = int(self.t0 * self.sr)
            end = int(self.t1 * self.sr) if self.t1 > self.t0 else len(self.y)

            start = max(0, min(start, len(self.y)))
            end = max(start, min(end, len(self.y)))

            # Boş veya çok kısa aralıkları ele al
            if start >= end or (end - start) < self.sr * 0.01:
                self.finished.emit({
                    'y_out': self.y[start:end].copy().astype(np.float32),
                    't0': self.t0,
                    't1': self.t1,
                    'msg': 'Aralık çok kısa (işlem yapılmadı)'
                })
                return

            original_length = end - start

            # Ses bölümünü çıkar
            try:
                y_work = self.y[start:end].copy().astype(np.float32)
            except IndexError:
                raise RuntimeError(f"Ses dizisi dilimleme hatası: Başlangıç={start}, Bitiş={end}, Toplam={len(self.y)}")

            # Dilimlemeden sonra uzunluğun tutarlı olduğundan emin ol
            if len(y_work) != original_length:
                if len(y_work) > original_length:
                    y_work = y_work[:original_length]
                else:
                    y_work = np.pad(y_work, (0, original_length - len(y_work)), mode='constant')

            self.progress.emit(5)

            # Hiçbir işlem istenmediyse atla
            if not self.do_denoise and not self.do_lufs:
                self.finished.emit({'y_out': y_work, 't0': self.t0, 't1': self.t1, 'msg': 'İşlem yapılmadı'})
                return

            # Gürültü azaltmayı uygula
            if self.do_denoise:
                if nr is None:
                    raise RuntimeError("noisereduce kurulu değil (pip install noisereduce).")

                n_fft_safe = NOISE_REDUCTION_N_FFT
                win_length_safe = NOISE_REDUCTION_WIN_LENGTH

                if len(y_work) < n_fft_safe:
                    print("Uyarı: Ses bölümü gürültü azaltma için çok kısa")
                else:
                    # Gürültüyü bölümün ilk kısmından tahmin et (1 saniye veya %10)
                    noise_len_s = min(int(1 * self.sr), len(y_work) // 10)
                    y_noise = y_work[:noise_len_s] if noise_len_s > 0 else y_work

                    original_len = len(y_work)
                    y_work = nr.reduce_noise(
                        y=y_work,
                        y_noise=y_noise,
                        sr=self.sr,
                        prop_decrease=0.9,
                        stationary=False,
                        n_fft=n_fft_safe,
                        win_length=win_length_safe
                    ).astype(np.float32)

                    # Uzunluk değiştiyse özgün uzunluğa geri getir
                    if len(y_work) != original_len:
                        if len(y_work) > original_len:
                            y_work = y_work[:original_len]
                        else:
                            y_work = np.pad(y_work, (0, original_len - len(y_work)), mode='constant')

                self.progress.emit(45)

            # LUFS ses şiddeti normalizasyonunu uygula
            if self.do_lufs:
                if pyln is None:
                    raise RuntimeError("pyloudnorm kurulu değil (pip install pyloudnorm).")

                if len(y_work) > 0:
                    meter = pyln.Meter(self.sr)
                    try:
                        current_loudness = meter.integrated_loudness(y_work.astype(float))
                        gain = self.target_lufs - current_loudness
                        linear_gain = 10 ** (gain / 20.0)

                        original_len = len(y_work)
                        # Kazancı uygula ve [-1.0, 1.0] aralığına kırp
                        y_work = np.clip(y_work * linear_gain, -1.0, 1.0).astype(np.float32)

                        # Uzunluk değiştiyse özgün uzunluğa geri getir
                        if len(y_work) != original_len:
                            if len(y_work) > original_len:
                                y_work = y_work[:original_len]
                            else:
                                y_work = np.pad(y_work, (0, original_len - len(y_work)), mode='constant')
                    except Exception as e:
                        print(f"LUFS normalizasyonu hatası: {e}")

                self.progress.emit(80)

            self.finished.emit({'y_out': y_work, 't0': self.t0, 't1': self.t1, 'msg': "İşlem tamamlandı."})

        except Exception as e:
            import traceback
            print(f"KRİTİK İŞLEME HATASI: {e}")
            traceback.print_exc()
            self.error.emit(str(e))
