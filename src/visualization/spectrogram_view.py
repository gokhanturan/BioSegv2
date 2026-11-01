"""Spektrogram görünüm ayarları (renk haritası, kontrast, frekans aralığı). Qt'den bağımsız, test edilebilir.

Ayarlar yalnızca ekrandaki gösterimi değiştirir; hesaplanan spektrogram (önbellek) ve CSV'ye yazılan hiçbir değer değişmez.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .spectrogram import hz_to_mel

# (açılır liste metni, matplotlib adı)
COLORMAPS: Tuple[Tuple[str, str], ...] = (
    ("Magma (varsayılan)", "magma"),
    ("Gri tonlu (baskı için)", "gray_r"),
    ("Viridis", "viridis"),
    ("Inferno", "inferno"),
)
CONTRAST_PERCENTILE = 99.5          # üst sınır: değerlerin %99,5'i; tek bir şiddetli ses kontrastı bozmasın
MEL_TICKS_HZ = (0, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 16000, 20000, 24000)


def contrast_limits(S_db: np.ndarray, range_db: int) -> Tuple[Optional[float], Optional[float]]:
    """(vmin, vmax). range_db <= 0 -> (None, None): matplotlib tam aralığı kullanır (eski davranış).

    Aksi halde vmax = %99,5'lik değer, vmin = vmax - range_db. Böylece tek bir şiddetli ses
    (mikrofon çarpması gibi) diğer ötüşleri soluk göstermez."""
    if not range_db or range_db <= 0 or S_db is None or S_db.size == 0:
        return None, None
    vmax = float(np.percentile(S_db, CONTRAST_PERCENTILE))
    return vmax - float(range_db), vmax


def clean_freq_range(f_lo: float, f_hi: float, nyquist: float) -> Tuple[float, float]:
    """Geçersiz/ters girişte tüm bandı (0, Nyquist) döndürür; sınırları Nyquist'e kırpar. f_hi=0 -> tümü."""
    lo = max(0.0, float(f_lo or 0.0))
    hi = float(f_hi or 0.0)
    hi = nyquist if hi <= 0 else min(hi, nyquist)
    if lo >= hi:
        return 0.0, nyquist
    return lo, hi


def hz_to_mel_row(hz: float, nyquist: float, n_mels: float) -> float:
    """Mel spektrogramında (0..n_mels, mel ölçeğinde doğrusal) bir frekansın satır konumu."""
    return float(n_mels) * float(hz_to_mel(hz)) / float(hz_to_mel(nyquist))


def y_limits(mode: str, f_lo: float, f_hi: float, nyquist: float, n_mels: float = 128) -> Tuple[float, float]:
    """Eksen sınırları: 'spec' için Hz, 'melspec' için mel satırı."""
    lo, hi = clean_freq_range(f_lo, f_hi, nyquist)
    if mode == "melspec":
        return hz_to_mel_row(lo, nyquist, n_mels), hz_to_mel_row(hi, nyquist, n_mels)
    return lo, hi


def mel_ticks(f_lo: float, f_hi: float, nyquist: float, n_mels: float = 128) -> Tuple[Sequence[float], Sequence[str]]:
    """Mel ekseni için Hz etiketli işaretler (satır konumu, 'x kHz')."""
    lo, hi = clean_freq_range(f_lo, f_hi, nyquist)
    hz = [h for h in MEL_TICKS_HZ if lo <= h <= hi]
    pos = [hz_to_mel_row(h, nyquist, n_mels) for h in hz]
    lab = [f"{h / 1000:g} kHz" if h >= 1000 else f"{int(h)} Hz" for h in hz]
    return pos, lab
