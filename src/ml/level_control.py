"""
src/ml/level_control.py
=======================
Deney 3 (ses seviyesi kontrolü): segmentlerin ses seviyesi, sınıflandırma sonucunu açıklayan bir kısayol mu?

Önceden tanımlı kollar:
  BirdNET        : mevcut segmentlerin BirdNET gömmeleri
  BirdNET_rms    : her segment, veri setinin medyan RMS seviyesine ölçeklenir (tepe 0.99'u aşarsa tepeye göre kısılır)
  BirdNET_tepe   : her segment tepe -3 dBFS olacak biçimde ölçeklenir
  Seviye         : yalnızca [RMS dBFS, tepe dBFS] özellikleriyle sınıflandırma
  Sabit          : tek sabit özellik (özelliksiz taban çizgisi)
"""

from __future__ import annotations

import os
import shutil
from typing import Callable, List, Optional

import numpy as np

from .dataset_builder import TrainingSample

KIPLER = ("rms", "tepe")
TEPE_HEDEF_DBFS = -3.0
TAVAN = 0.99
SESSIZ_DBFS = -120.0


def _db(x: float) -> float:
    return 20.0 * np.log10(x) if x > 0 else SESSIZ_DBFS


def _read_mono(path: str):
    import soundfile as sf

    y, sr = sf.read(path, always_2d=True, dtype="float64")
    return y.mean(axis=1), int(sr)


def measure_level(y: np.ndarray):
    """(RMS dBFS, tepe dBFS); sessiz sinyalde SESSIZ_DBFS."""
    if len(y) == 0:
        return SESSIZ_DBFS, SESSIZ_DBFS
    return _db(float(np.sqrt(np.mean(y ** 2)))), _db(float(np.abs(y).max()))


def level_features(samples: List[TrainingSample]) -> np.ndarray:
    """(N, 2) = [RMS dBFS, tepe dBFS]."""
    return np.array([measure_level(_read_mono(s.filepath)[0]) for s in samples], dtype=float)


def median_rms_dbfs(samples: List[TrainingSample]) -> float:
    return float(np.median(level_features(samples)[:, 0]))


def normalize_waveform(y: np.ndarray, mode: str, target_rms_dbfs: Optional[float] = None) -> np.ndarray:
    if mode not in KIPLER:
        raise ValueError(f"Bilinmeyen kip: {mode!r} (geçerli: {KIPLER})")
    y = np.asarray(y, dtype=np.float64)
    peak = float(np.abs(y).max()) if len(y) else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if peak == 0.0 or rms == 0.0:
        return y
    if mode == "tepe":
        return y * (10 ** (TEPE_HEDEF_DBFS / 20.0) / peak)
    if target_rms_dbfs is None:
        raise ValueError("'rms' kipi için target_rms_dbfs gerekir.")
    out = y * (10 ** (target_rms_dbfs / 20.0) / rms)
    p = float(np.abs(out).max())
    if p > TAVAN:
        out = out * (TAVAN / p)
    return out


def write_normalized_copies(samples: List[TrainingSample], mode: str, out_dir: str,
                            target_rms_dbfs: Optional[float] = None) -> List[TrainingSample]:
    """Normalize edilmiş kopyaları (float32 WAV, özgün örnekleme hızında) yazar; tür ve kayıt kimliği korunur."""
    import soundfile as sf

    os.makedirs(out_dir, exist_ok=True)
    out = []
    for i, s in enumerate(samples):
        y, sr = _read_mono(s.filepath)
        path = os.path.join(out_dir, f"{mode}_{i:05d}.wav")
        sf.write(path, normalize_waveform(y, mode, target_rms_dbfs).astype(np.float32), sr, subtype="FLOAT")
        out.append(TrainingSample(filepath=path, species=s.species, recording_id=s.recording_id))
    return out


def embeddings_for_mode(samples: List[TrainingSample], mode: str, cache_path: Optional[str] = None,
                        tmp_dir: str = "seviye_gecici", library: str = "litert",
                        target_rms_dbfs: Optional[float] = None,
                        progress_callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Normalize edilmiş segmentlerin BirdNET gömmeleri, `samples` ile satır satır hizalı."""
    from . import birdnet_embedding as bne
    from .embedding_comparison import _load_cached, sample_keys

    cached = _load_cached(cache_path, samples)
    if cached is not None:
        return cached
    if mode == "rms" and target_rms_dbfs is None:
        target_rms_dbfs = median_rms_dbfs(samples)
    folder = os.path.join(tmp_dir, mode)
    try:
        copies = write_normalized_copies(samples, mode, folder, target_rms_dbfs)
        X, kept, n_fail, last_error = bne.extract_embeddings_for_samples(
            copies, library=library, progress_callback=progress_callback)
        if len(kept) != len(samples):
            raise RuntimeError(f"{n_fail} dosyadan gömme çıkarılamadı (son hata: {last_error}); "
                               "kollar aynı örneklerle karşılaştırılmalı.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if cache_path:
        np.savez(cache_path, X=X, keys=sample_keys(samples))
    return X
