"""
src/ml/input_format_control.py
=============================
Perch 2.0 girdi biçimi kontrolü: 3 sn'lik segmentin 5 sn'lik pencereye nasıl
yerleştirildiğinin gömme (embedding) kalitesine etkisini ölçmek için, aynı
segmentlerden farklı doldurma kipleriyle gömme üretir.

Kipler:
  "sifir"  : segment pencerenin başına konur, kalan kısım sıfırla doldurulur
             (uygulamadaki mevcut davranış).
  "tekrar" : segment 5 sn dolana kadar art arda tekrarlanır ve 5 sn'ye kırpılır.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..detection.perch_classifier import PERCH_SAMPLE_RATE, WINDOW_SAMPLES
from .dataset_builder import TrainingSample
from .embedding_extraction import embedding_from_waveform

logger = logging.getLogger("input_format_control")

KIPLER = ("sifir", "tekrar")


def fill_to_window(y: np.ndarray, mode: str, window_samples: int = WINDOW_SAMPLES) -> np.ndarray:
    """5 sn'den kısa dalga formunu seçilen kipe göre pencere uzunluğuna getirir.
    Pencere uzunluğuna eşit veya daha uzun sinyaller değiştirilmez."""
    if mode not in KIPLER:
        raise ValueError(f"Bilinmeyen kip: {mode!r} (geçerli: {KIPLER})")
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        raise ValueError("Dalga formu boş.")
    if n >= window_samples:
        return y
    if mode == "sifir":
        return np.pad(y, (0, window_samples - n))
    reps = int(np.ceil(window_samples / n))
    return np.tile(y, reps)[:window_samples]


def embedding_for_mode(classifier, y: np.ndarray, mode: str, sr: int = PERCH_SAMPLE_RATE,
                       hop_seconds: float = 2.5, source: str = "") -> np.ndarray:
    return embedding_from_waveform(classifier, fill_to_window(y, mode), sr,
                                   hop_seconds=hop_seconds, source=source)


def _keys(samples: List[TrainingSample]) -> np.ndarray:
    return np.array([f"{s.filepath}||{s.species}||{s.recording_id}" for s in samples])


def extract_perch_embeddings_mode(
    classifier,
    samples: List[TrainingSample],
    mode: str,
    cache_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """`samples` ile satır satır hizalı (N, 1536) Perch gömmesi. Herhangi bir dosya
    okunamazsa hata verir (kipler arası karşılaştırma aynı örneklerle yapılmalıdır)."""
    import librosa

    keys = _keys(samples)
    if cache_path and os.path.isfile(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        if "keys" in data.files and np.array_equal(data["keys"], keys):
            return data["X"]

    X: List[np.ndarray] = []
    total = len(samples)
    for i, s in enumerate(samples):
        y, _ = librosa.load(s.filepath, sr=PERCH_SAMPLE_RATE, mono=True)
        X.append(embedding_for_mode(classifier, y, mode, source=s.filepath))
        if progress_callback is not None:
            progress_callback(i + 1, total)
    X_arr = np.array(X)
    if cache_path:
        np.savez(cache_path, X=X_arr, keys=keys)
    return X_arr


def padding_effect_rows(rows, zero_name: str = "LR_Perch_sifir", tile_name: str = "LR_Perch_tekrar",
                        birdnet_name: str = "LR_BirdNET", metric: str = "makro_f1"):
    """Önceden tanımlı karşılaştırma çiftleri: birincil = doldurma etkisi (tekrar − sıfır);
    ikincil = her Perch kipi ile BirdNET farkı."""
    from .embedding_comparison import paired_difference

    out = []
    for label, a, b in (
        ("Perch tekrar − Perch sıfır", tile_name, zero_name),
        ("BirdNET − Perch sıfır", birdnet_name, zero_name),
        ("BirdNET − Perch tekrar", birdnet_name, tile_name),
    ):
        try:
            d = paired_difference(rows, a, b, metric)
        except (KeyError, ValueError):
            continue
        d = dict(d)
        d["karsilastirma"] = label
        out.append(d)
    return out
