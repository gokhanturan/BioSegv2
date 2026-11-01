"""
src/ml/embedding_extraction.py
================
Bir ses dosyasının tamamı için, Google DeepMind Perch 2.0'ın 5 saniyelik
pencereler üzerinden ürettiği embedding'lerin (1536 boyutlu ara temsil)
ORTALAMASINI (mean pooling) alarak tek bir sabit-boyutlu vektör üretir.

Segmentler genelde 5 saniyeden kısa olduğu için çoğu dosya tek bir
(sıfır-dolgulu) pencereye denk gelecektir; daha uzun segmentler için
birden fazla pencerenin ortalaması alınır.

NOT: Bu modül YALNIZCA Perch 2.0 gömmesini çıkarır. BirdNET 2.4 gömmesi
`birdnet_embedding.py` (`birdnet` paketi) ile çıkarılır; birleşik model her ikisini
birlikte kullanır (bkz. `model_registry.py`). Eski `birdnetlib` yolu ("Tensor data is
null. Run allocate_tensors() first" hatası) artık kullanılmamaktadır.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..detection.perch_classifier import PerchClassifier, PERCH_SAMPLE_RATE, generate_windows
from .dataset_builder import TrainingSample

logger = logging.getLogger("embedding_extraction")


def embedding_from_waveform(classifier: PerchClassifier, y: np.ndarray, sr: int = PERCH_SAMPLE_RATE,
                            hop_seconds: float = 2.5, source: str = "") -> np.ndarray:
    """Bellekteki bir dalga formu (32 kHz mono) için tek bir (1536,) embedding: pencerelerin ortalaması.
    Dosyadan okuma ile veri artırma (augmentation) aynı yolu kullansın diye ayrılmıştır."""
    if len(y) == 0:
        raise ValueError(f"Ses dosyası boş: {source}")

    windows = generate_windows(y, sr, hop_seconds=hop_seconds)
    if not windows:
        raise ValueError(f"Ses dosyasından pencere üretilemedi: {source}")

    embeddings = [classifier.predict_window_embedding(chunk) for _, _, chunk in windows]
    return np.mean(embeddings, axis=0)


def extract_embedding_for_file(classifier: PerchClassifier, filepath: str, hop_seconds: float = 2.5) -> np.ndarray:
    """Perch 2.0 ile bir ses dosyasının tamamı için tek bir (1536,) embedding vektörü döndürür."""
    import librosa

    y, sr = librosa.load(filepath, sr=PERCH_SAMPLE_RATE, mono=True)
    return embedding_from_waveform(classifier, y, sr, hop_seconds=hop_seconds, source=filepath)


def extract_embeddings_for_samples(
    classifier: PerchClassifier,
    samples: List[TrainingSample],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, List[TrainingSample], int, Optional[str]]:
    """
    Örnek listesinden embedding matrisi (N, 1536) ve BAŞARIYLA işlenen
    örneklerin listesini döndürür. Okunamayan/bozuk bir dosya varsa, o
    örnek sessizce atlanır (tüm işlem durdurulmaz) — sonuçta döndürülen
    `kept_samples` listesi, `X`'in satırlarıyla BİREBİR eşleşir.

    Ayrıca (n_failures, last_error_message) da döndürülür — TÜM örnekler
    başarısız olursa (örn. Perch 2.0 modeli yüklenemediyse), çağıran kod
    bunu anlamlı bir hata mesajına çevirebilsin diye.
    """
    X: List[np.ndarray] = []
    kept_samples: List[TrainingSample] = []
    n_failures = 0
    last_error: Optional[str] = None

    total = len(samples)
    for i, sample in enumerate(samples):
        try:
            emb = extract_embedding_for_file(classifier, sample.filepath)
            X.append(emb)
            kept_samples.append(sample)
        except Exception as exc:  # noqa: BLE001
            n_failures += 1
            last_error = str(exc)
            logger.warning("Embedding çıkarılamadı (%s): %s", sample.filepath, exc)

        if progress_callback is not None:
            progress_callback(i + 1, total)

    return np.array(X), kept_samples, n_failures, last_error
