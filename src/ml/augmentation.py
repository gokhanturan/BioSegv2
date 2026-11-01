"""
src/ml/augmentation.py
======================
Eğitim kümesi için SES düzeyinde veri artırma (augmentation) deneyi.

İlkeler (deney sonuçlara bakılmadan, önceden belirlenmiştir):
  1. Artırma YALNIZCA eğitim katlarına uygulanır. Test katı her zaman özgün segmentlerden oluşur.
     Bölme kayıt-gruplu olduğundan, bir eğitim segmentinin kopyası ile aynı kaydın test segmenti
     asla karşılaşmaz (run_augmentation_comparison bunu ayrıca denetler).
  2. Yalnızca etiketi bozmayan dönüşümler: ses seviyesi, zaman kaydırma, sentetik pembe gürültü.
     Perde kaydırma / güçlü zaman uzatma KULLANILMAZ (tür kimliği frekansa bağlıdır).
     Başka kayıttan ses karıştırma KULLANILMAZ (test sesinin eğitime sızma ve yabancı tür karışma riski).
  3. Aynı bölmelerde "artırmasız" ile "artırmalı" eşleştirilmiş karşılaştırılır (Nadeau-Bengio
     düzeltmeli p). Fark yoksa "fark bulunmadı" olarak raporlanır.
  4. Her dönüşüm her örnek için TEK kopya üretir; kopyalar örnek anahtarı + dönüşüm + tohumdan
     deterministik türetilir (yeniden çalıştırmada aynı ses).

Gömme çıkarımı Perch 2.0 gerektirir (TensorFlow); kopyaların gömmeleri önbelleğe yazılır.
"""
from __future__ import annotations

import hashlib
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

DEFAULT_KINDS = ("gain", "shift", "noise")
SR = 32_000

GAIN_DB_RANGE = (-6.0, 6.0)
SHIFT_S_RANGE = (-0.5, 0.5)
SNR_DB_RANGE = (15.0, 30.0)


def _rng(key: str, kind: str, seed: int) -> np.random.Generator:
    h = hashlib.sha256(f"{key}|{kind}|{seed}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f güç spektrumlu (pembe) gürültü, birim RMS."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec = spec / np.sqrt(f)
    spec[0] = 0.0
    x = np.fft.irfft(spec, n)
    rms = np.sqrt(np.mean(x ** 2))
    return x / rms if rms > 0 else x


def augment_waveform(y: np.ndarray, kind: str, key: str, seed: int = 0, sr: int = SR) -> np.ndarray:
    """Tek dönüşüm uygular; çıktı aynı uzunlukta float32. `key` kopyayı deterministik kılar."""
    y = np.asarray(y, dtype=np.float32)
    rng = _rng(key, kind, seed)
    if kind == "gain":
        db = rng.uniform(*GAIN_DB_RANGE)
        out = np.clip(y * (10.0 ** (db / 20.0)), -1.0, 1.0)
    elif kind == "shift":
        k = int(round(rng.uniform(*SHIFT_S_RANGE) * sr))
        out = np.zeros_like(y)
        if k >= 0:
            out[k:] = y[:len(y) - k] if k < len(y) else 0.0
        else:
            out[:k] = y[-k:] if -k < len(y) else 0.0
    elif kind == "noise":
        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        if rms == 0.0:
            return y.copy()
        snr = rng.uniform(*SNR_DB_RANGE)
        noise = _pink_noise(len(y), rng) * rms / (10.0 ** (snr / 20.0))
        out = np.clip(y + noise.astype(np.float32), -1.0, 1.0)
    else:
        raise ValueError(f"Bilinmeyen dönüşüm: {kind} (geçerli: {', '.join(DEFAULT_KINDS)})")
    return out.astype(np.float32)


def _keys(kept_samples) -> np.ndarray:
    return np.array([f"{s.filepath}||{s.species}||{s.recording_id}" for s in kept_samples])


def extract_augmented_embeddings(kept_samples, kinds: Sequence[str] = DEFAULT_KINDS, seed: int = 0,
                                 classifier=None, cache_path: Optional[str] = None,
                                 progress_callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Her örnek ve her dönüşüm için Perch gömmesi: (N, K, d). Önbellek; örnek anahtarları, dönüşüm
    listesi ve tohum aynıysa kullanılır."""
    kinds = list(kinds)
    keys = _keys(kept_samples)
    if cache_path and os.path.isfile(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        if ({"X", "keys", "kinds", "seed"} <= set(data.files)
                and np.array_equal(data["keys"], keys)
                and data["kinds"].tolist() == kinds and int(data["seed"]) == int(seed)):
            return data["X"]

    from .embedding_extraction import embedding_from_waveform
    import librosa

    if classifier is None:
        from ..detection.detector import get_shared_perch_classifier
        classifier = get_shared_perch_classifier()

    rows: List[np.ndarray] = []
    for i, s in enumerate(kept_samples):
        y, sr = librosa.load(s.filepath, sr=SR, mono=True)
        per = [embedding_from_waveform(classifier, augment_waveform(y, k, keys[i], seed, sr), sr,
                                       source=s.filepath) for k in kinds]
        rows.append(np.stack(per))
        if progress_callback:
            progress_callback(i + 1, len(kept_samples))
    X = np.stack(rows).astype(np.float32)
    if cache_path:
        np.savez(cache_path, X=X, keys=keys, kinds=np.array(kinds), seed=np.array(seed))
    return X


def run_augmentation_comparison(X: np.ndarray, X_aug: np.ndarray, kinds: Sequence[str], kept_samples: list,
                                n_splits: int = 5, n_repeats: int = 3, base_seed: int = 42,
                                class_weight: bool = True, include_single: bool = True,
                                progress_callback: Optional[Callable[[int, int], None]] = None):
    """Aynı kayıt-gruplu bölmelerde LR: artırmasız / tüm dönüşümler / (isteğe bağlı) tek tek dönüşümler.
    Döndürür: (fold_rows, oof, n_classes) — embedding_comparison.summarize_comparison/pairwise_table ile uyumlu.
    Yöntem adları: LR_Artirmasiz, LR_Artirmali(tum), LR_Artirmali(<dönüşüm>)."""
    from sklearn.preprocessing import LabelEncoder
    from .baselines import _metrics, fit_predict_logreg
    from .train_classifier import group_kfold_splits

    kinds = list(kinds)
    if X_aug.ndim != 3 or X_aug.shape[0] != len(kept_samples) or X_aug.shape[1] != len(kinds):
        raise ValueError(f"X_aug beklenen biçimde değil: {X_aug.shape} (N={len(kept_samples)}, K={len(kinds)}).")
    if X.shape[0] != len(kept_samples) or X.shape[1] != X_aug.shape[2]:
        raise ValueError("X ile X_aug uyuşmuyor.")

    le = LabelEncoder()
    y = le.fit_transform([s.species for s in kept_samples])
    n_classes = len(le.classes_)
    recs = np.array([s.recording_id for s in kept_samples])

    variants: Dict[str, List[int]] = {"Artirmasiz": []}
    variants["Artirmali(tum)"] = list(range(len(kinds)))
    if include_single and len(kinds) > 1:
        for j, k in enumerate(kinds):
            variants[f"Artirmali({k})"] = [j]

    rows: List[dict] = []
    oof = {n: {"y_true": [], "y_pred": [], "rec": [], "tekrar": []} for n in variants}
    total = n_repeats * n_splits * len(variants)
    done = 0
    for rep in range(n_repeats):
        seed = base_seed + rep
        for fold_i, sp in enumerate(group_kfold_splits(kept_samples, n_splits=n_splits, random_state=seed)):
            if set(recs[sp.train_idx]) & set(recs[sp.test_idx]):
                raise RuntimeError("Sızıntı: aynı kayıt hem eğitimde hem testte.")
            for name, js in variants.items():
                parts = [X[sp.train_idx]] + [X_aug[sp.train_idx, j] for j in js]
                Xtr = np.vstack(parts)
                ytr = np.concatenate([y[sp.train_idx]] * (1 + len(js)))
                pred = fit_predict_logreg(Xtr, ytr, X[sp.test_idx], class_weight, seed=seed)
                rows.append({"tekrar": rep + 1, "kat": fold_i + 1, "yontem": "LR_" + name,
                             "sinif_agirligi": bool(class_weight),
                             **_metrics(y[sp.test_idx], pred, n_classes),
                             "n_train": len(sp.train_idx), "n_train_artirmali": len(ytr),
                             "n_test": len(sp.test_idx)})
                o = oof[name]
                o["y_true"].extend(y[sp.test_idx].tolist())
                o["y_pred"].extend(np.asarray(pred).tolist())
                o["rec"].extend(recs[sp.test_idx].tolist())
                o["tekrar"].extend([rep + 1] * len(sp.test_idx))
                done += 1
                if progress_callback:
                    progress_callback(done, total)
    return rows, {("LR_" + k): v for k, v in oof.items()}, n_classes
