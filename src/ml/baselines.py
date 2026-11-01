"""
src/ml/baselines.py
===================
Çapraz doğrulama için TEMEL ÇİZGİ (baseline) yöntemleri. Amaç, BirdNET embedding +
MLP seçiminin gerekçesini göstermektir; yöntemler AYNI kayıt-bazlı (grup-farkında)
bölmelerde (group_kfold_splits, aynı tohumlar) değerlendirilir.

Yöntemler
  * LR_BirdNET          : BirdNET embedding (1024) + lojistik regresyon (MLP'nin doğrusal karşılığı)
  * LR_MFCC           : klasik öznitelik: MFCC(20)+delta özet istatistikleri (ort., s.s.) + lojistik regresyon
  * SifirAtis_Perch   : eğitim YOK; Perch'in kendi hazır tür çıktısı (14 bin+ sınıf), yalnızca
                        bizim türlerle sınırlanarak argmax alınır. Perch etiket listesinde
                        olmayan türler hep yanlış sayılır (dürüst alt sınır).

Not: Eğitimli temel çizgiler yalnızca train_idx ile eğitilir (doğrulama kümesi MLP'de
erken durdurma içindir; burada kullanılmaz), test_idx ile değerlendirilir.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

MFCC_SR = 32000
N_MFCC = 20


def mfcc_summary_features(y: np.ndarray, sr: int = MFCC_SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    """MFCC ve delta-MFCC'nin zaman boyunca ortalama ve standart sapması -> (4*n_mfcc,) vektör."""
    import librosa

    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d = librosa.feature.delta(mf, width=3 if mf.shape[1] >= 3 else 1) if mf.shape[1] >= 3 else np.zeros_like(mf)
    return np.concatenate([mf.mean(axis=1), mf.std(axis=1), d.mean(axis=1), d.std(axis=1)]).astype(np.float32)


def extract_mfcc_for_file(filepath: str) -> np.ndarray:
    import librosa

    y, sr = librosa.load(filepath, sr=MFCC_SR, mono=True)
    if len(y) == 0:
        raise ValueError(f"Ses dosyası boş: {filepath}")
    return mfcc_summary_features(y, sr)


def _keys(kept_samples) -> np.ndarray:
    return np.array([f"{s.filepath}||{s.species}||{s.recording_id}" for s in kept_samples])


def _cached(cache_path: Optional[str], kept_samples) -> Optional[np.ndarray]:
    if cache_path and os.path.isfile(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        if "keys" in data.files and np.array_equal(data["keys"], _keys(kept_samples)):
            return data["X"]
    return None


def load_or_extract_mfcc(kept_samples, cache_path: Optional[str] = None,
                         progress_callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Embedding önbelleğiyle AYNI örnek sırası için MFCC özet matrisi (N, 80)."""
    cached = _cached(cache_path, kept_samples)
    if cached is not None:
        return cached
    rows = []
    for i, s in enumerate(kept_samples):
        rows.append(extract_mfcc_for_file(s.filepath))
        if progress_callback:
            progress_callback(i + 1, len(kept_samples))
    X = np.vstack(rows)
    if cache_path:
        np.savez(cache_path, X=X, keys=_keys(kept_samples))
    return X


def compute_zero_shot_scores(kept_samples, class_names: Sequence[str], classifier=None,
                             cache_path: Optional[str] = None,
                             progress_callback: Optional[Callable[[int, int], None]] = None,
                             hop_seconds: float = 2.5) -> np.ndarray:
    """
    (N, n_classes) puan matrisi: Perch'in hazır tür olasılıklarının pencereler boyunca
    MAKSİMUMU, yalnızca class_names sütunlarına indirgenmiş. Perch etiketlerinde bulunmayan
    sınıfların sütunu -inf olur (asla seçilmez). Yalnızca yerelde (TensorFlow + Perch) çalışır.
    """
    cached = _cached(cache_path, kept_samples)
    if cached is not None and cached.shape[1] == len(class_names):
        return cached

    import librosa
    from ..detection.perch_classifier import PERCH_SAMPLE_RATE, generate_windows
    if classifier is None:
        from .pipeline import get_shared_perch_classifier
        classifier = get_shared_perch_classifier()
    classifier.ensure_loaded()

    label_to_idx = {str(l).strip().lower(): i for i, l in enumerate(classifier.labels or [])}
    cols = [label_to_idx.get(str(c).strip().lower(), -1) for c in class_names]
    out = np.full((len(kept_samples), len(class_names)), -np.inf, dtype=np.float32)
    for i, s in enumerate(kept_samples):
        y, sr = librosa.load(s.filepath, sr=PERCH_SAMPLE_RATE, mono=True)
        probs = [classifier.predict_window(chunk) for _, _, chunk in generate_windows(y, sr, hop_seconds=hop_seconds)]
        if probs:
            pmax = np.max(probs, axis=0)
            for j, c in enumerate(cols):
                if c >= 0:
                    out[i, j] = pmax[c]
        if progress_callback:
            progress_callback(i + 1, len(kept_samples))
    if cache_path:
        np.savez(cache_path, X=out, keys=_keys(kept_samples))
    return out


def zero_shot_coverage(class_names: Sequence[str], perch_labels: Sequence[str]) -> float:
    """Sınıfların Perch etiket listesinde bulunan oranı."""
    have = {str(l).strip().lower() for l in perch_labels}
    return float(np.mean([str(c).strip().lower() in have for c in class_names])) if len(class_names) else 0.0


def _metrics(y_true, y_pred, n_classes) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score
    from .olcutler import makro_f1
    return {
        "dogruluk": float(accuracy_score(y_true, y_pred)),
        "makro_f1": makro_f1(y_true, y_pred),
    }


def fit_predict_logreg(X_train, y_train, X_test, class_weight: bool, C: float = 1.0, seed: int = 0):
    """Standartlaştırma + çok sınıflı lojistik regresyon; test tahminlerini (sınıf indeksi) döndürür."""
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced" if class_weight else None,
                             random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(sc.transform(X_train), y_train)
    return clf.predict(sc.transform(X_test))


def run_baselines(
    X: np.ndarray,
    kept_samples: list,
    mfcc: Optional[np.ndarray] = None,
    zero_shot_scores: Optional[np.ndarray] = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    base_seed: int = 42,
    class_weight_options=(False, True),
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[dict]:
    """
    Her (tekrar, kat) için temel çizgileri değerlendirir. run_cross_validation ile AYNI
    bölmeler (group_kfold_splits, tohum = base_seed + tekrar). Satır anahtarları:
    tekrar, kat, yontem, sinif_agirligi, dogruluk, makro_f1, n_test.
    """
    from sklearn.preprocessing import LabelEncoder
    from .train_classifier import group_kfold_splits

    le = LabelEncoder()
    y = le.fit_transform([s.species for s in kept_samples])
    n_classes = len(le.classes_)
    if zero_shot_scores is not None and zero_shot_scores.shape != (len(y), n_classes):
        raise ValueError("zero_shot_scores boyutu (N, sınıf sayısı) ile uyuşmuyor; sınıf sırası LabelEncoder sırası olmalı.")

    feats = {"LR_BirdNET": X}
    if mfcc is not None:
        feats["LR_MFCC"] = mfcc
    rows: List[dict] = []
    total = n_repeats * n_splits * (len(feats) * len(class_weight_options) + (1 if zero_shot_scores is not None else 0))
    done = 0
    for rep in range(n_repeats):
        seed = base_seed + rep
        for fold_i, sp in enumerate(group_kfold_splits(kept_samples, n_splits=n_splits, random_state=seed)):
            for name, F in feats.items():
                for cw in class_weight_options:
                    pred = fit_predict_logreg(F[sp.train_idx], y[sp.train_idx], F[sp.test_idx], bool(cw), seed=seed)
                    rows.append({"tekrar": rep + 1, "kat": fold_i + 1, "yontem": name, "sinif_agirligi": bool(cw),
                                 **_metrics(y[sp.test_idx], pred, n_classes), "n_test": len(sp.test_idx)})
                    done += 1
                    if progress_callback:
                        progress_callback(done, total)
            if zero_shot_scores is not None:
                pred = np.argmax(zero_shot_scores[sp.test_idx], axis=1)
                rows.append({"tekrar": rep + 1, "kat": fold_i + 1, "yontem": "SifirAtis_Perch", "sinif_agirligi": False,
                             **_metrics(y[sp.test_idx], pred, n_classes), "n_test": len(sp.test_idx)})
                done += 1
                if progress_callback:
                    progress_callback(done, total)
    return rows


def summarize(rows: List[dict]) -> List[dict]:
    """Yöntem x sınıf-ağırlığı bazında kat ortalaması ve s.s."""
    out = []
    keys = sorted({(r["yontem"], r["sinif_agirligi"]) for r in rows})
    for yontem, cw in keys:
        sel = [r for r in rows if r["yontem"] == yontem and r["sinif_agirligi"] == cw]
        d = {"yontem": yontem, "sinif_agirligi": cw, "n_kat": len(sel)}
        for m in ("dogruluk", "makro_f1"):
            v = np.array([r[m] for r in sel])
            d[m + "_ort"] = float(v.mean())
            d[m + "_ss"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out.append(d)
    return out


def paired_comparison(mlp_rows: List[dict], base_rows: List[dict], baseline: str,
                      base_class_weight: Optional[bool], mlp_class_weight: bool = True,
                      metric: str = "makro_f1") -> dict:
    """
    MLP ile bir temel çizgi arasında AYNI (tekrar, kat) bölmelerinde eşleştirilmiş fark.
    Katlar bağımsız olmadığından (tekrarlı CV) p değeri Nadeau-Bengio düzeltmesiyle verilir:
        var_düzeltilmiş = (1/k + n_test/n_eğitim) · s²,   n_eğitim = n_train + n_val
    Döndürür: fark ortalaması, s.s., kazanılan kat sayısı, naif ve düzeltilmiş p.
    mlp_rows: run_cross_validation fold_rows; base_rows: run_baselines satırları.
    """
    from scipy import stats

    mlp = {(r["tekrar"], r["kat"]): r for r in mlp_rows if bool(r["sinif_agirligi"]) == bool(mlp_class_weight)}
    base = {(r["tekrar"], r["kat"]): r for r in base_rows
            if r["yontem"] == baseline and (base_class_weight is None or bool(r["sinif_agirligi"]) == bool(base_class_weight))}
    keys = sorted(set(mlp) & set(base))
    if len(keys) < 3:
        raise ValueError("Eşleştirme için yeterli ortak (tekrar, kat) yok.")
    d = np.array([mlp[k][metric] - base[k][metric] for k in keys])
    k = len(d)
    mu, sd = float(d.mean()), float(d.std(ddof=1))
    ratio = float(np.mean([mlp[x]["n_test"] / (mlp[x]["n_train"] + mlp[x]["n_val"]) for x in keys]))
    if sd == 0:
        p_naif = p_duz = 1.0 if mu == 0 else 0.0
    else:
        p_naif = float(2 * stats.t.sf(abs(mu / (sd / np.sqrt(k))), k - 1))
        p_duz = float(2 * stats.t.sf(abs(mu / np.sqrt((1 / k + ratio) * sd ** 2)), k - 1))
    return {"karsilastirma": f"MLP - {baseline}", "metrik": metric, "n_kat": k, "fark_ort": mu, "fark_ss": sd,
            "mlp_kazanilan_kat": int((d > 0).sum()), "p_naif": p_naif, "p_duzeltilmis": p_duz}
