"""
Sınıflandırmayı güçlendirme denemeleri (TensorFlow gerektirmeyen kısım) ve seçici tahmin.

Hepsi run_cross_validation / run_baselines ile AYNI bölmelerde (kayıt bazlı, tohum = base_seed + tekrar)
değerlendirilir; böylece eşleştirilmiş fark testi (baselines.paired_comparison) doğrudan uygulanır.
Test kümesi hiçbir ayar seçimi için kullanılmaz; tüm hiperparametreler SABİTTİR (a priori):

  LR_SifirAtis_Perch : lojistik regresyon, öznitelik = Perch embedding + Perch'in kendi tür skorları
  Prototip_BirdNET     : kosinüs benzerliğiyle en yakın tür merkezi (az örnekli türler için)
  Topluluk_BirdNET     : LR + prototip (+ sıfır-atış varsa) olasılık ortalaması
  MLP_Topluluk       : (TensorFlow) farklı tohumlarla eğitilmiş MLP'lerin olasılık ortalaması

Seçici tahmin: en yüksek olasılık eşiğin altındaysa "tahmin etme". Birleştirilmiş (tüm kat/tekrar)
test tahminleri üzerinde kapsam (kabul edilen oran), kabul edilenlerde doğruluk/makro F1 ve ECE.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROTO_TEMP = 0.05          # prototip kosinüs skorlarını olasılığa çeviren sabit sıcaklık
ZS_TEMP = 1.0              # sıfır-atış skorları için softmax sıcaklığı (sabit)
THRESHOLDS = (0.0, 0.5, 0.7, 0.8, 0.9)


def _softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sanitize_zero_shot(scores: np.ndarray) -> np.ndarray:
    """Perch etiket listesinde olmayan türlerin (NaN/-inf) skorunu, sonlu skorların en küçüğüne çeker."""
    S = np.array(scores, dtype=float)
    finite = np.isfinite(S)
    if not finite.any():
        return np.zeros_like(S)
    S[~finite] = S[finite].min()
    return S


def _full_proba(clf_classes, proba: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((proba.shape[0], n_classes))
    out[:, np.asarray(clf_classes, dtype=int)] = proba
    return out


def lr_proba(X_train, y_train, X_test, n_classes: int, class_weight: bool = True, seed: int = 0) -> np.ndarray:
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced" if class_weight else None, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(sc.transform(X_train), y_train)
    return _full_proba(clf.classes_, clf.predict_proba(sc.transform(X_test)), n_classes)


def prototype_proba(X_train, y_train, X_test, n_classes: int) -> np.ndarray:
    def nrm(A):
        return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Xt, Xs = nrm(X_train), nrm(X_test)
    P = np.full((n_classes, X_train.shape[1]), np.nan)
    for c in np.unique(y_train):
        P[c] = Xt[y_train == c].mean(axis=0)
    present = ~np.isnan(P).any(axis=1)
    P[present] = nrm(P[present])
    sims = np.full((X_test.shape[0], n_classes), -1.0)
    sims[:, present] = Xs @ P[present].T
    return _softmax(sims, PROTO_TEMP)


def _metrics(y_true, y_pred, n_classes) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score
    from .olcutler import makro_f1
    return {"dogruluk": float(accuracy_score(y_true, y_pred)),
            "makro_f1": makro_f1(y_true, y_pred)}


def ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Beklenen kalibrasyon hatası (en yüksek olasılık, eşit genişlikli kutular)."""
    conf = proba.max(axis=1)
    ok = (proba.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(total)


def selective_rows(name: str, y_true: np.ndarray, proba: np.ndarray, n_classes: int,
                   thresholds: Sequence[float] = THRESHOLDS) -> List[dict]:
    """Eşik başına kapsam ve kabul edilenlerde başarı (birleştirilmiş OOF tahminler)."""
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    e = ece(y_true, proba)
    rows = []
    for t in thresholds:
        m = conf >= t
        if m.sum() == 0:
            rows.append({"yontem": name, "esik": t, "kapsam": 0.0, "dogruluk_kabul": float("nan"),
                         "makro_f1_kabul": float("nan"), "ece": e})
            continue
        mt = _metrics(y_true[m], pred[m], n_classes)
        rows.append({"yontem": name, "esik": t, "kapsam": float(m.mean()), "dogruluk_kabul": mt["dogruluk"],
                     "makro_f1_kabul": mt["makro_f1"], "ece": e})
    return rows


def run_improvements(
    X: np.ndarray, kept_samples: list, zero_shot_scores: Optional[np.ndarray] = None,
    n_splits: int = 5, n_repeats: int = 3, base_seed: int = 42,
    extra_proba_fns: Optional[Dict[str, Callable]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Döndürür: (kat bazlı satırlar [run_baselines ile aynı şema], seçici tahmin satırları).
    extra_proba_fns: {ad: f(train_idx, val_idx, test_idx, seed, fold_i) -> (n_test, n_classes) olasılık}
    (örn. TensorFlow'lu MLP topluluğu; cv_runner enjekte eder).
    """
    from sklearn.preprocessing import LabelEncoder
    from .train_classifier import group_kfold_splits

    le = LabelEncoder()
    y = le.fit_transform([s.species for s in kept_samples])
    n_classes = len(le.classes_)
    S = None
    if zero_shot_scores is not None:
        if zero_shot_scores.shape != (len(y), n_classes):
            raise ValueError("zero_shot_scores boyutu (N, sınıf sayısı) ile uyuşmuyor.")
        S = sanitize_zero_shot(zero_shot_scores)
    XS = np.hstack([X, S]) if S is not None else None

    names = ["Prototip_BirdNET", "Topluluk_BirdNET"] + (["LR_SifirAtis_Perch"] if S is not None else []) \
        + list((extra_proba_fns or {}).keys())
    rows: List[dict] = []
    pooled: Dict[str, Tuple[list, list]] = {n: ([], []) for n in names}
    total = n_repeats * n_splits * len(names)
    done = 0
    for rep in range(n_repeats):
        seed = base_seed + rep
        for fold_i, sp in enumerate(group_kfold_splits(kept_samples, n_splits=n_splits, random_state=seed)):
            tr, te = sp.train_idx, sp.test_idx
            probas: Dict[str, np.ndarray] = {}
            p_lr = lr_proba(X[tr], y[tr], X[te], n_classes, True, seed)
            p_pr = prototype_proba(X[tr], y[tr], X[te], n_classes)
            probas["Prototip_BirdNET"] = p_pr
            parts = [p_lr, p_pr]
            if S is not None:
                probas["LR_SifirAtis_Perch"] = lr_proba(XS[tr], y[tr], XS[te], n_classes, True, seed)
                parts.append(_softmax(S[te], ZS_TEMP))
            probas["Topluluk_BirdNET"] = np.mean(parts, axis=0)
            for nm, fn in (extra_proba_fns or {}).items():
                probas[nm] = fn(sp.train_idx, sp.val_idx, te, seed, fold_i)
            for nm in names:
                pred = probas[nm].argmax(axis=1)
                rows.append({"tekrar": rep + 1, "kat": fold_i + 1, "yontem": nm, "sinif_agirligi": nm != "Prototip_BirdNET",
                             **_metrics(y[te], pred, n_classes), "n_test": len(te)})
                pooled[nm][0].append(y[te])
                pooled[nm][1].append(probas[nm])
                done += 1
                if progress_callback:
                    progress_callback(done, total)
    sel: List[dict] = []
    for nm in names:
        yt = np.concatenate(pooled[nm][0])
        pp = np.vstack(pooled[nm][1])
        sel += selective_rows(nm, yt, pp, n_classes)
    return rows, sel


def mlp_ensemble_proba_fn(X: np.ndarray, y: np.ndarray, n_classes: int, n_models: int = 5,
                          max_epochs: int = 100) -> Callable:
    """TensorFlow gerektirir: n_models farklı tohumlu MLP'nin (sınıf ağırlıklı) olasılık ortalaması."""
    def fn(train_idx, val_idx, test_idx, seed, fold_i):
        from .train_classifier import build_and_train_mlp
        acc = np.zeros((len(test_idx), n_classes))
        for k in range(n_models):
            Xv = X[val_idx] if len(val_idx) else None
            yv = y[val_idx] if len(val_idx) else None
            model, _ = build_and_train_mlp(X[train_idx], y[train_idx], Xv, yv, n_classes=n_classes,
                                           embedding_dim=X.shape[1], max_epochs=max_epochs,
                                           random_state=seed * 100 + fold_i * 10 + k + 1, use_class_weights=True)
            acc += model.predict(X[test_idx], verbose=0)
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except Exception:  # noqa: BLE001
                pass
        return acc / n_models
    return fn
