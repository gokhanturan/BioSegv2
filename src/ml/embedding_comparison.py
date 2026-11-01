"""
src/ml/embedding_comparison.py
==============================
Farklı ses gömme (embedding) modellerini AYNI kayıt-gruplu bölmelerde, AYNI sınıflandırıcıyla
(standartlaştırma + lojistik regresyon) karşılaştırır. Amaç: "hangi gömme daha iyi?" sorusunu
tek bir şanslı bölmeye değil, tekrarlı çapraz doğrulamaya, eşleştirilmiş farklara ve
kayıt düzeyinde bootstrap güven aralığına dayandırmak.

Gömme kaynakları (hepsi `kept_samples` ile AYNI sırada (N, d) matris döndürür)
  * Perch 2.0        : mevcut embedding_onbellek.npz (pipeline.load_or_extract_embeddings)
  * BirdNET 2.4      : `birdnet` paketi (>=1.x) ile; 48 kHz, 3 sn pencere, 1024 boyut, geçerli
                       pencerelerin ortalaması. Model ilk kullanımda internetten iner.
  * Harici (.npz)    : başka bir araçla (ör. AVES, BEATs, BioLingual) çıkarılmış gömmeler;
                       `keys` + `X` içerir, `embedding_onbellek.npz` ile aynı biçim.
  * Birleşik         : seçilen gömmelerin yan yana eklenmesi (concatenation).

Değerlendirme: pipeline/baselines ile AYNI bölmeler (group_kfold_splits, tohum = base_seed + tekrar).
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

BIRDNET_SR = 48000
BIRDNET_WINDOW_S = 3.0


# ----------------------------------------------------------------------------- gömme kaynakları
def sample_keys(kept_samples) -> np.ndarray:
    return np.array([f"{s.filepath}||{s.species}||{s.recording_id}" for s in kept_samples])


def _load_cached(cache_path: Optional[str], kept_samples) -> Optional[np.ndarray]:
    if cache_path and os.path.isfile(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        if "keys" in data.files and np.array_equal(data["keys"], sample_keys(kept_samples)):
            return data["X"]
    return None


def load_external_embeddings(npz_path: str, kept_samples) -> np.ndarray:
    """Başka bir araçla çıkarılmış gömmeleri `kept_samples` sırasına hizalar.
    Dosyada `X` (M, d) ve `keys` (M,) bulunmalı; kept_samples'taki her anahtar `keys` içinde olmalı."""
    data = np.load(npz_path, allow_pickle=False)
    if "X" not in data.files or "keys" not in data.files:
        raise ValueError(f"{npz_path}: 'X' ve 'keys' dizileri gerekir.")
    index = {k: i for i, k in enumerate(data["keys"].tolist())}
    want = sample_keys(kept_samples).tolist()
    missing = [k for k in want if k not in index]
    if missing:
        raise ValueError(f"{npz_path}: {len(missing)} örnek için gömme yok (ilk eksik: {missing[0]}).")
    return np.asarray(data["X"])[[index[k] for k in want]]


def pool_birdnet_result(result, index: int = 0) -> np.ndarray:
    """BirdNET `encode_arrays` sonucundan `index`. girdi için ortalama gömme (d,) üretir.
    `embeddings` (n_girdi, max_pencere, d); `embeddings_masked` True = geçersiz pencere."""
    emb = np.asarray(result.embeddings)[index]
    masked = np.asarray(result.embeddings_masked)[index]
    valid = ~masked.all(axis=1)
    if not valid.any():
        raise ValueError("BirdNET geçerli pencere üretmedi.")
    return emb[valid].mean(axis=0)


def extract_birdnet_embeddings(kept_samples, model=None, cache_path: Optional[str] = None,
                               library: str = "litert", chunk_size: int = 32,
                               progress_callback: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """BirdNET 2.4 gömmeleri (N, 1024). `model` verilmezse `birdnet.load('acoustic','2.4','tf',
    library=library)` ile yüklenir (TensorFlow gerekmesin diye varsayılan 'litert').

    `encode_arrays` her çağrıda yeni bir işlem oturumu (çoklu süreç) açtığı için segmentler TEK TEK değil,
    `chunk_size`'lık gruplar halinde verilir; böylece oturum maliyeti 663 kez değil, birkaç kez ödenir."""
    cached = _load_cached(cache_path, kept_samples)
    if cached is not None:
        return cached
    import librosa

    if model is None:
        from .birdnet_embedding import configure_model_dir
        configure_model_dir()
        try:
            import birdnet
        except ImportError as exc:
            raise ImportError("BirdNET gömmeleri için 'pip install birdnet' gerekir.") from exc
        model = birdnet.load("acoustic", "2.4", "tf", library=library)

    min_len = int(BIRDNET_SR * BIRDNET_WINDOW_S)
    chunk_size = max(1, int(chunk_size))
    rows = []
    for start in range(0, len(kept_samples), chunk_size):
        batch = []
        for s in kept_samples[start:start + chunk_size]:
            y, _ = librosa.load(s.filepath, sr=BIRDNET_SR, mono=True)
            if len(y) == 0:
                raise ValueError(f"Ses dosyası boş: {s.filepath}")
            if len(y) < min_len:                       # 3 sn'den kısa segmentler sıfırla doldurulur
                y = np.pad(y, (0, min_len - len(y)))
            batch.append((y.astype(np.float32), BIRDNET_SR))
        res = model.encode_arrays(batch)
        for i in range(len(batch)):
            rows.append(pool_birdnet_result(res, index=i))
        if progress_callback:
            progress_callback(min(start + chunk_size, len(kept_samples)), len(kept_samples))
    X = np.vstack(rows).astype(np.float32)
    if cache_path:
        np.savez(cache_path, X=X, keys=sample_keys(kept_samples))
    return X


def concatenate_embeddings(feature_sets: Dict[str, np.ndarray], names: Optional[Sequence[str]] = None) -> np.ndarray:
    """Seçilen gömmeleri yan yana ekler. Her biri eğitim bölmesinde ayrı standartlaştırıldığından
    (fit_predict_logreg) ölçek farkı sorun olmaz."""
    names = list(names or feature_sets)
    if len(names) < 2:
        raise ValueError("Birleştirme için en az iki gömme gerekir.")
    n = {feature_sets[k].shape[0] for k in names}
    if len(n) != 1:
        raise ValueError("Gömmelerin satır sayıları eşit olmalı.")
    return np.hstack([feature_sets[k] for k in names])


# ----------------------------------------------------------------------------- değerlendirme
def run_embedding_comparison(feature_sets: Dict[str, np.ndarray], kept_samples: list, n_splits: int = 5,
                             n_repeats: int = 3, base_seed: int = 42, class_weight: bool = True,
                             add_concat: bool = True,
                             progress_callback: Optional[Callable[[int, int], None]] = None):
    """Her gömmeyi aynı bölmelerde LR ile değerlendirir.
    Döndürür: (fold_rows, oof)
      fold_rows : tekrar, kat, yontem, sinif_agirligi, dogruluk, makro_f1, n_train, n_test
      oof       : {yontem: {'y_true','y_pred','rec','tekrar'}}  (kat dışı tahminler, bootstrap için)"""
    from sklearn.preprocessing import LabelEncoder
    from .baselines import _metrics, fit_predict_logreg
    from .train_classifier import group_kfold_splits

    sets = dict(feature_sets)
    if add_concat and len(sets) >= 2:
        sets["Birlesik(" + "+".join(feature_sets) + ")"] = concatenate_embeddings(feature_sets)
    for k, v in sets.items():
        if v.shape[0] != len(kept_samples):
            raise ValueError(f"'{k}' satır sayısı ({v.shape[0]}) örnek sayısıyla ({len(kept_samples)}) uyuşmuyor.")

    le = LabelEncoder()
    y = le.fit_transform([s.species for s in kept_samples])
    n_classes = len(le.classes_)
    recs = np.array([s.recording_id for s in kept_samples])

    rows: List[dict] = []
    oof = {k: {"y_true": [], "y_pred": [], "rec": [], "tekrar": []} for k in sets}
    total = n_repeats * n_splits * len(sets)
    done = 0
    for rep in range(n_repeats):
        seed = base_seed + rep
        for fold_i, sp in enumerate(group_kfold_splits(kept_samples, n_splits=n_splits, random_state=seed)):
            for name, F in sets.items():
                pred = fit_predict_logreg(F[sp.train_idx], y[sp.train_idx], F[sp.test_idx], class_weight, seed=seed)
                rows.append({"tekrar": rep + 1, "kat": fold_i + 1, "yontem": "LR_" + name,
                             "sinif_agirligi": bool(class_weight),
                             **_metrics(y[sp.test_idx], pred, n_classes),
                             "n_train": len(sp.train_idx), "n_test": len(sp.test_idx)})
                o = oof[name]
                o["y_true"].extend(y[sp.test_idx].tolist())
                o["y_pred"].extend(np.asarray(pred).tolist())
                o["rec"].extend(recs[sp.test_idx].tolist())
                o["tekrar"].extend([rep + 1] * len(sp.test_idx))
                done += 1
                if progress_callback:
                    progress_callback(done, total)
    return rows, {("LR_" + k): v for k, v in oof.items()}, n_classes


def paired_difference(rows: List[dict], a: str, b: str, metric: str = "makro_f1") -> dict:
    """Aynı (tekrar, kat) bölmelerinde a − b farkı; p değeri Nadeau-Bengio düzeltmeli
    (var_düzeltilmiş = (1/k + n_test/n_train)·s²). Pozitif fark: a üstün."""
    from scipy import stats

    A = {(r["tekrar"], r["kat"]): r for r in rows if r["yontem"] == a}
    B = {(r["tekrar"], r["kat"]): r for r in rows if r["yontem"] == b}
    keys = sorted(set(A) & set(B))
    if len(keys) < 3:
        raise ValueError("Eşleştirme için yeterli ortak (tekrar, kat) yok.")
    d = np.array([A[k][metric] - B[k][metric] for k in keys])
    k = len(d)
    mu, sd = float(d.mean()), float(d.std(ddof=1))
    ratio = float(np.mean([A[x]["n_test"] / A[x]["n_train"] for x in keys]))
    if sd == 0:
        p_naif = p_duz = 1.0 if mu == 0 else 0.0
    else:
        p_naif = float(2 * stats.t.sf(abs(mu / (sd / np.sqrt(k))), k - 1))
        p_duz = float(2 * stats.t.sf(abs(mu / np.sqrt((1 / k + ratio) * sd ** 2)), k - 1))
    return {"karsilastirma": f"{a} - {b}", "metrik": metric, "n_kat": k, "fark_ort": mu, "fark_ss": sd,
            "a_kazanilan_kat": int((d > 0).sum()), "p_naif": p_naif, "p_duzeltilmis": p_duz}


def bootstrap_macro_f1_ci(y_true, y_pred, recs, n_classes: int, n_boot: int = 1000, seed: int = 0,
                          alpha: float = 0.05):
    """Kayıt düzeyinde bootstrap: KAYITLAR (segmentler değil) yerine koyarak örneklenir, böylece aynı
    kayıttaki ilişkili segmentler tek birim sayılır. Döndürür: (nokta, alt, üst)."""
    from .olcutler import makro_f1

    y_true, y_pred, recs = np.asarray(y_true), np.asarray(y_pred), np.asarray(recs)
    uniq = np.unique(recs)
    by_rec = {r: np.flatnonzero(recs == r) for r in uniq}
    point = makro_f1(y_true, y_pred)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_rec[r] for r in pick])
        vals.append(makro_f1(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def _presence_by_repeat(o: dict, n_classes: int) -> dict:
    """Kayıt başına tür varlığı F1'i (>=1 ve >=2 segment); her tekrar için hesaplanıp ortalanır."""
    from .recording_level import presence_from_arrays

    yt, yp, rc, rp = (np.asarray(o["y_true"]), np.asarray(o["y_pred"]),
                      np.asarray(o["rec"]), np.asarray(o["tekrar"]))
    out = {}
    for t in (1, 2):
        vals = [presence_from_arrays(yt[rp == r], yp[rp == r], rc[rp == r], n_classes, min_segment=t)["f1"]
                for r in np.unique(rp)]
        out[f"varlik_f1_min{t}"] = float(np.mean(vals))
    return out


def summarize_comparison(rows: List[dict], oof: dict, n_classes: int, n_boot: int = 1000) -> List[dict]:
    """Yöntem başına: kat ortalaması ± s.s. ve kayıt-bootstrap %95 GA (makro F1)."""
    out = []
    for name in sorted({r["yontem"] for r in rows}):
        sel = [r for r in rows if r["yontem"] == name]
        d = {"yontem": name, "n_kat": len(sel)}
        for m in ("dogruluk", "makro_f1"):
            v = np.array([r[m] for r in sel])
            d[m + "_ort"] = float(v.mean())
            d[m + "_ss"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        o = oof.get(name)
        if o and o["y_true"]:
            p, lo, hi = bootstrap_macro_f1_ci(o["y_true"], o["y_pred"], o["rec"], n_classes, n_boot=n_boot)
            d.update(makro_f1_havuz=p, makro_f1_ga_alt=lo, makro_f1_ga_ust=hi)
            d.update(_presence_by_repeat(o, n_classes))
        out.append(d)
    out.sort(key=lambda r: -r["makro_f1_ort"])
    return out


def pairwise_table(rows: List[dict], reference: str, metric: str = "makro_f1") -> List[dict]:
    """Her yöntemi `reference` ile eşleştirilmiş karşılaştırır (yöntem − referans)."""
    res = []
    for name in sorted({r["yontem"] for r in rows} - {reference}):
        try:
            res.append(paired_difference(rows, name, reference, metric))
        except ValueError:
            pass
    return res
