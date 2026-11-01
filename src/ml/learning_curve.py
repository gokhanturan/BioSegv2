"""
Öğrenme eğrisi: eğitim verisinin %25/50/75/100'ü ile başarı.

Bölmeler run_cross_validation ile AYNIDIR (kayıt bazlı, tohum = base_seed + tekrar). Test ve doğrulama
kümeleri her oranda sabittir; yalnızca eğitim KAYITLARI azaltılır. Azaltma tür başına yapılır
(ceil(oran × o türün eğitimdeki kayıt sayısı), en az 1 kayıt) ve alt kümeler iç içedir (%25 ⊂ %50 ⊂ %75 ⊂ %100).
Oranlar önceden sabittir; sonuca bakılarak seçilmez.
"""
import csv
import math
from typing import Callable, List, Optional, Sequence

import numpy as np

from .pipeline import _resolve_trainer, fold_extra_metrics
from .train_classifier import evaluate_model, group_kfold_splits

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
METRICS = ("dogruluk", "makro_f1", "agirlikli_f1")


def subsample_train_indices(train_idx, samples, fraction: float, seed: int) -> np.ndarray:
    train_idx = np.asarray(train_idx, dtype=int)
    if fraction >= 1.0:
        return train_idx
    by_species: dict = {}
    for i in train_idx:
        by_species.setdefault(samples[i].species, set()).add(samples[i].recording_id)
    rng = np.random.default_rng(seed)
    keep_recs = set()
    for sp in sorted(by_species):
        recs = sorted(by_species[sp])
        order = rng.permutation(len(recs))
        n_keep = max(1, math.ceil(fraction * len(recs) - 1e-9))
        keep_recs.update(recs[j] for j in order[:n_keep])
    return np.array([i for i in train_idx if samples[i].recording_id in keep_recs], dtype=int)


def run_learning_curve(
    X: np.ndarray,
    kept_samples: list,
    fractions: Sequence[float] = FRACTIONS,
    n_splits: int = 5,
    n_repeats: int = 2,
    max_epochs: int = 100,
    base_seed: int = 42,
    use_class_weights: bool = False,
    trainer: str = "mlp",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    full_rows: Optional[list] = None,
) -> dict:
    """Döndürür: {'satirlar': [...], 'ozet': [...], 'sinif_adlari': [...]}.

    `full_rows`: aynı bölme ve ayarlarla yapılmış çapraz doğrulamanın kat satırları (pipeline.run_cross_validation
    `fold_rows`, tek sınıf-ağırlığı ayarı). Verilirse %100 oranı için yeniden eğitim yapılmaz, bu satırlar kullanılır
    (aynı tohum, aynı eğitim kümesi: sonuç özdeştir)."""
    from sklearn.preprocessing import LabelEncoder

    train_fn = _resolve_trainer(trainer)
    le = LabelEncoder()
    y = le.fit_transform([s.species for s in kept_samples])
    class_names = list(le.classes_)
    n_classes = len(class_names)
    fractions = tuple(sorted(set(float(f) for f in fractions)))

    rows: List[dict] = []
    reuse = {(r["tekrar"], r["kat"]): r for r in (full_rows or [])}
    total = n_repeats * n_splits * len(fractions)
    done = 0
    for rep in range(n_repeats):
        seed = base_seed + rep
        splits = group_kfold_splits(kept_samples, n_splits=n_splits, random_state=seed)
        for fold_i, sp in enumerate(splits):
            X_val = X[sp.val_idx] if len(sp.val_idx) else None
            y_val = y[sp.val_idx] if len(sp.val_idx) else None
            for frac in fractions:
                tr_idx = subsample_train_indices(sp.train_idx, kept_samples, frac, seed * 100 + fold_i)
                old = reuse.get((rep + 1, fold_i + 1)) if frac >= 1.0 else None
                if old is not None:
                    rows.append({
                        "tekrar": rep + 1, "kat": fold_i + 1, "oran": frac,
                        "dogruluk": float(old["dogruluk"]), "makro_f1": float(old["makro_f1"]),
                        "agirlikli_f1": float(old["agirlikli_f1"]),
                        "n_egitim_segment": int(len(tr_idx)),
                        "n_egitim_kayit": len({kept_samples[i].recording_id for i in tr_idx}),
                        "n_test_segment": int(len(sp.test_idx)),
                    })
                    done += 1
                    if progress_callback is not None:
                        progress_callback(done, total)
                    continue
                model, _ = train_fn(
                    X[tr_idx], y[tr_idx], X_val, y_val,
                    n_classes=n_classes, embedding_dim=X.shape[1], max_epochs=max_epochs,
                    random_state=seed * 100 + fold_i, use_class_weights=use_class_weights,
                )
                ev = evaluate_model(model, X[sp.test_idx], y[sp.test_idx], class_names)
                proba = model.predict(X[sp.test_idx], verbose=0)
                a_f1, _ = fold_extra_metrics(y[sp.test_idx], proba, n_classes)
                rows.append({
                    "tekrar": rep + 1, "kat": fold_i + 1, "oran": frac,
                    "dogruluk": float(ev["accuracy"]), "makro_f1": float(ev["macro_f1"]),
                    "agirlikli_f1": float(a_f1),
                    "n_egitim_segment": int(len(tr_idx)),
                    "n_egitim_kayit": len({kept_samples[i].recording_id for i in tr_idx}),
                    "n_test_segment": int(len(sp.test_idx)),
                })
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except Exception:  # noqa: BLE001
                    pass
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total)
    return {"satirlar": rows, "ozet": summarize_learning_curve(rows), "sinif_adlari": class_names,
            "n_kat": n_splits, "n_tekrar": n_repeats, "sinif_agirligi": bool(use_class_weights)}


def summarize_learning_curve(rows: list) -> List[dict]:
    """Oran başına ortalama ± SS (kat SS'si, GA değil) ve aynı katta %100'e göre eşleştirilmiş fark."""
    full = max(r["oran"] for r in rows)
    ref = {(r["tekrar"], r["kat"]): r for r in rows if r["oran"] == full}
    out = []
    for frac in sorted({r["oran"] for r in rows}):
        sel = [r for r in rows if r["oran"] == frac]
        item = {"oran": frac, "n_kat": len(sel),
                "n_egitim_segment_ort": float(np.mean([r["n_egitim_segment"] for r in sel])),
                "n_egitim_kayit_ort": float(np.mean([r["n_egitim_kayit"] for r in sel]))}
        for m in METRICS:
            v = np.array([r[m] for r in sel], dtype=float)
            item[m + "_ort"] = float(v.mean())
            item[m + "_ss"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            d = np.array([r[m] - ref[(r["tekrar"], r["kat"])][m] for r in sel], dtype=float)
            item[m + "_fark_tam"] = float(d.mean())
        out.append(item)
    return out


def write_csvs(folder: str, result: dict) -> List[str]:
    import os
    files = []
    for name, rows in (("ogrenme_egrisi.csv", result["satirlar"]), ("ogrenme_egrisi_ozet.csv", result["ozet"])):
        if not rows:
            continue
        with open(os.path.join(folder, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(rows)
        files.append(name)
    return files


def format_text(lc: dict) -> str:
    if not lc:
        return ""
    if "hata" in lc:
        return f"⚠ Öğrenme eğrisi çalıştırılamadı: {lc['hata']}"
    lines = ["ÖĞRENME EĞRİSİ (eğitim kayıtlarının oranına göre; test kümesi sabit; ± = katlar arası SS):"]
    for o in lc["ozet"]:
        lines.append(
            f"  %{o['oran'] * 100:.0f} (≈{o['n_egitim_kayit_ort']:.0f} eğitim kaydı): doğruluk %{o['dogruluk_ort'] * 100:.1f} ± "
            f"{o['dogruluk_ss'] * 100:.1f}, makro F1 {o['makro_f1_ort']:.3f} ± {o['makro_f1_ss']:.3f}, "
            f"%100'e göre makro F1 farkı {o['makro_f1_fark_tam']:+.3f}")
    return "\n".join(lines)


def make_figure(ozet: list, model_name: str = ""):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7.5, 4.6), dpi=100)
    ax = fig.add_subplot(111)
    xs = [o["oran"] * 100 for o in ozet]
    for key, label, color in (("makro_f1", "Makro F1 (×100)", "#c0392b"), ("dogruluk", "Doğruluk (%)", "#2e6fb5")):
        ax.errorbar(xs, [o[key + "_ort"] * 100 for o in ozet], yerr=[o[key + "_ss"] * 100 for o in ozet],
                    marker="o", capsize=3, label=label, color=color)
    ax.set_title("Eğitim verisi oranına göre başarı" + (f" — {model_name}" if model_name else ""))
    ax.set_xlabel("Kullanılan eğitim kaydı oranı (%)")
    ax.set_ylabel("Başarı")
    ax.set_xticks(xs)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
