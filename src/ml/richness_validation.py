"""
src/ml/richness_validation.py
=============================
Yapay zekâ (model) ile belirlenen tür zenginliğini, uzman etiketlerinden
elde edilen zenginlikle karşılaştırır.

Kapsam (açıkça): Girdi, uzmanın zaten ayırdığı SEGMENTLERDİR; model her segmentin
türünü sınıflandırır. Dolayısıyla ölçülen şey, sınıflandırma hatalarının nokta
düzeyindeki tür zenginliğine yansımasıdır (tespit/segmentasyon hatası dahil değildir).
Evren, modelin bildiği türlerdir (eğitim filtresinden sonra kalan türler).

Dürüstlük ilkesi: her segmentin sınıflandırması, o segmentin KAYDINI eğitimde görmemiş bir
modelden gelir (kayıt bazlı, grup-farkında çapraz doğrulama; run_cross_validation ile
aynı bölmeler). Böylece "model kendi eğitim verisini sınıflandırıyor" yanılgısı olmaz.

Neden önemli: yanlış sınıflandırılan segmentler, noktada gerçekte olmayan "yanlış pozitif tür"
üretir; bunlar belirlenen tür sayısını şişirebilir.
Güven eşiği (min_confidence) bu etkiyi azaltır; sonuçlar eşiğe göre raporlanır.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..data import richness_core as R


# ---------------------------------------------------------------------------
# Çapraz doğrulama dışı (out-of-fold) çıktılar
# ---------------------------------------------------------------------------
def _lr_predict(X_tr, y_tr, X_te, n_classes, seed):
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(sc.transform(X_tr), y_tr)
    proba = np.zeros((len(X_te), n_classes))
    proba[:, clf.classes_] = clf.predict_proba(sc.transform(X_te))
    return proba


def _mlp_predict(X_tr, y_tr, X_val, y_val, X_te, n_classes, seed, max_epochs):
    from .train_classifier import build_and_train_mlp

    model, _ = build_and_train_mlp(
        X_tr, y_tr, X_val, y_val, n_classes=n_classes, embedding_dim=X_tr.shape[1],
        max_epochs=max_epochs, random_state=seed, use_class_weights=True)
    proba = model.predict(X_te, verbose=0)
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:  # noqa: BLE001
        pass
    return proba


def out_of_fold_predictions(
    X: np.ndarray,
    kept_samples: list,
    method: str = "mlp",
    n_splits: int = 5,
    n_repeats: int = 3,
    base_seed: int = 42,
    max_epochs: int = 100,
    zero_shot_scores: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """
    Her tekrar için, her segmentin (test katında yer aldığı) model çıktısını döndürür:
    (class_names, [ (pred_idx[N], conf[N]) , ... ]) — tekrar sayısı kadar eleman.
    method: "mlp" (Perch + MLP, sınıf ağırlıklı), "lr" (Perch + lojistik regresyon),
            "zeroshot" (Perch hazır çıktısı; zero_shot_scores gerekir).
    """
    from sklearn.preprocessing import LabelEncoder
    from .train_classifier import group_kfold_splits

    le = LabelEncoder()
    y = le.fit_transform([s.species for s in kept_samples])
    n_classes = len(le.classes_)
    n = len(y)
    if method == "zeroshot":
        if zero_shot_scores is None or zero_shot_scores.shape != (n, n_classes):
            raise ValueError("zeroshot için (N, sınıf sayısı) boyutunda zero_shot_scores gerekir.")
    elif method not in ("mlp", "lr"):
        raise ValueError(f"Bilinmeyen yöntem: {method}")

    out = []
    done, total = 0, n_repeats * (n_splits if method != "zeroshot" else 1)
    for rep in range(n_repeats):
        seed = base_seed + rep
        pred = np.full(n, -1, dtype=int)
        conf = np.zeros(n)
        if method == "zeroshot":
            sc = np.where(np.isfinite(zero_shot_scores), zero_shot_scores, -np.inf)
            pred, conf = sc.argmax(axis=1), sc.max(axis=1)
            out.append((pred, conf))
            done += 1
            if progress_callback:
                progress_callback(done, total)
            continue
        for fold_i, sp in enumerate(group_kfold_splits(kept_samples, n_splits=n_splits, random_state=seed)):
            if method == "lr":
                proba = _lr_predict(X[sp.train_idx], y[sp.train_idx], X[sp.test_idx], n_classes, seed)
            else:
                Xv = X[sp.val_idx] if len(sp.val_idx) else None
                yv = y[sp.val_idx] if len(sp.val_idx) else None
                proba = _mlp_predict(X[sp.train_idx], y[sp.train_idx], Xv, yv, X[sp.test_idx],
                                     n_classes, seed * 100 + fold_i, max_epochs)
            pred[sp.test_idx] = proba.argmax(axis=1)
            conf[sp.test_idx] = proba.max(axis=1)
            done += 1
            if progress_callback:
                progress_callback(done, total)
        out.append((pred, conf))
    return list(le.classes_), out


# ---------------------------------------------------------------------------
# Nokta × tür tabloları ve karşılaştırma
# ---------------------------------------------------------------------------
def point_incidence(labels: np.ndarray, point_ids: Sequence, n_classes: int,
                    points: Optional[Sequence] = None) -> np.ndarray:
    """(nokta × tür) bool tablo. labels = sınıf indeksi (-1 = tespit yok). points: satır sırası."""
    pts = list(points) if points is not None else sorted(set(point_ids), key=str)
    idx = {p: i for i, p in enumerate(pts)}
    inc = np.zeros((len(pts), n_classes), dtype=bool)
    for lab, p in zip(labels, point_ids):
        if lab >= 0 and p in idx:
            inc[idx[p], lab] = True
    return inc


def compare_incidence(expert: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """İki nokta×tür tablosunu (aynı satır/sütun düzeni) karşılaştırır."""
    from scipy.stats import spearmanr

    tp = int((expert & predicted).sum())
    fp = int((~expert & predicted).sum())
    fn = int((expert & ~predicted).sum())
    s_e, s_p = expert.sum(axis=1), predicted.sum(axis=1)
    rho = float("nan")
    if len(s_e) > 2 and np.ptp(s_e) > 0 and np.ptp(s_p) > 0:
        rho = float(spearmanr(s_e, s_p)[0])
    st_e = R.incidence_stats(expert)
    st_p = R.incidence_stats(predicted)
    return {
        "nokta_sayisi": int(expert.shape[0]),
        "tur_uzman": st_e.s_obs, "tur_model": st_p.s_obs,
        "nokta_basina_tur_uzman_ort": float(s_e.mean()), "nokta_basina_tur_model_ort": float(s_p.mean()),
        "nokta_basina_tur_mutlak_fark_ort": float(np.abs(s_e - s_p).mean()),
        "nokta_basina_tur_spearman": rho,
        "var_yok_kesinlik": tp / (tp + fp) if tp + fp else float("nan"),
        "var_yok_duyarlilik": tp / (tp + fn) if tp + fn else float("nan"),
        "var_yok_jaccard": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
        "sahte_tur_nokta_cifti": fp, "kacirilan_tur_nokta_cifti": fn,
    }


def evaluate_richness_agreement(
    kept_samples: list,
    point_of_recording: Dict[str, str],
    class_names: Sequence[str],
    predictions: List[tuple],
    thresholds: Sequence[float] = (0.0, 0.5, 0.7, 0.9),
) -> List[dict]:
    """
    Her (tekrar, güven eşiği) için uzman ve model zenginliğini karşılaştırır.
    point_of_recording: kayıt kimliği -> Habitat_ID (nokta). Noktası bilinmeyen kayıtlar atlanır.
    """
    cls_idx = {c: i for i, c in enumerate(class_names)}
    y_true = np.array([cls_idx[s.species] for s in kept_samples])
    pts = np.array([point_of_recording.get(s.recording_id) for s in kept_samples], dtype=object)
    ok = np.array([p is not None for p in pts])
    if not ok.any():
        raise ValueError("Hiçbir kaydın noktası (Habitat_ID) bulunamadı.")
    points = sorted({p for p in pts[ok]}, key=str)
    n_cls = len(class_names)
    expert = point_incidence(y_true[ok], pts[ok], n_cls, points)

    rows = []
    for rep, (pred, conf) in enumerate(predictions, start=1):
        for thr in thresholds:
            lab = np.where(conf >= thr, pred, -1)[ok]
            pm = point_incidence(lab, pts[ok], n_cls, points)
            row = {"tekrar": rep, "guven_esigi": thr,
                   "tespit_orani": float((lab >= 0).mean())}
            row.update(compare_incidence(expert, pm))
            rows.append(row)
    return rows


def per_point_table(kept_samples, point_of_recording, class_names, prediction, threshold=0.0) -> List[dict]:
    """Tek bir sınıflandırma çıktısı seti için nokta bazlı tür sayısı (uzman / model) — dağılım grafiği için."""
    cls_idx = {c: i for i, c in enumerate(class_names)}
    y_true = np.array([cls_idx[s.species] for s in kept_samples])
    pts = np.array([point_of_recording.get(s.recording_id) for s in kept_samples], dtype=object)
    ok = np.array([p is not None for p in pts])
    points = sorted({p for p in pts[ok]}, key=str)
    pred, conf = prediction
    lab = np.where(conf >= threshold, pred, -1)[ok]
    e = point_incidence(y_true[ok], pts[ok], len(class_names), points)
    m = point_incidence(lab, pts[ok], len(class_names), points)
    return [{"nokta": p, "tur_uzman": int(e[i].sum()), "tur_model": int(m[i].sum())} for i, p in enumerate(points)]


def summarize_by_threshold(rows: List[dict]) -> List[dict]:
    """Tekrarlar üzerinden eşik bazında ortalama ve s.s."""
    out = []
    for thr in sorted({r["guven_esigi"] for r in rows}):
        sel = [r for r in rows if r["guven_esigi"] == thr]
        d = {"guven_esigi": thr, "n_tekrar": len(sel)}
        for k in sel[0]:
            if k in ("tekrar", "guven_esigi"):
                continue
            v = np.array([r[k] for r in sel], dtype=float)
            d[k + "_ort"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            d[k + "_ss"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else 0.0
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Eğitim hattına bağlantı (çapraz doğrulama tahminlerini yeniden kullanır; ek eğitim yapmaz)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = (0.0, 0.5, 0.7, 0.9)


def load_point_info(csv_path: str) -> Dict[str, dict]:
    """Habitat_ID -> {'tip': en sık Habitat_Type, 'konum': en sık Location, 'n_kayit': kayıt sayısı}."""
    import pandas as pd
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    df = df[df["Habitat_ID"].str.strip() != ""].copy()
    if "Original_Recording_Name" in df.columns:   # L/M çiftleri tek kayıt sayılır (src/data/recording_groups.py)
        from ..data.recording_groups import canonical_recording_ids
        df["Original_Recording_Name"] = canonical_recording_ids(df)

    def top(col, sub):
        if col not in sub:
            return ""
        v = sub[col].str.strip()
        v = v[v != ""]
        return str(v.value_counts().index[0]) if len(v) else ""

    out = {}
    for hid, sub in df.groupby(df["Habitat_ID"].str.strip()):
        out[hid] = {"tip": top("Habitat_Type", sub), "konum": top("Location", sub),
                    "n_kayit": int(sub["Original_Recording_Name"].nunique()) if "Original_Recording_Name" in sub else 0}
    return out


def per_point_detail(kept_samples, point_of_recording, class_names, prediction, threshold=0.0,
                     point_info: Optional[Dict[str, dict]] = None) -> List[dict]:
    """Nokta başına: uzman / model tür sayısı, yanlış pozitif (modelin eklediği) ve kaçırılan türlerin adları, habitat bilgisi."""
    cls_idx = {c: i for i, c in enumerate(class_names)}
    y_true = np.array([cls_idx[s.species] for s in kept_samples])
    pts = np.array([point_of_recording.get(s.recording_id) for s in kept_samples], dtype=object)
    ok = np.array([p is not None for p in pts])
    points = sorted({p for p in pts[ok]}, key=str)
    pred, conf = prediction
    lab = np.where(conf >= threshold, pred, -1)[ok]
    e = point_incidence(y_true[ok], pts[ok], len(class_names), points)
    m = point_incidence(lab, pts[ok], len(class_names), points)
    rec_count, seg_count = {}, {}
    for s_, p_ in zip(kept_samples, pts):
        if p_ is not None:
            rec_count.setdefault(p_, set()).add(s_.recording_id)
            seg_count[p_] = seg_count.get(p_, 0) + 1
    info = point_info or {}
    rows = []
    for i, p in enumerate(points):
        fp = [class_names[j] for j in np.flatnonzero(m[i] & ~e[i])]
        fn = [class_names[j] for j in np.flatnonzero(e[i] & ~m[i])]
        rows.append({"nokta": p, "habitat_tipi": info.get(p, {}).get("tip", ""), "konum": info.get(p, {}).get("konum", ""),
                     "n_kayit": len(rec_count.get(p, ())), "n_segment": seg_count.get(p, 0), "tur_uzman": int(e[i].sum()), "tur_model": int(m[i].sum()),
                     "sahte_turler": "; ".join(fp), "kacirilan_turler": "; ".join(fn)})
    return rows


def load_point_map(csv_path: str) -> Dict[str, str]:
    """Orijinal kayıt adı -> Habitat_ID (nokta). Habitat_ID boş olan kayıtlar yer almaz."""
    import pandas as pd
    from ..data.recording_groups import canonical_recording_ids
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    df = df[(df["Original_Recording_Name"].str.strip() != "") & (df["Habitat_ID"].str.strip() != "")].copy()
    # Eğitimde kullanılan (birleştirilmiş) kayıt kimliği ve orijinal ad ikisi de nokta eşlemesinde bulunur.
    df["_g"] = canonical_recording_ids(df)
    out = {}
    for col in ("Original_Recording_Name", "_g"):
        for r, h in zip(df[col], df["Habitat_ID"]):
            out.setdefault(r.strip(), h.strip())
    return out


def agreement_from_oof(kept_samples: list, point_of_recording: Dict[str, str], class_names: Sequence[str],
                       predictions: List[tuple], thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
                       point_info: Optional[Dict[str, dict]] = None) -> dict:
    """{'esikler': eşik başına özet, 'nokta': ilk tekrar/eşik 0 için nokta tablosu, ...}."""
    rows = evaluate_richness_agreement(kept_samples, point_of_recording, class_names, predictions, thresholds)
    rec_ids = {s.recording_id for s in kept_samples}
    return {
        "esikler": summarize_by_threshold(rows),
        "nokta": per_point_detail(kept_samples, point_of_recording, class_names, predictions[0], thresholds[0], point_info),
        "n_nokta": rows[0]["nokta_sayisi"],
        "n_tekrar": len(predictions),
        "n_noktasiz_kayit": len(rec_ids - set(point_of_recording)),
    }


def format_text(z: dict) -> str:
    if not z:
        return ""
    if "hata" in z:
        return f"⚠ Tür zenginliği uyumu hesaplanamadı: {z['hata']}"
    lines = [f"TÜR ZENGİNLİĞİ UYUMU (model vs uzman, {z['n_nokta']} nokta, çapraz doğrulama tahminleri; "
             "yalnızca sınıflandırma hatasının etkisi, tespit aşaması hariç):"]
    for o in z["esikler"]:
        lines.append(
            f"  güven ≥ {o['guven_esigi']:.1f}: tespit oranı {o['tespit_orani_ort']:.2f} | tür uzman {o['tur_uzman_ort']:.0f} / "
            f"model {o['tur_model_ort']:.1f} | Jaccard {o['var_yok_jaccard_ort']:.2f} (kesinlik {o['var_yok_kesinlik_ort']:.2f}, "
            f"duyarlılık {o['var_yok_duyarlilik_ort']:.2f}) | yanlış pozitif tür (nokta,tür çifti) {o['sahte_tur_nokta_cifti_ort']:.1f}, "
            f"kaçırılan {o['kacirilan_tur_nokta_cifti_ort']:.1f}")
    if z.get("n_noktasiz_kayit"):
        lines.append(f"  ⚠ {z['n_noktasiz_kayit']} kaydın Habitat_ID değeri yok; karşılaştırmaya girmedi.")
    return "\n".join(lines)


def write_csvs(folder: str, z: dict) -> List[str]:
    import csv
    import os
    files = []
    for name, rows in (("zenginlik_ozet.csv", z.get("esikler")), ("zenginlik_nokta.csv", z.get("nokta"))):
        if not rows:
            continue
        with open(os.path.join(folder, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(rows)
        files.append(name)
    return files


def group_points(nokta_rows: list) -> dict:
    """(uzman tür sayısı, model tür sayısı) -> o değere düşen nokta satırları (doğal sırayla)."""
    out: dict = {}
    for r in nokta_rows:
        out.setdefault((int(r["tur_uzman"]), int(r["tur_model"])), []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: _natural_key(str(r.get("nokta", ""))))
    return out


def hover_html(rows: list, name_fn=lambda t: t, max_points: int = 6) -> str:
    """Haritadaki açılır bilgi kutusuyla aynı düzende: 'ID (habitat tipi) — konum', sayılar, yanlış pozitif/kaçırılan türler."""
    def names(text):
        return "; ".join(name_fn(t) for t in str(text).split("; ") if t)

    blocks = []
    for r in rows[:max_points]:
        title = str(r.get("nokta", ""))
        if r.get("habitat_tipi"):
            title += f" ({r['habitat_tipi']})"
        if r.get("konum"):
            title += f" — {r['konum']}"
        lines = [f"<b>{title}</b>"]
        meta = []
        if r.get("n_kayit") is not None:
            meta.append(f"Kayıt: {r['n_kayit']}")
        if r.get("n_segment"):
            meta.append(f"Segment: {r['n_segment']}")
        if meta:
            lines.append(" · ".join(meta))
        lines.append(f"Tür sayısı: uzman {r['tur_uzman']} / model {r['tur_model']}")
        if r.get("sahte_turler"):
            lines.append(f"<span style='color:#b45309'>Yanlış pozitif türler: {names(r['sahte_turler'])}</span>")
        if r.get("kacirilan_turler"):
            lines.append(f"<span style='color:#b91c1c'>Kaçırılan türler: {names(r['kacirilan_turler'])}</span>")
        if not r.get("sahte_turler") and not r.get("kacirilan_turler") and r["tur_uzman"] == r["tur_model"]:
            lines.append("Tür listesi uzmanla aynı")
        blocks.append("<br>".join(lines))
    if len(rows) > max_points:
        blocks.append(f"<i>+{len(rows) - max_points} nokta daha (tabloya bakın)</i>")
    return "<div>" + "<hr>".join(blocks) + "</div>"


def _natural_key(text: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def _wrap_names(items: list, per_line: int = 3, max_items: int = 9, max_len: int = 12) -> str:
    items = [i if len(i) <= max_len else i[:max_len - 1] + "…" for i in items if i]
    extra = max(0, len(items) - max_items)
    items = items[:max_items]
    lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    if extra:
        lines.append(f"+{extra} nokta daha")
    return "\n".join(lines)


SCATTER_INFO = "güven eşiği yok, ilk tekrar"


def make_scatter_figure(nokta_rows: list, model_name: str = "", compact: bool = False, info: str = SCATTER_INFO):
    """Aynı (uzman, model) değerine düşen noktalar üst üste biner; bu yüzden her kombinasyon tek işaretle,
    boyutu ve içindeki sayı nokta adedini gösterecek şekilde çizilir."""
    from collections import Counter
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    pairs = Counter((int(r["tur_uzman"]), int(r["tur_model"])) for r in nokta_rows)
    fig_marker_sizes: dict = {}
    names: dict = {}
    for r in nokta_rows:
        names.setdefault((int(r["tur_uzman"]), int(r["tur_model"])), []).append(str(r.get("nokta", "")))
    n = sum(pairs.values())
    n_eq = sum(c for (x, y), c in pairs.items() if x == y)
    n_up = sum(c for (x, y), c in pairs.items() if y > x)
    n_dn = sum(c for (x, y), c in pairs.items() if y < x)
    top = max([1] + [max(x, y) for x, y in pairs]) + 1

    fig = Figure(figsize=(6.4, 6.6) if compact else (8.6, 7.6), dpi=100)
    ax = fig.add_subplot(111)
    lo = -0.9                                  # alt/sol boşluk: 'model eksik' etiketleri eksene sığsın
    ax.set_xlim(lo, top)
    ax.set_ylim(lo, top)
    ax.set_aspect("equal")
    ax.fill_between([lo, top], [lo, top], [top, top], color="#f59e0b", alpha=0.07, linewidth=0)
    ax.fill_between([lo, top], [lo, lo], [lo, top], color="#dc2626", alpha=0.06, linewidth=0)
    ax.plot([lo, top], [lo, top], color="#6b7280", linestyle="--", linewidth=1.2)
    ax.text(0.2, top - 0.35, "Model daha fazla tür buldu", color="#b45309", fontsize=10, va="top")
    ax.text(top - 0.2, -0.2, "Model daha az tür buldu", color="#b91c1c", fontsize=10, ha="right", va="bottom")

    style = ({"eq": ("#2e6fb5", "o", f"Aynı sayıda ({n_eq})"), "up": ("#d97706", "^", f"Model fazla ({n_up})"),
              "dn": ("#b91c1c", "v", f"Model eksik ({n_dn})")} if compact else
             {"eq": ("#2e6fb5", "o", f"Aynı tür sayısı ({n_eq} nokta)"),
              "up": ("#d97706", "^", f"Model fazla ({n_up} nokta)"),
              "dn": ("#b91c1c", "v", f"Model eksik ({n_dn} nokta)")})
    seen = set()
    for (x, y), c in sorted(pairs.items()):
        key = "eq" if x == y else ("up" if y > x else "dn")
        color, marker, label = style[key]
        fig_marker_sizes[(x, y)] = 150 + 150 * c
        ax.scatter([x], [y], s=150 + 150 * c, color=color, marker=marker, alpha=0.85, edgecolor="white",
                   linewidth=1.2, zorder=3, label=None if key in seen else label)
        seen.add(key)
        ax.text(x, y, str(c), color="white", fontsize=10, fontweight="bold", ha="center", va="center", zorder=4)
        label = _wrap_names(sorted(names[(x, y)], key=_natural_key))
        if compact and key == "eq":
            label = ""          # kompakt görünümde yalnızca uyumsuz noktaların Habitat_ID'si yazılır (okunabilirlik)
        if label:
            if key == "eq":                      # uyumlu noktalar: işaretin sağında
                off, ha, va = (14, 0), "left", "center"
            elif key == "up":                    # model fazla: işaretin üstünde (komşu işaretlerle çakışmasın)
                off, ha, va = (0, 14), "center", "bottom"
            else:                                # model eksik: işaretin altında
                off, ha, va = (0, -14), "center", "top"
            ax.annotate(label, (x, y), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=8,
                        color="#374151", zorder=5, linespacing=1.15)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Uzman: tür sayısı (nokta başına)" if compact else "Uzman etiketlerinden tür sayısı (nokta başına)")
    ax.set_ylabel("Model: tür sayısı (nokta başına)" if compact else "Model tahminlerinden tür sayısı (nokta başına)")
    ax.set_title(("Her noktada uzman ve modelin bulduğu tür sayısı" if compact else "Nokta başına tür sayısı: uzman ve model")
                 + (f"\n{model_name}" if (model_name and compact) else (f" — {model_name}" if model_name else "")),
                 fontsize=11 if compact else 12, pad=34 if compact else 30)
    ax.grid(alpha=0.25)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=m, linestyle="", markersize=10, markerfacecolor=c, markeredgecolor="white",
                      label=lab) for c, m, lab in (style["eq"], style["up"], style["dn"])]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13 if compact else -0.1), ncol=3,
              frameon=False, fontsize=9)
    fig.marker_sizes = fig_marker_sizes
    sub = f"{n} nokta · {info}"
    ax.text(0.5, 1.012, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#374151")
    foot = ("İşaretin boyutu ve içindeki sayı = o değerdeki nokta sayısı.\nİşaretin yanındaki sayılar = Habitat_ID "
            "(yalnızca uyumsuz noktalar).\nKöşegen = aynı SAYIDA tür; türlerin de aynı olduğu anlamına gelmez." if compact else
            "İşaretin içindeki sayı = nokta adedi, yanındaki yazı = Habitat_ID; ayrıntı için üzerine gelin.\n"
            "Köşegen = aynı SAYIDA tür; türlerin de aynı olduğu anlamına gelmez.")
    fig.text(0.5, 0.005, foot, ha="center", va="bottom", fontsize=8.5, color="#4b5563", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.1 if compact else 0.07, 1, 1))
    return fig


def find_marker_at(fig, px: float, py: float, slack_px: float = 5.0):
    """Ekran pikseli (px, py) bir işaretin üzerindeyse (uzman, model) anahtarını döndürür; değilse None."""
    ax = fig.axes[0]
    best, best_d = None, None
    for (x, y), size in getattr(fig, "marker_sizes", {}).items():
        cx, cy = ax.transData.transform((x, y))
        radius = (size ** 0.5) / 2 * fig.dpi / 72.0 + slack_px
        d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if d <= radius and (best_d is None or d < best_d):
            best, best_d = (x, y), d
    return best
