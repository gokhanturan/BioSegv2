"""
src/ml/richness_inference.py
============================
Uçtan uca zincir: ETİKETSİZ ses kayıtları -> BirdNET (3 sn) pencere gömmeleri -> sınıflandırıcı
-> nokta (Habitat_ID) başına tür listesi -> tür zenginliği belirleme (belirlenen tür sayısı).

İki kullanım:
  * Doğrulama: uzman etiketli noktalarda, her kaydı eğitimde görmemiş bir modelle tara ve
    ortaya çıkan tür listesini uzman listesiyle karşılaştır (kayıt bazlı çapraz doğrulama).
  * Uygulama: tüm etiketli veriyle eğitilmiş modeli yeni (etiketsiz) kayıtlara uygula.

Önemli tasarım kararları (veri setine bakarak değil, önceden sabitlenmiştir):
  * Kayıt, BirdNET'in 3 sn'lik (örtüşmesiz) pencerelerine bölünür; her pencere için sınıf olasılıkları hesaplanır.
  * Bir pencere, en yüksek olasılık >= tau ise o türün tespiti sayılır.
  * Bir tür bir noktada, en az k BAĞIMSIZ (>= min_gap sn arayla) tespit varsa "var" sayılır.
    Tek seferlik yanlış tespitler belirlenen tür listesine sahte türler ekler; k >= 2 bunu azaltır.
  * tau ve k test sonuçlarına bakılarak seçilmez: ön-tanımlı değerler raporlanır, geri kalan
    kombinasyonlar duyarlılık analizi olarak verilir.

Sınırlar (açıkça):
  * Sınıflandırıcı KAPALI KÜMEDİR: yalnızca eğitimdeki türleri tanır; "kuş yok / bilinmeyen tür"
    sınıfı yoktur. Bu yüzden arka plan pencereleri yanlış tür olarak etiketlenebilir (sahte tespit);
    tau ve k bunu hafifletir, doğrulama modu ise ölçer (kesinlik / sahte tür sayısı).
  * Modelin bilmediği türler zenginliğe girmez; belirlenen tür sayısı, bilinen tür evreni içindir.
"""
from __future__ import annotations

import hashlib
import os
import zlib
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..data import richness_core as R
from . import richness_validation as RV

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


# ---------------------------------------------------------------------------
# Ses dosyalarını bulma
# ---------------------------------------------------------------------------
def build_audio_index(root: str) -> Dict[str, str]:
    """Klasör altındaki ses dosyaları: küçük harfli dosya adı -> tam yol (ilk bulunan)."""
    idx: Dict[str, str] = {}
    for dp, _dn, files in os.walk(root):
        for f in files:
            if f.lower().endswith(AUDIO_EXT):
                idx.setdefault(f.lower(), os.path.join(dp, f))
    return idx


def resolve_audio(path: Optional[str], name: Optional[str], index: Optional[Dict[str, str]]) -> Optional[str]:
    """Önce verilen yol, olmazsa dosya adıyla (index) arar. Bulunamazsa None."""
    if path and os.path.isfile(path):
        return path
    if index:
        for cand in (os.path.basename(str(path or "")).lower(), str(name or "").lower()):
            if cand and cand in index:
                return index[cand]
            if cand and not cand.endswith(AUDIO_EXT):
                for ext in AUDIO_EXT:
                    if cand + ext in index:
                        return index[cand + ext]
    return None


# ---------------------------------------------------------------------------
# Kaydı pencerelere bölüp gömme çıkarma (önbellekli, yarıda kalırsa devam eder)
# ---------------------------------------------------------------------------
def _cache_file(cache_dir: str, filepath: str, hop: float, tag: str = "perch") -> str:
    st = os.stat(filepath)
    raw = f"{os.path.abspath(filepath)}|{st.st_size}|{int(st.st_mtime)}|{hop}"
    if tag != "perch":
        raw += f"|{tag}"
    key = hashlib.sha1(raw.encode()).hexdigest()[:20]
    return os.path.join(cache_dir, key + ".npz")


def scan_recording(classifier, filepath: str, hop: float = 5.0, cache_dir: Optional[str] = None):
    """Kaydın her penceresi için (başlangıç sn, gömme). Döner: (starts[W], emb[W, D]).
    `classifier` `embed_recording(filepath, hop)` sağlıyorsa (BirdNETEmbedder) o kullanılır; aksi halde
    Perch penceresi (`predict_window_embedding`)."""
    cf = None
    tag = getattr(classifier, "cache_tag", "perch")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cf = _cache_file(cache_dir, filepath, hop, tag)
        if os.path.isfile(cf):
            z = np.load(cf)
            return z["starts"], z["emb"]
    if hasattr(classifier, "embed_recording"):
        starts, emb = classifier.embed_recording(filepath, hop)
        if cf:
            np.savez_compressed(cf, starts=starts, emb=emb)
        return starts, emb
    import librosa
    from ..detection.perch_classifier import PERCH_SAMPLE_RATE, generate_windows

    y, sr = librosa.load(filepath, sr=PERCH_SAMPLE_RATE, mono=True)
    wins = generate_windows(y, sr, hop_seconds=hop)
    if not wins:
        raise ValueError(f"Pencere üretilemedi: {filepath}")
    starts = np.array([w[0] for w in wins], dtype=float)
    emb = np.array([classifier.predict_window_embedding(w[2]) for w in wins], dtype=np.float32)
    if cf:
        np.savez_compressed(cf, starts=starts, emb=emb)
    return starts, emb


def build_window_table(records: List[dict], classifier, hop: float = 5.0, cache_dir: Optional[str] = None,
                       progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """
    records: [{"rec_id", "point", "path"}]. Döner: tablo (rec_idx, point, start, emb) + başarısızlar.
    """
    rec_idx, starts, embs, ok_recs, failures = [], [], [], [], []
    for i, r in enumerate(records):
        try:
            s, e = scan_recording(classifier, r["path"], hop, cache_dir)
            rec_idx.append(np.full(len(s), len(ok_recs), dtype=int))
            starts.append(s)
            embs.append(e)
            ok_recs.append(r)
        except Exception as exc:  # noqa: BLE001
            failures.append((r.get("rec_id"), str(exc)))
        if progress:
            progress(i + 1, len(records))
    if not ok_recs:
        raise RuntimeError("Hiçbir kayıt taranamadı: " + (failures[0][1] if failures else "kayıt yok"))
    return {"records": ok_recs, "rec_idx": np.concatenate(rec_idx), "start": np.concatenate(starts),
            "emb": np.vstack(embs), "failures": failures}


# ---------------------------------------------------------------------------
# Pencere tespitlerinden nokta × tür tablosu
# ---------------------------------------------------------------------------
def count_independent(starts: Sequence[float], min_gap: float = 5.0) -> int:
    """Başlangıç zamanları arasında en az min_gap sn olan bağımsız tespit sayısı (açgözlü)."""
    n, last = 0, -np.inf
    for s in sorted(starts):
        if s - last >= min_gap:
            n += 1
            last = s
    return n


def detection_incidence(rec_idx, start, pred, conf, rec_point: Sequence, points: Sequence, n_classes: int,
                        tau: float, k: int = 1, min_gap: float = 5.0) -> np.ndarray:
    """
    (nokta × tür) bool tablo: tür, noktada en az k bağımsız tespite (conf >= tau) sahipse True.
    Bağımsızlık kayıt içinde sayılır; farklı kayıtların tespitleri toplanır.
    """
    pidx = {p: i for i, p in enumerate(points)}
    inc = np.zeros((len(points), n_classes), dtype=bool)
    sel = np.where(conf >= tau)[0]
    groups: Dict[tuple, List[float]] = {}
    for w in sel:
        p = rec_point[rec_idx[w]]
        if p not in pidx:
            continue
        groups.setdefault((pidx[p], int(rec_idx[w]), int(pred[w])), []).append(float(start[w]))
    totals: Dict[tuple, int] = {}
    for (pi, _r, c), st in groups.items():
        totals[(pi, c)] = totals.get((pi, c), 0) + count_independent(st, min_gap)
    for (pi, c), n in totals.items():
        if n >= k:
            inc[pi, c] = True
    return inc


# ---------------------------------------------------------------------------
# Model: pencere olasılıkları
# ---------------------------------------------------------------------------
def _fit_predict(method: str, X_tr, y_tr, X_val, y_val, X_te, n_classes, seed, max_epochs):
    if method == "lr":
        return RV._lr_predict(X_tr, y_tr, X_te, n_classes, seed)
    return RV._mlp_predict(X_tr, y_tr, X_val, y_val, X_te, n_classes, seed, max_epochs)


def out_of_fold_window_proba(table: dict, X: np.ndarray, kept: list, class_names: Sequence[str],
                             method: str = "mlp", n_splits: int = 5, seed: int = 42, max_epochs: int = 100,
                             progress: Optional[Callable[[int, int], None]] = None):
    """
    Her kaydın pencereleri, o kaydı eğitimde görmemiş bir modelden sınıflandırılır.
    Döner: (pred[W], conf[W]) — pencere tablosuyla aynı sırada.
    """
    from .train_classifier import group_kfold_splits

    cls_idx = {c: i for i, c in enumerate(class_names)}
    y = np.array([cls_idx[s.species] for s in kept])
    n_classes = len(class_names)
    fold_of_rec: Dict[str, int] = {}
    splits = group_kfold_splits(kept, n_splits=n_splits, random_state=seed)
    for fi, sp in enumerate(splits):
        for i in sp.test_idx:
            fold_of_rec[kept[i].recording_id] = fi
    W = len(table["start"])
    pred = np.full(W, -1, dtype=int)
    conf = np.zeros(W)
    win_fold = np.empty(W, dtype=int)
    for wi_rec, r in enumerate(table["records"]):
        f = fold_of_rec.get(r["rec_id"])
        if f is None:   # eğitim kümesinde hiç segmenti olmayan kayıt: hiçbir model onu görmemiştir
            f = zlib.crc32(str(r["rec_id"]).encode()) % n_splits
        win_fold[table["rec_idx"] == wi_rec] = f
    for fi, sp in enumerate(splits):
        sel = np.where(win_fold == fi)[0]
        if len(sel):
            Xv = X[sp.val_idx] if len(sp.val_idx) else None
            yv = y[sp.val_idx] if len(sp.val_idx) else None
            proba = _fit_predict(method, X[sp.train_idx], y[sp.train_idx], Xv, yv,
                                 table["emb"][sel], n_classes, seed * 100 + fi, max_epochs)
            pred[sel] = proba.argmax(axis=1)
            conf[sel] = proba.max(axis=1)
        if progress:
            progress(fi + 1, len(splits))
    return pred, conf


# ---------------------------------------------------------------------------
# Doğrulama: tarama sonucu vs uzman listesi
# ---------------------------------------------------------------------------
def validate_scan_vs_expert(table: dict, preds: Sequence[tuple], kept: list, class_names: Sequence[str],
                            point_of_recording: Dict[str, str], taus: Sequence[float], ks: Sequence[int],
                            default_tau: float, default_k: int, min_gap: float = 5.0) -> List[dict]:
    """Her (tekrar, tau, k) için tarama kaynaklı ve uzman nokta×tür tablolarını karşılaştırır."""
    cls_idx = {c: i for i, c in enumerate(class_names)}
    y_true = np.array([cls_idx[s.species] for s in kept])
    pts_seg = np.array([point_of_recording.get(s.recording_id) for s in kept], dtype=object)
    # Adil karşılaştırma: uzman listesi yalnızca GERÇEKTEN TARANAN kayıtların etiketlerinden kurulur.
    # (Ham kayıtların bir kısmı yoksa, taranmayan kayıtların türleri "kaçırılmış" sayılmamalıdır.)
    scanned = {r["rec_id"] for r in table["records"]}
    ok = np.array([(p is not None) and (s.recording_id in scanned) for p, s in zip(pts_seg, kept)])
    if not ok.any():
        raise ValueError("Taranan kayıtlar arasında uzman etiketli ve noktası (Habitat_ID) bilinen kayıt yok.")
    points = sorted({p for p in pts_seg[ok]}, key=str)
    expert = RV.point_incidence(y_true[ok], pts_seg[ok], len(class_names), points)
    rec_point = [r["point"] for r in table["records"]]
    rows = []
    for rep, (pred, conf) in enumerate(preds, start=1):
        for tau in taus:
            for k in ks:
                inc = detection_incidence(table["rec_idx"], table["start"], pred, conf, rec_point, points,
                                          len(class_names), tau, k, min_gap)
                row = {"tekrar": rep, "guven_esigi": tau, "en_az_tespit": k,
                       "onceden_belirlenen": bool(tau == default_tau and k == default_k),
                       "tespit_penceresi_orani": float((conf >= tau).mean())}
                row.update(RV.compare_incidence(expert, inc))
                rows.append(row)
    return rows


def summarize(rows: List[dict]) -> List[dict]:
    """(tau, k) bazında tekrarlar üzerinden ortalama ve s.s."""
    out = []
    keys = sorted({(r["guven_esigi"], r["en_az_tespit"]) for r in rows})
    for tau, k in keys:
        sel = [r for r in rows if r["guven_esigi"] == tau and r["en_az_tespit"] == k]
        d = {"guven_esigi": tau, "en_az_tespit": k, "onceden_belirlenen": sel[0]["onceden_belirlenen"],
             "n_tekrar": len(sel)}
        for key in sel[0]:
            if key in ("tekrar", "guven_esigi", "en_az_tespit", "onceden_belirlenen"):
                continue
            v = np.array([r[key] for r in sel], dtype=float)
            d[key + "_ort"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            d[key + "_ss"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else 0.0
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Uygulama: yeni kayıtlardan nokta başına tür listesi ve zenginlik belirleme
# ---------------------------------------------------------------------------
def apply_model_to_table(table: dict, X: np.ndarray, kept: list, class_names: Sequence[str],
                         method: str = "mlp", seed: int = 42, max_epochs: int = 100):
    """Tüm etiketli veriyle (doğrulama kümesi eğitim sonundan ayrılarak) eğitilen modelle pencereleri sınıflandırır."""
    from .train_classifier import group_aware_split

    cls_idx = {c: i for i, c in enumerate(class_names)}
    y = np.array([cls_idx[s.species] for s in kept])
    if method == "lr":
        proba = RV._lr_predict(X, y, table["emb"], len(class_names), seed)
    else:
        sp = group_aware_split(kept, random_state=seed)   # yalnızca erken durdurma için val ayrılır
        tr = np.concatenate([sp.train_idx, sp.test_idx])   # geri kalan tüm etiketli veri eğitimde
        Xv = X[sp.val_idx] if len(sp.val_idx) else None
        yv = y[sp.val_idx] if len(sp.val_idx) else None
        proba = RV._mlp_predict(X[tr], y[tr], Xv, yv, table["emb"],
                                len(class_names), seed, max_epochs)
    return proba.argmax(axis=1), proba.max(axis=1)


def species_lists(table: dict, pred, conf, class_names: Sequence[str], tau: float, k: int,
                  min_gap: float = 5.0) -> List[dict]:
    """Nokta başına tür listesi: tür, bağımsız tespit sayısı, en yüksek güven."""
    rec_point = [r["point"] for r in table["records"]]
    points = sorted({p for p in rec_point}, key=str)
    inc = detection_incidence(table["rec_idx"], table["start"], pred, conf, rec_point, points,
                              len(class_names), tau, k, min_gap)
    pidx = {p: i for i, p in enumerate(points)}
    best: Dict[tuple, float] = {}
    cnt: Dict[tuple, int] = {}
    per: Dict[tuple, List[float]] = {}
    for w in np.where(conf >= tau)[0]:
        key = (pidx[rec_point[table["rec_idx"][w]]], int(pred[w]))
        best[key] = max(best.get(key, 0.0), float(conf[w]))
        per.setdefault((key[0], key[1], int(table["rec_idx"][w])), []).append(float(table["start"][w]))
    for (pi, c, _r), st in per.items():
        cnt[(pi, c)] = cnt.get((pi, c), 0) + count_independent(st, min_gap)
    rows = []
    for pi, c in zip(*np.where(inc)):
        rows.append({"nokta": points[pi], "tur": class_names[c], "bagimsiz_tespit": cnt.get((pi, c), 0),
                     "en_yuksek_guven": round(best.get((pi, c), 0.0), 3)})
    return sorted(rows, key=lambda r: (str(r["nokta"]), -r["bagimsiz_tespit"]))


def richness_grid(table: dict, pred, conf, class_names: Sequence[str], taus: Sequence[float],
                  ks: Sequence[int], default_tau: float, default_k: int, min_gap: float = 5.0) -> List[dict]:
    """(tau, k) kombinasyonları için belirlenen tür sayıları (birim = nokta)."""
    rec_point = [r["point"] for r in table["records"]]
    points = sorted({p for p in rec_point}, key=str)
    rows = []
    for tau in taus:
        for k in ks:
            inc = detection_incidence(table["rec_idx"], table["start"], pred, conf, rec_point, points,
                                      len(class_names), tau, k, min_gap)
            st = R.incidence_stats(inc)
            d = {"guven_esigi": tau, "en_az_tespit": k,
                 "onceden_belirlenen": bool(tau == default_tau and k == default_k),
                 "nokta_sayisi": st.m, "belirlenen_tur": st.s_obs}
            rows.append(d)
    return rows
