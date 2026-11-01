"""
src/ml/recording_level.py
=========================
Segment doğruluğunun ötesinde, tür zenginliği belirleme için daha anlamlı iki değerlendirme:

1. KAYIT BAZLI HATA TABLOSU  - hatalar hangi test kayıtlarından geliyor? (tek bir kayıt mı, yaygın mı?)
2. KAYIT BAŞINA TÜR VARLIĞI  - her test kaydı için "hangi türler var?" sorusu: bir tür kayıtta
   >= `min_segment` segmentte tahmin edildiyse "var" sayılır; gerçek tür kümesiyle karşılaştırılır
   (precision / recall / F1; (kayıt, tür) çiftleri üzerinde).

Neden kayıt başına TEK karar değil? Bir kayıt birden fazla tür içerebilir; segmentleri tek bir etikete
indirgemek yanlış olur. Varlık ölçüsü bunu doğal biçimde ele alır ve tek bir segmentin yanlış
sınıflanmasına karşı dayanıklıdır (min_segment > 1 ile).

Kapsam notu: değerlendirme KAPALI KÜMEDİR; yalnızca modelin eğitildiği türlerin test segmentleri
vardır. Bu yüzden sonuçlar, eğitime alınmayan türlerin bulunduğu gerçek kayıtlar için geçerli bir
doğruluk tahmini değildir.
"""
from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np


def build_prediction_rows(samples, idx, y_true, proba, class_names: Sequence[str]) -> List[dict]:
    """Test segmentleri için satırlar: recording_id, dosya, gercek, tahmin, guven."""
    proba = np.asarray(proba)
    pred = proba.argmax(axis=1)
    rows = []
    for k, i in enumerate(idx):
        rows.append({
            "recording_id": samples[i].recording_id,
            "dosya": os.path.basename(samples[i].filepath),
            "gercek": class_names[int(y_true[k])],
            "tahmin": class_names[int(pred[k])],
            "guven": float(proba[k, pred[k]]),
        })
    return rows


def recording_error_table(rows: List[dict]) -> List[dict]:
    """Kayıt başına: segment sayısı, doğru sayısı, doğruluk, gerçek/tahmin türleri, en sık hata.
    Hata sayısına göre azalan sıralıdır (en çok hata yapılan kayıt üstte)."""
    by = defaultdict(list)
    for r in rows:
        by[r["recording_id"]].append(r)
    out = []
    for rec, rs in by.items():
        n = len(rs)
        ok = sum(r["gercek"] == r["tahmin"] for r in rs)
        truth = Counter(r["gercek"] for r in rs)
        pred = Counter(r["tahmin"] for r in rs)
        errs = Counter((r["gercek"], r["tahmin"]) for r in rs if r["gercek"] != r["tahmin"])
        top = ""
        if errs:
            (a, b), c = errs.most_common(1)[0]
            top = f"{a} -> {b} ({c})"
        out.append({
            "kayit": rec, "n_segment": n, "n_dogru": ok, "n_hata": n - ok, "dogruluk": ok / n,
            "gercek_turler": "; ".join(f"{k} ({v})" for k, v in truth.most_common()),
            "tahmin_turler": "; ".join(f"{k} ({v})" for k, v in pred.most_common()),
            "en_sik_hata": top,
        })
    out.sort(key=lambda d: (-d["n_hata"], d["kayit"]))
    return out


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def presence_metrics(rows: List[dict], min_segment: int = 1, min_conf: float = 0.0) -> dict:
    """(kayıt, tür) çiftleri üzerinde varlık metrikleri.
    Gerçek varlık: türün kayıtta >= 1 test segmenti var. Tahmin varlığı: tür, güveni >= min_conf olan
    en az `min_segment` segmentte tahmin edilmiş. Döndürür: mikro precision/recall/F1, tam-eşleşen
    kayıt oranı ve tür bazlı tablo."""
    by = defaultdict(list)
    for r in rows:
        by[r["recording_id"]].append(r)
    tp = fp = fn = 0
    sp = defaultdict(lambda: [0, 0, 0])      # tür -> [tp, fp, fn]
    exact = 0
    for rec, rs in by.items():
        truth = {r["gercek"] for r in rs}
        cnt = Counter(r["tahmin"] for r in rs if r["guven"] >= min_conf)
        pred = {s for s, c in cnt.items() if c >= min_segment}
        for s in truth & pred:
            sp[s][0] += 1
        for s in pred - truth:
            sp[s][1] += 1
        for s in truth - pred:
            sp[s][2] += 1
        tp += len(truth & pred); fp += len(pred - truth); fn += len(truth - pred)
        exact += truth == pred
    p, r, f = _prf(tp, fp, fn)
    per = []
    for s in sorted(sp):
        a, b, c = sp[s]
        pp, rr, ff = _prf(a, b, c)
        per.append({"tur": s, "tp": a, "fp": b, "fn": c, "precision": pp, "recall": rr, "f1": ff})
    return {"min_segment": min_segment, "min_conf": min_conf, "n_kayit": len(by),
            "n_cift_gercek": tp + fn, "precision": p, "recall": r, "f1": f,
            "tam_eslesen_kayit": exact / len(by) if by else 0.0, "tur_bazli": per}


def presence_from_arrays(y_true, y_pred, recs, n_classes: int, min_segment: int = 1) -> dict:
    """Aynı ölçü, sınıf indeksleri ve kayıt kimlikleri (ör. çapraz doğrulama kat-dışı tahminleri) için."""
    names = [str(i) for i in range(n_classes)]
    rows = [{"recording_id": str(r), "gercek": names[int(a)], "tahmin": names[int(b)], "guven": 1.0}
            for a, b, r in zip(y_true, y_pred, recs)]
    return presence_metrics(rows, min_segment=min_segment)


def dominance_warning(rows: List[dict], share_threshold: float = 0.15, error_share_threshold: float = 0.4,
                      min_errors: int = 5) -> str:
    """Tek bir kayıt test kümesine ya da hatalara baskınsa (tek bölme sonucu kararsız demektir) uyarı metni."""
    if not rows:
        return ""
    errs = recording_error_table(rows)
    n = len(rows)
    top_share = max(e["n_segment"] for e in errs) / n
    n_err = sum(e["n_hata"] for e in errs)
    worst = errs[0]
    big = max(errs, key=lambda e: e["n_segment"])
    parts = []
    if top_share >= share_threshold:
        parts.append(f"{big['kayit']} test kümesinin %{100 * top_share:.0f}'ini")
    if n_err >= min_errors and worst["n_hata"] / n_err >= error_share_threshold:
        parts.append(f"{worst['kayit']} hataların %{100 * worst['n_hata'] / n_err:.0f}'ini")
    if not parts:
        return ""
    return ("⚠ Tek kayıt sonucu belirliyor: " + " ve ".join(parts)
            + " oluşturuyor. Tek bölme sonucu kararsızdır; çapraz doğrulama sonucuna bakın.")


def format_presence_summary(rows: List[dict], thresholds: Sequence[int] = (1, 2)) -> str:
    """Eğitim penceresinde gösterilecek kısa metin."""
    if not rows:
        return ""
    errs = recording_error_table(rows)
    n_err = sum(e["n_hata"] for e in errs)
    lines = []
    if n_err:
        top = errs[0]
        lines.append(f"En çok hata: {top['kayit']} ({top['n_hata']}/{n_err} hata; {top['en_sik_hata']}).")
    for t in thresholds:
        m = presence_metrics(rows, min_segment=t)
        lines.append(f"Kayıt başına tür varlığı (≥{t} segment): kesinlik {m['precision']:.2f}, "
                     f"duyarlılık {m['recall']:.2f}, F1 {m['f1']:.2f}; "
                     f"tür kümesi tam doğru olan kayıt %{100 * m['tam_eslesen_kayit']:.0f} ({m['n_kayit']} kayıt)")
    return "\n".join(lines)


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def write_reports(folder: str, rows: List[dict], thresholds: Sequence[int] = (1, 2)) -> List[str]:
    """test_segment_tahminleri.csv, test_kayit_bazli.csv, test_tur_varligi.csv dosyalarını yazar."""
    os.makedirs(folder, exist_ok=True)
    written = []
    seg = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]
    _write(os.path.join(folder, "test_segment_tahminleri.csv"), seg); written.append("test_segment_tahminleri.csv")
    _write(os.path.join(folder, "test_kayit_bazli.csv"),
           [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()}
            for e in recording_error_table(rows)]); written.append("test_kayit_bazli.csv")
    per = []
    for t in thresholds:
        m = presence_metrics(rows, min_segment=t)
        per.append({"min_segment": t, "tur": "(TÜMÜ - mikro)", "tp": "", "fp": "", "fn": "",
                    "precision": round(m["precision"], 4), "recall": round(m["recall"], 4), "f1": round(m["f1"], 4)})
        per += [{"min_segment": t, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}}
                for d in m["tur_bazli"]]
    _write(os.path.join(folder, "test_tur_varligi.csv"), per); written.append("test_tur_varligi.csv")
    return written
