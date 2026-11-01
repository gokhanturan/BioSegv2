"""
src/ml/model_significance.py
============================
Kayıtlı model sonuçları arasında EŞLEŞTİRİLMİŞ anlamlılık testi (Model Karşılaştırma penceresi için).

Girdi: her modelin kayıtlı çapraz doğrulama özetindeki kat bazlı sonuçlar (`cv_summary["kat_satirlari"]`) ve
bölme parmak izi (`cv_summary["bolme_izi"]`). İki model YALNIZCA parmak izleri aynıysa (aynı (tekrar, kat) test
kayıtları ve aynı sınıf kümesi) eşleştirilir; aksi halde p değeri HESAPLANMAZ.

Test: her (tekrar, kat) için makro F1 farkı d = A − B; tekrarlı çapraz doğrulamada katlar bağımsız olmadığından
Nadeau–Bengio düzeltmeli eşleştirilmiş t testi:
    t = ort(d) / sqrt((1/J + n_test/n_eğitim_havuzu) · s²),   J = kat sayısı (tekrar × kat), df = J − 1
(n_eğitim_havuzu = n_train + n_val: test dışında kalan tüm segmentler.) Bu bir YAKLAŞIMDIR; Nadeau–Bengio
varyans düzeltmesi tekrarlı rastgele bölmeler için türetilmiştir, tekrarlı k-katlı CV için yaygın kullanılan
uygulamasıdır. p değeri, "bu veri kümesinin yeniden örneklenen eğitim/test bölmelerinde iki algoritma farklı mı?"
sorusunu yanıtlar; YENİ nokta/alan için genelleme iddiası değildir.

Çoklu karşılaştırma: hipotezler AİLELERE ayrılır ve düzeltme her ailenin içinde yapılır (Holm):
  ana      : BirdNET ↔ Perch (sonuçlardan önce belirlenmiş tek ana karşılaştırma; düzeltme gerekmez)
  birlesik : birleşik model ↔ BirdNET ve ↔ Perch (önceden belirlenmiş 2 hipotez; Holm)
  kesif    : kalan tüm çiftler (keşif amaçlı; Holm)
"""
from __future__ import annotations

import csv
import io
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np

ALPHA = 0.05
PRIMARY_PAIR = ("birdnet", "perch")
COMBINED_ID = "birlesik_birdnet_perch"
FAMILY_ORDER = ("ana", "birlesik", "kesif")
FAMILY_LABEL = {"ana": "Ana", "birlesik": "Birleşik (Holm)",
                "kesif": "Keşif (Holm)"}
FAMILY_HELP = ("Ana: önceden belirlenmiş tek karşılaştırma (BirdNET ↔ Perch), düzeltme yok. "
               "Birleşik: önceden belirlenmiş 2 hipotez (birleşik ↔ BirdNET, ↔ Perch), Holm düzeltmeli. "
               "Keşif: kalan tüm çiftler, Holm düzeltmeli; keşif amaçlıdır.")


# ----------------------------------------------------------------------------- çekirdek
def paired_nb_test(rows_a: Sequence[dict], rows_b: Sequence[dict], metric: str = "makro_f1") -> dict:
    """A − B eşleştirilmiş fark ve Nadeau–Bengio düzeltmeli t testi. Katlar özdeş değilse ValueError."""
    from scipy import stats

    A = {(int(r["tekrar"]), int(r["kat"])): r for r in rows_a}
    B = {(int(r["tekrar"]), int(r["kat"])): r for r in rows_b}
    if set(A) != set(B):
        raise ValueError("İki modelin (tekrar, kat) kümeleri aynı değil.")
    keys = sorted(A)
    if len(keys) < 3:
        raise ValueError("Eşleştirme için en az 3 kat sonucu gerekir.")
    for k in keys:
        if int(A[k]["n_test"]) != int(B[k]["n_test"]):
            raise ValueError(f"Kat {k}: test boyutu farklı; bölmeler özdeş değil.")
    d = np.array([float(A[k][metric]) - float(B[k][metric]) for k in keys])
    J = len(d)
    mu = float(d.mean())
    sd = float(d.std(ddof=1))
    ratio = float(np.mean([int(A[k]["n_test"]) / max(1, int(A[k]["n_train"]) + int(A[k].get("n_val", 0)))
                           for k in keys]))
    se_nb = float(np.sqrt((1.0 / J + ratio) * sd ** 2))
    se_naif = sd / np.sqrt(J)
    tcrit = float(stats.t.ppf(1 - ALPHA / 2, J - 1))
    if sd == 0:
        p_nb = p_naif = 1.0 if mu == 0 else 0.0
        t = 0.0 if mu == 0 else float("inf")
    else:
        t = mu / se_nb
        p_nb = float(2 * stats.t.sf(abs(t), J - 1))
        p_naif = float(2 * stats.t.sf(abs(mu / se_naif), J - 1))
    return {"metrik": metric, "n_kat": J, "fark": mu, "fark_ss": sd, "t": float(t),
            "ga_alt": mu - tcrit * se_nb, "ga_ust": mu + tcrit * se_nb,
            "kazanilan_kat": int((d > 0).sum()), "berabere_kat": int((d == 0).sum()),
            "p_nb": p_nb, "p_naif": p_naif}


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    """Holm basamaklı-aşağı düzeltmesi (tek yönlü monoton, en çok 1)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p, kind="stable")):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return [float(x) for x in adj]


def family_of(id_a: str, id_b: str) -> str:
    pair = {id_a, id_b}
    if pair == set(PRIMARY_PAIR):
        return "ana"
    if COMBINED_ID in pair and len(pair & set(PRIMARY_PAIR)) == 1:
        return "birlesik"
    return "kesif"


# ----------------------------------------------------------------------------- kayıtlı sonuçlardan tablo
def _cv_of(st) -> Optional[dict]:
    cv = st.result.training_result.cv_summary
    return cv if isinstance(cv, dict) and "dogruluk" in cv else None


def compare_stored(stored_list, metric: str = "makro_f1") -> dict:
    """[(ModelSpec, StoredResult)] -> {'satirlar': [...], 'uyarilar': [...]}.

    Her satır bir model çifti (A − B). Sıra/yön: ana ve birleşik ailelerde önceden belirlenmiş yön (BirdNET − Perch;
    birleşik − tek model); keşif ailesinde ortalaması yüksek olan model A (işaret okumayı kolaylaştırır, p'yi
    etkilemez). `durum` 'ok' değilse p hesaplanmamıştır ve nedeni yazılıdır."""
    uyarilar: List[str] = []
    models: Dict[str, dict] = {}
    for spec, st in stored_list:
        cv = _cv_of(st)
        if cv is None:
            uyarilar.append(f"{spec.ad}: çapraz doğrulama sonucu yok (tek bölme); teste katılamaz.")
            continue
        if not cv.get("kat_satirlari") or not cv.get("bolme_izi"):
            uyarilar.append(f"{spec.ad}: kat bazlı sonuçlar kayıtlı değil (eski kayıt); teste katılmak için "
                            "çapraz doğrulamayı yeniden çalıştırın.")
            continue
        models[spec.id] = {"spec": spec, "rows": cv["kat_satirlari"], "iz": cv["bolme_izi"],
                           "ort": float(np.mean([r[metric] for r in cv["kat_satirlari"]]))}

    order = {sid: i for i, sid in enumerate([s.id for s, _ in stored_list])}
    rows: List[dict] = []
    for ida, idb in combinations(sorted(models, key=lambda s: order[s]), 2):
        fam = family_of(ida, idb)
        a, b = ida, idb
        if fam == "ana" and a != PRIMARY_PAIR[0]:
            a, b = b, a
        elif fam == "birlesik" and b == COMBINED_ID:
            a, b = b, a
        elif fam == "kesif" and models[b]["ort"] > models[a]["ort"]:
            a, b = b, a
        row = {"aile": fam, "a": models[a]["spec"].ad, "b": models[b]["spec"].ad, "a_id": a, "b_id": b,
               "metrik": metric, "durum": "ok", "p_nb": None, "p_duzeltilmis": None, "anlamli": None}
        if models[a]["iz"] != models[b]["iz"]:
            row["durum"] = "bölmeler farklı (veri/sınıf kümesi ya da ayar farkı): eşleştirilemez"
        else:
            try:
                row.update(paired_nb_test(models[a]["rows"], models[b]["rows"], metric))
            except ValueError as exc:
                row["durum"] = str(exc)
        rows.append(row)

    for fam in FAMILY_ORDER:
        idx = [i for i, r in enumerate(rows) if r["aile"] == fam and r["durum"] == "ok"]
        if not idx:
            continue
        adj = [rows[i]["p_nb"] for i in idx] if fam == "ana" else holm_adjust([rows[i]["p_nb"] for i in idx])
        for i, p in zip(idx, adj):
            rows[i]["p_duzeltilmis"] = p
            rows[i]["anlamli"] = bool(p < ALPHA)
    fam_rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    rows.sort(key=lambda r: (fam_rank[r["aile"]], r["p_duzeltilmis"] if r["p_duzeltilmis"] is not None else 2.0))
    return {"satirlar": rows, "uyarilar": uyarilar}


# ----------------------------------------------------------------------------- gösterim
def _p(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return "<0,001" if v < 0.001 else f"{v:.3f}".replace(".", ",")


def table_cells(row: dict) -> tuple:
    """Pencere tablosu için hücre metinleri."""
    fam = FAMILY_LABEL[row["aile"]]
    if row["durum"] != "ok":
        return (fam, f"{row['a']} − {row['b']}", "—", "—", "—", "—", "—", row["durum"])
    sonuc = "anlamlı fark" if row["anlamli"] else "fark kanıtlanamadı"
    return (fam, f"{row['a']} − {row['b']}", f"{row['fark']:+.3f}".replace(".", ","),
            f"[{row['ga_alt']:+.3f}, {row['ga_ust']:+.3f}]".replace(".", ","),
            f"{row['kazanilan_kat']}/{row['n_kat']}", _p(row["p_nb"]), _p(row["p_duzeltilmis"]), sonuc)


TABLE_COLUMNS = ("Aile", "Karşılaştırma (A − B)", "F1 farkı", "%95 GA", "A kazandı", "p (NB)", "p (Holm)", "Sonuç")
TABLE_COLUMN_HELP = ("Hipotez ailesi (aile içinde çoklu karşılaştırma düzeltmesi yapılır).",
                     "A − B; pozitif fark A'nın üstün olduğu anlamına gelir.",
                     "Kat başına makro F1 farkının ortalaması.",
                     "Nadeau–Bengio varyans düzeltmeli %95 güven aralığı.",
                     "A'nın B'den yüksek makro F1 aldığı kat sayısı / toplam kat.",
                     "Nadeau–Bengio düzeltmeli eşleştirilmiş t testi p değeri (düzeltilmemiş).",
                     "Aile içi Holm düzeltmeli p (Ana ailede tek hipotez olduğundan p (NB) ile aynıdır).",
                     "α = 0,05 için: Holm düzeltmeli p < 0,05 ise 'anlamlı fark'; aksi halde 'fark kanıtlanamadı' (modellerin eşit olduğu anlamına gelmez: bu veriyle fark gösterilemedi).")


def to_csv(result: dict) -> str:
    """Tez tablosu için CSV metni (UTF-8; ';' ayraç, Excel/Türkçe yerel ayarla uyumlu)."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", lineterminator="\n")
    w.writerow(["aile", "A", "B", "metrik", "n_kat", "fark_A_eksi_B", "fark_ss", "ga_alt", "ga_ust",
                "A_kazandigi_kat", "p_nb", "p_aile_ici_holm", "anlamli_0_05", "durum"])
    for r in result["satirlar"]:
        ok = r["durum"] == "ok"
        w.writerow([r["aile"], r["a"], r["b"], r["metrik"], r.get("n_kat", ""),
                    f"{r['fark']:.6f}" if ok else "", f"{r['fark_ss']:.6f}" if ok else "",
                    f"{r['ga_alt']:.6f}" if ok else "", f"{r['ga_ust']:.6f}" if ok else "",
                    r.get("kazanilan_kat", "") if ok else "",
                    f"{r['p_nb']:.6g}" if ok else "", f"{r['p_duzeltilmis']:.6g}" if ok else "",
                    "" if r["anlamli"] is None else int(r["anlamli"]), r["durum"]])
    return buf.getvalue()
