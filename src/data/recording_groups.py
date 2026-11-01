"""
Aynı kaydın iki kanalını/sürümünü (…L.wav / …M.wav) tek kayıt kimliğinde birleştirir.

NEDEN: Adı yalnızca son harfle (L / M) ayrılan iki dosya aynı kaydın iki kanalı/sürümüdür (kullanıcı
doğruladı, 2026-10-20). Bunlar farklı "kayıt" sayılırsa kayıt-bazlı bölmede biri eğitime, biri teste
düşebilir (aynı ses olayı iki tarafta: veri sızıntısı).

KURAL (kullanıcının tanımı): ad `<taban><L|M><uzantı>` biçiminde, taban ve uzantı aynı ve hem L hem M
varsa tek kayıttır. Birleşen kimlik: `<taban>[LM]<uzantı>` (örn. Tsu5_1[LM].wav). Diğer kayıt adları aynen
kalır. Nokta ya da süre eşitliği ARANMAZ; ama eşleşenler arasında süre/nokta farkı varsa `merge_conflicts`
bunu raporlar (beklenmeyen bir durum: veri girişi hatası ya da gerçekten farklı kayıt olabilir).
"""
import re
from typing import List

import pandas as pd

from .point_effort_report import parse_seconds

_NAME = re.compile(r"^(?P<base>.+?)(?P<ch>[LM])(?P<ext>\.[A-Za-z0-9]+)?$")


def _clean(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def canonical_recording_ids(df: pd.DataFrame, name_col: str = "Original_Recording_Name", **_ignored) -> pd.Series:
    """Her satır için (birleştirilmiş) kayıt kimliği; df ile aynı indeks. Sütun yoksa boş kimlikler döner."""
    if name_col not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    names = df[name_col].map(_clean)
    chans: dict = {}
    parsed = []
    for nm in names:
        m = _NAME.match(nm) if nm else None
        parsed.append(m)
        if m:
            chans.setdefault((m.group("base"), m.group("ext") or ""), set()).add(m.group("ch"))
    out: List[str] = []
    for nm, m in zip(names, parsed):
        if m and len(chans[(m.group("base"), m.group("ext") or "")]) > 1:   # hem L hem M var
            out.append(f"{m.group('base')}[LM]{m.group('ext') or ''}")
        else:
            out.append(nm)
    return pd.Series(out, index=df.index)


def merge_conflicts(df: pd.DataFrame, name_col: str = "Original_Recording_Name",
                    duration_col: str = "Original_Duration", point_col: str = "Habitat_ID") -> List[str]:
    """Birleştirilen kayıtlar arasında farklı süre veya farklı Habitat_ID varsa uyarı metinleri (beklenmeyen)."""
    canon = canonical_recording_ids(df, name_col)
    names = df[name_col].map(_clean) if name_col in df.columns else canon
    warns = []
    d = pd.DataFrame({"g": canon, "n": names})
    d = d[d["g"] != d["n"]]
    if d.empty:
        return warns
    if duration_col in df.columns:
        d["dur"] = df.loc[d.index, duration_col].map(parse_seconds).map(lambda v: None if v is None else round(v, 3))
    if point_col in df.columns:
        d["pt"] = df.loc[d.index, point_col].map(_clean)
    for g, grp in d.groupby("g"):
        if "dur" in grp and grp["dur"].dropna().nunique() > 1:
            warns.append(f"⚠ {g}: L ve M süreleri farklı ({sorted(set(grp['dur'].dropna()))}); aynı kayıt mı?")
        if "pt" in grp and grp["pt"].replace("", pd.NA).dropna().nunique() > 1:
            warns.append(f"⚠ {g}: L ve M farklı Habitat_ID altında ({sorted(set(grp['pt']))}).")
    return warns


def merged_pairs(df: pd.DataFrame, **kw) -> List[tuple]:
    """Birleştirilen (orijinal ad(lar) -> birleşik kimlik) listesi; raporlama için."""
    name_col = kw.get("name_col", "Original_Recording_Name")
    canon = canonical_recording_ids(df, name_col)
    names = df[name_col].map(_clean)
    res = {}
    for n, c in zip(names, canon):
        if n and n != c:
            res.setdefault(c, set()).add(n)
    return sorted((c, sorted(v)) for c, v in res.items())


def lm_overlap_report(df: pd.DataFrame) -> pd.DataFrame:
    """Birleştirilen L/M çiftleri için: segmentler zaman olarak çakışıyor mu? (SALT OKUNUR tanı)

    Her birleşik kayıt için L ve M segment sayısı, aynı türde zamanı çakışan (L,M) çiftleri ve
    neredeyse birebir aynı pencereler (başlangıç farkı < 0,05 sn) sayılır.
    """
    need = {"Original_Recording_Name", "Start_Time", "End_Time"}
    if not need <= set(df.columns):
        return pd.DataFrame()
    sp_col = next((c for c in ("Scientific_Name", "Turkish_Name") if c in df.columns), None)
    d = df.copy()
    d["_name"] = d["Original_Recording_Name"].map(_clean)
    d["_grp"] = canonical_recording_ids(d)
    d = d[d["_grp"] != d["_name"]]
    d["_s"] = d["Start_Time"].map(parse_seconds)
    d["_e"] = d["End_Time"].map(parse_seconds)
    d["_ch"] = d["_name"].map(lambda n: (_NAME.match(n).group("ch") if _NAME.match(n) else ""))
    rows = []
    for grp, g in d.groupby("_grp"):
        L, M = g[g["_ch"] == "L"], g[g["_ch"] == "M"]
        n_ov = n_same = n_ov_sp = 0
        for _, a in L.iterrows():
            for _, b in M.iterrows():
                if a["_s"] is None or b["_s"] is None or pd.isna(a["_s"]) or pd.isna(b["_s"]):
                    continue
                ov = min(a["_e"], b["_e"]) - max(a["_s"], b["_s"])
                if ov > 0:
                    n_ov += 1
                    if sp_col and _clean(a[sp_col]) == _clean(b[sp_col]):
                        n_ov_sp += 1
                if abs(a["_s"] - b["_s"]) < 0.05:
                    n_same += 1
        rows.append({"Kayit": grp, "L_segment": len(L), "M_segment": len(M), "Cakisan_cift": n_ov,
                     "Cakisan_ayni_tur": n_ov_sp, "Ayni_pencere": n_same})
    return pd.DataFrame(rows)
