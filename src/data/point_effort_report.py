"""
Nokta başına kayıt süresi raporu (SALT OKUNUR).

Veri setine, modele ve bölme mantığına dokunmaz; yalnızca bird_sound_dataset.csv'yi okur ve
her Habitat_ID için şunları çıkarır: benzersiz kayıt sayısı, toplam kayıt süresi, etiketli segment
sayısı ve süresi, tür sayısı. Kayıt tekilliği Original_Recording_Name ile belirlenir; süre her kayıt
için bir kez sayılır (aynı kaydın birden çok segmenti çift sayılmaz).

ÖNEMLİ SINIRLAMA (rapor metnine de yazılır): CSV yalnızca etiketlenmiş segmenti olan kayıtları içerir.
Kuşsuz ya da hiç etiketlenmemiş kayıtlar burada görünmez; bu yüzden "toplam süre" = etiketli
segmenti bulunan kayıtların süresidir, noktanın gerçek toplam örnekleme eforu olmayabilir.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SPECIES_COLS = ("Turkish_Name", "Scientific_Name")


def _norm_id(x) -> str:
    t = str(x).strip()
    if t.endswith(".0") and t[:-2].isdigit():
        t = t[:-2]
    return t


def parse_seconds(value) -> Optional[float]:
    """Saniye (ondalık nokta/virgül) ya da 'ss:dd:ss' / 'dd:ss' biçimini saniyeye çevirir; olmazsa None."""
    if value is None:
        return None
    t = str(value).strip()
    if not t or t.lower() == "nan":
        return None
    if ":" in t:
        try:
            parts = [float(p.replace(",", ".")) for p in t.split(":")]
        except ValueError:
            return None
        if len(parts) not in (2, 3) or any(p < 0 for p in parts):
            return None
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec
    try:
        f = float(t.replace(",", "."))
    except ValueError:
        return None
    return f if f == f and f >= 0 else None


def fmt_hms(seconds: Optional[float]) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "-"
    s = int(round(seconds))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


@dataclass
class EffortReport:
    per_point: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_recording: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def format(self) -> str:
        s = self.summary
        L = ["NOKTA BAŞINA KAYIT SÜRESİ RAPORU (salt okunur)"]
        if not s:
            return "\n".join(L + self.warnings)
        L += [
            f"Satır (segment): {s['n_segment']}; etiketli nokta: {s['n_nokta']}; benzersiz kayıt: {s['n_kayit']}; "
            f"tür: {s['n_tur']}",
            f"Toplam kayıt süresi (benzersiz kayıtlar): {fmt_hms(s['toplam_sure_sn'])}",
            f"Nokta başına kayıt sayısı: en az {s['kayit_min']:.0f}, ortanca {s['kayit_med']:.1f}, en çok {s['kayit_max']:.0f}",
        ]
        if s.get("sure_min_sn") is not None:
            L.append(f"Nokta başına toplam süre: en az {fmt_hms(s['sure_min_sn'])}, ortanca {fmt_hms(s['sure_med_sn'])}, "
                     f"en çok {fmt_hms(s['sure_max_sn'])}")
        if s.get("kayit_sure_min_sn") is not None:
            L.append(f"Tek kayıt süresi: en kısa {fmt_hms(s['kayit_sure_min_sn'])}, ortanca "
                     f"{fmt_hms(s['kayit_sure_med_sn'])}, en uzun {fmt_hms(s['kayit_sure_max_sn'])}")
        if s.get("etiket_orani_genel") is not None:
            L.append(f"Etiketli segment süresi / toplam kayıt süresi (genel): %{s['etiket_orani_genel'] * 100:.1f}")
        L += ["", "Nokta bazlı (ilk 40, tür sayısına göre azalan; tamamı CSV'de):"]
        cols = ["Habitat_ID", "Kayit_Sayisi", "Toplam_Sure", "Segment_Sayisi", "Etiketli_Sure_Orani", "Tur_Sayisi"]
        d = self.per_point.copy()
        d["Etiketli_Sure_Orani"] = d["Etiketli_Sure_Orani"].map(lambda v: "-" if pd.isna(v) else f"%{v * 100:.0f}")
        L.append(d[cols].head(40).to_string(index=False))
        L += ["", "SINIRLAMA: CSV yalnızca etiketli segmenti olan kayıtları içerir; kuşsuz/etiketlenmemiş kayıtlar "
              "görünmez. 'Toplam süre' gerçek toplam örnekleme eforu olmayabilir."]
        if self.warnings:
            L += [""] + self.warnings
        return "\n".join(L)


def build_effort_report(df: pd.DataFrame) -> EffortReport:
    rep = EffortReport()
    need = {"Habitat_ID", "Original_Recording_Name"}
    miss = need - set(df.columns)
    if miss:
        rep.warnings.append(f"⚠ Gerekli sütun(lar) yok: {', '.join(sorted(miss))}")
        return rep
    sp_col = next((c for c in SPECIES_COLS if c in df.columns), None)
    d = df.copy()
    d["_hid"] = d["Habitat_ID"].map(lambda x: "" if pd.isna(x) else _norm_id(x))
    d["_rec"] = d["Original_Recording_Name"].map(lambda x: "" if pd.isna(x) else str(x).strip())
    # Aynı kaydın L/M kanalları tek kayıt sayılır (kural: src/data/recording_groups.py).
    from .recording_groups import canonical_recording_ids, merged_pairs, merge_conflicts
    pairs = merged_pairs(d)
    d["_rec"] = canonical_recording_ids(d)
    rep.warnings.extend(merge_conflicts(d.assign(Habitat_ID=d["_hid"])))
    if pairs:
        rep.warnings.append(f"ℹ {len(pairs)} kayıt (L/M kanal çifti) tek kayıt sayıldı: "
                            + "; ".join(f"{c} ← {' + '.join(v)}" for c, v in pairs))
    n_no_hid = int((d["_hid"].isin(["", "nan"])).sum())
    n_no_rec = int((d["_rec"].isin(["", "nan"])).sum())
    if n_no_hid:
        rep.warnings.append(f"⚠ {n_no_hid} segmentte Habitat_ID boş; rapora alınmadı.")
    if n_no_rec:
        rep.warnings.append(f"⚠ {n_no_rec} segmentte Original_Recording_Name boş; rapora alınmadı.")
    d = d[~d["_hid"].isin(["", "nan"]) & ~d["_rec"].isin(["", "nan"])]
    if d.empty:
        rep.warnings.append("⚠ Raporlanacak etiketli satır yok.")
        return rep

    d["_dur_rec"] = d["Original_Duration"].map(parse_seconds) if "Original_Duration" in d.columns else np.nan
    if "Duration" in d.columns:
        seg = d["Duration"].map(parse_seconds)
    else:
        seg = pd.Series(np.nan, index=d.index)
    if {"Start_Time", "End_Time"} <= set(d.columns):
        alt = d["End_Time"].map(parse_seconds) - d["Start_Time"].map(parse_seconds)
        seg = seg.where(seg.notna(), alt.where(alt >= 0))
    d["_seg_dur"] = seg
    if sp_col:
        d["_sp"] = d[sp_col].map(lambda x: "" if pd.isna(x) else str(x).strip())
        d.loc[d["_sp"].str.lower() == "nan", "_sp"] = ""

    # Bir kayıt birden çok noktaya yazılmışsa uyar (kayıt tekilliği Original_Recording_Name'e dayanır).
    multi = d.groupby("_rec")["_hid"].nunique()
    bad = multi[multi > 1]
    if len(bad):
        rep.warnings.append(f"⚠ {len(bad)} kayıt adı birden çok Habitat_ID altında geçiyor "
                            f"(örn. {', '.join(map(str, bad.index[:3]))}). Süreler her noktada ayrı sayıldı.")
    # Aynı kayıt için tutarsız süre
    incons = d.dropna(subset=["_dur_rec"]).groupby(["_hid", "_rec"])["_dur_rec"].nunique()
    n_inc = int((incons > 1).sum())
    if n_inc:
        rep.warnings.append(f"⚠ {n_inc} kayıtta Original_Duration satırlar arasında tutarsız; ilk değer alındı.")

    # Kayıt düzeyi tablo
    rec_rows = []
    for (hid, rec), g in d.groupby(["_hid", "_rec"], sort=True):
        dur = g["_dur_rec"].dropna()
        dur_v = float(dur.iloc[0]) if len(dur) else None
        seg_total = float(g["_seg_dur"].sum()) if g["_seg_dur"].notna().any() else None
        rec_rows.append({
            "Habitat_ID": hid, "Kayit": rec,
            "Kayit_Suresi_sn": dur_v,
            "Segment_Sayisi": int(len(g)),
            "Etiketli_Sure_sn": seg_total,
            "Etiketli_Sure_Orani": (seg_total / dur_v) if (seg_total is not None and dur_v) else None,
            "Tur_Sayisi": int(g["_sp"].replace("", np.nan).nunique()) if sp_col else None,
        })
    rec_df = pd.DataFrame(rec_rows)
    n_nodur = int(rec_df["Kayit_Suresi_sn"].isna().sum())
    if n_nodur:
        rep.warnings.append(f"⚠ {n_nodur} kayıtta Original_Duration okunamadı; o kayıtlar toplam süreye dahil değil.")

    # Nokta düzeyi tablo
    loc_col = "Location" if "Location" in d.columns else None
    typ_col = "Habitat_Type" if "Habitat_Type" in d.columns else None
    pt_rows = []
    for hid, g in d.groupby("_hid", sort=True):
        r = rec_df[rec_df["Habitat_ID"] == hid]
        tot = float(r["Kayit_Suresi_sn"].sum()) if r["Kayit_Suresi_sn"].notna().any() else None
        lab = float(g["_seg_dur"].sum()) if g["_seg_dur"].notna().any() else None
        # Etiket oranı yalnızca süresi bilinen kayıtlar üzerinden (pay ve payda aynı kayıtlar).
        known = r[r["Kayit_Suresi_sn"].notna()]
        lab_known = float(known["Etiketli_Sure_sn"].sum()) if len(known) and known["Etiketli_Sure_sn"].notna().any() else None
        den = float(known["Kayit_Suresi_sn"].sum()) if len(known) else 0.0
        ratio = (lab_known / den) if (lab_known is not None and den > 0) else None
        row = {"Habitat_ID": hid}
        if loc_col:
            m = g[loc_col].dropna().astype(str).str.strip()
            row["Location"] = m.mode().iloc[0] if len(m) and not m.mode().empty else ""
        if typ_col:
            m = g[typ_col].dropna().astype(str).str.strip()
            row["Habitat_Type"] = m.mode().iloc[0] if len(m) and not m.mode().empty else ""
        row.update({
            "Kayit_Sayisi": int(r["Kayit"].nunique()),
            "Toplam_Sure_sn": tot,
            "Toplam_Sure": fmt_hms(tot),
            "Segment_Sayisi": int(len(g)),
            "Etiketli_Sure_sn": lab,
            "Etiketli_Sure_Orani": ratio,
            "Tur_Sayisi": int(g["_sp"].replace("", np.nan).nunique()) if sp_col else None,
        })
        pt_rows.append(row)
    pt = pd.DataFrame(pt_rows)
    if sp_col:
        pt = pt.sort_values(["Tur_Sayisi", "Habitat_ID"], ascending=[False, True]).reset_index(drop=True)
    rep.per_point, rep.per_recording = pt, rec_df

    sure = pt["Toplam_Sure_sn"].dropna()
    rsure = rec_df["Kayit_Suresi_sn"].dropna()
    tot_all = float(rsure.sum()) if len(rsure) else None
    known = rec_df[rec_df["Kayit_Suresi_sn"].notna()]
    lab_all = float(known["Etiketli_Sure_sn"].sum()) if len(known) and known["Etiketli_Sure_sn"].notna().any() else None
    rep.summary = {
        "n_segment": int(len(d)), "n_nokta": int(pt.shape[0]), "n_kayit": int(rec_df.shape[0]),
        "n_tur": int(d["_sp"].replace("", np.nan).nunique()) if sp_col else None,
        "toplam_sure_sn": tot_all,
        "kayit_min": float(pt["Kayit_Sayisi"].min()), "kayit_med": float(pt["Kayit_Sayisi"].median()),
        "kayit_max": float(pt["Kayit_Sayisi"].max()),
        "sure_min_sn": float(sure.min()) if len(sure) else None,
        "sure_med_sn": float(sure.median()) if len(sure) else None,
        "sure_max_sn": float(sure.max()) if len(sure) else None,
        "kayit_sure_min_sn": float(rsure.min()) if len(rsure) else None,
        "kayit_sure_med_sn": float(rsure.median()) if len(rsure) else None,
        "kayit_sure_max_sn": float(rsure.max()) if len(rsure) else None,
        "etiket_orani_genel": (lab_all / tot_all) if (lab_all is not None and tot_all) else None,
    }
    return rep


def write_csvs(folder: str, rep: EffortReport) -> List[str]:
    import os
    files = []
    for name, df in (("nokta_kayit_suresi.csv", rep.per_point), ("nokta_kayit_ayrintisi.csv", rep.per_recording)):
        if df is not None and not df.empty:
            df.to_csv(os.path.join(folder, name), sep=";", index=False, encoding="utf-8-sig")
            files.append(name)
    return files
