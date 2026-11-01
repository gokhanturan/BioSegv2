"""
src/data/habitat_info.py
========================
"Habitat Bilgileri" penceresinin verisi: `habitat_data.json` (tanımlı tüm noktalar: habitat tipi, kod, bölge,
koordinat) ile `bird_sound_dataset.csv` (etiketlenmiş segmentler) birleştirilir.

Ne ölçülür (açıkça): bu bir ENVANTER ve KAPSAM özetidir. "Tanımlı nokta" JSON'daki nokta sayısıdır; "etiketli nokta"
CSV'de en az bir segmenti olan noktadır. Etiketsiz bir nokta "tür yok" demek değildir: henüz etiketlenmemiş demektir.
Bu nedenle buradaki sayılardan habitatlar arası tür zenginliği farkı çıkarılamaz.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import pandas as pd

# JSON anahtarları (gerçek dosya Türkçe anahtar kullanır; varsayılan yedek İngilizcedir)
_ALIASES = {
    "nokta": ("habitat_ID", "habitat_id", "id"),
    "habitat_tipi": ("habitat_tipi", "habitat_type"),
    "kodu": ("kodu", "code"),
    "konum": ("konum", "location"),
    "lat": ("lat",),
    "lon": ("lon",),
    "utm_zone": ("utm_zone",),
    "utm_easting": ("utm_easting",),
    "utm_northing": ("utm_northing",),
    "google_maps": ("google_maps",),
}
POINT_COLUMNS = ("nokta", "habitat_tipi", "kodu", "konum", "lat", "lon", "utm_zone", "utm_easting", "utm_northing",
                 "google_maps")
COORD_TOLERANCE = 1e-4      # derece (~10 m); CSV ve JSON koordinatı bundan fazla ayrışırsa uyarılır
LOW_EFFORT_SEGMENTS = 5     # location_map ile aynı eşik: bu kadar segmentten azı "düşük çaba"


def _first(rec: dict, names) -> object:
    for n in names:
        if n in rec and rec[n] is not None:
            return rec[n]
    return None


def _num(x) -> Optional[float]:
    try:
        v = float(str(x).strip().replace(",", "."))
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def points_frame(habitat_data) -> pd.DataFrame:
    """JSON listesi -> tek biçimli DataFrame (sütunlar: POINT_COLUMNS)."""
    rows = []
    for rec in habitat_data or []:
        if not isinstance(rec, dict):
            continue
        r = {k: _first(rec, a) for k, a in _ALIASES.items()}
        pid = _num(r["nokta"])
        if pid is None:
            continue
        r["nokta"] = int(pid)
        for k in ("habitat_tipi", "kodu", "konum", "utm_zone", "google_maps"):
            r[k] = "" if r[k] is None else str(r[k]).strip()
        for k in ("lat", "lon", "utm_easting", "utm_northing"):
            r[k] = _num(r[k])
        rows.append(r)
    return pd.DataFrame(rows, columns=list(POINT_COLUMNS))


def _csv_points(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """CSV'den nokta bazında segment/tür/kayıt sayısı."""
    cols = ["nokta", "segment", "tur", "kayit"]
    if df is None or "Habitat_ID" not in df.columns or len(df) == 0:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    d["_id"] = pd.to_numeric(d["Habitat_ID"], errors="coerce")
    d = d[d["_id"].notna()]
    sp = "Scientific_Name" if "Scientific_Name" in d.columns else ("Turkish_Name" if "Turkish_Name" in d.columns else None)
    rec = "Original_Recording_Name" if "Original_Recording_Name" in d.columns else None
    g = d.groupby(d["_id"].astype(int))
    out = pd.DataFrame({
        "segment": g.size(),
        "tur": g[sp].nunique() if sp else 0,
        "kayit": g[rec].nunique() if rec else 0,
    })
    out.index.name = "nokta"
    return out.reset_index()[cols]


def _group_table(points: pd.DataFrame, df: Optional[pd.DataFrame], key: str, csv_key: str) -> pd.DataFrame:
    """Habitat tipi veya bölge bazında özet satırları."""
    cols = [key, "tanimli", "etiketli", "kapsam", "segment", "tur", "seg_ortanca", "seg_min", "seg_maks", "dusuk_cabali"]
    if points.empty:
        return pd.DataFrame(columns=cols)
    species = {}
    if df is not None and csv_key in df.columns and "Scientific_Name" in df.columns:
        species = df.groupby(csv_key)["Scientific_Name"].nunique().to_dict()
    rows = []
    for name, g in points.groupby(key):
        lab = g[g["segment"] > 0]
        rows.append({
            key: name, "tanimli": len(g), "etiketli": len(lab),
            "kapsam": 100.0 * len(lab) / len(g) if len(g) else 0.0,
            "segment": int(g["segment"].sum()),
            "tur": int(species.get(name, 0)),
            "seg_ortanca": float(lab["segment"].median()) if len(lab) else float("nan"),
            "seg_min": int(lab["segment"].min()) if len(lab) else 0,
            "seg_maks": int(lab["segment"].max()) if len(lab) else 0,
            "dusuk_cabali": int((lab["segment"] < LOW_EFFORT_SEGMENTS).sum()) if len(lab) else 0,
        })
    return pd.DataFrame(rows, columns=cols).sort_values(["segment", "tanimli"], ascending=False).reset_index(drop=True)


def build(habitat_data, df: Optional[pd.DataFrame]) -> dict:
    """Tüm tabloları ve uyarıları üretir.

    Dönüş: {'noktalar','tipler','bolgeler','matris_tanimli','matris_etiketli','uyarilar','ozet'}"""
    pts = points_frame(habitat_data)
    warns: List[str] = []
    if pts.empty:
        return {"noktalar": pts, "tipler": pd.DataFrame(), "bolgeler": pd.DataFrame(),
                "matris_tanimli": pd.DataFrame(), "matris_etiketli": pd.DataFrame(),
                "uyarilar": ["habitat_data.json bulunamadı veya boş."], "ozet": {}}

    dup = pts["nokta"][pts["nokta"].duplicated()].unique()
    if len(dup):
        warns.append(f"habitat_data.json'da yinelenen nokta numarası var: {', '.join(map(str, dup[:8]))}. İlk kayıt kullanıldı.")
        pts = pts.drop_duplicates("nokta", keep="first")

    cp = _csv_points(df)
    pts = pts.merge(cp, on="nokta", how="left")
    for c in ("segment", "tur", "kayit"):
        pts[c] = pts[c].fillna(0).astype(int)
    pts["etiketli"] = pts["segment"] > 0

    # --- tutarlılık denetimleri
    if df is not None and "Habitat_ID" in df.columns:
        ids = pd.to_numeric(df["Habitat_ID"], errors="coerce")
        n_noid = int(ids.isna().sum())
        if n_noid:
            warns.append(f"CSV'de {n_noid} segmentin Habitat_ID değeri yok; bu segmentler nokta tablolarına girmedi.")
        known = set(pts["nokta"])
        unknown = sorted({int(i) for i in ids.dropna().unique()} - known)
        if unknown:
            warns.append(f"CSV'de olup habitat_data.json'da bulunmayan nokta: {', '.join(map(str, unknown[:10]))}.")
        m = pts.set_index("nokta")
        if "Habitat_Type" in df.columns:
            t = df.assign(hid=ids).dropna(subset=["hid"]).drop_duplicates("hid")
            bad = [int(r.hid) for r in t.itertuples() if int(r.hid) in m.index
                   and str(r.Habitat_Type).strip() != m.loc[int(r.hid), "habitat_tipi"]]
            if bad:
                warns.append(f"CSV'deki habitat tipi JSON'dakiyle uyuşmuyor (nokta: {', '.join(map(str, bad[:8]))}).")
        if {"Lat", "Lon"}.issubset(df.columns):
            t = df.assign(hid=ids).dropna(subset=["hid"]).drop_duplicates("hid")
            bad = []
            for r in t.itertuples():
                i = int(r.hid)
                la, lo = _num(r.Lat), _num(r.Lon)
                if i in m.index and la is not None and lo is not None and m.loc[i, "lat"] is not None:
                    if abs(la - m.loc[i, "lat"]) > COORD_TOLERANCE or abs(lo - m.loc[i, "lon"]) > COORD_TOLERANCE:
                        bad.append(i)
            if bad:
                warns.append(f"CSV ve JSON koordinatları ayrışıyor (nokta: {', '.join(map(str, bad[:8]))}).")

    tipler = _group_table(pts, df, "habitat_tipi", "Habitat_Type")
    bolgeler = _group_table(pts, df, "konum", "Location")
    mt = pts.pivot_table(index="konum", columns="habitat_tipi", values="nokta", aggfunc="count", fill_value=0)
    me = pts[pts["etiketli"]].pivot_table(index="konum", columns="habitat_tipi", values="nokta", aggfunc="count",
                                          fill_value=0).reindex(index=mt.index, columns=mt.columns, fill_value=0)

    lab = pts[pts["etiketli"]]
    ozet = {
        "n_tanimli": len(pts), "n_etiketli": len(lab),
        "kapsam": 100.0 * len(lab) / len(pts) if len(pts) else 0.0,
        "n_segment": int(pts["segment"].sum()),
        "n_tip": int(pts["habitat_tipi"].nunique()), "n_tip_etiketli": int(lab["habitat_tipi"].nunique()),
        "n_bolge": int(pts["konum"].nunique()), "n_bolge_etiketli": int(lab["konum"].nunique()),
        "seg_ortanca": float(lab["segment"].median()) if len(lab) else float("nan"),
        "n_dusuk_cabali": int((lab["segment"] < LOW_EFFORT_SEGMENTS).sum()) if len(lab) else 0,
    }
    return {"noktalar": pts, "tipler": tipler, "bolgeler": bolgeler, "matris_tanimli": mt, "matris_etiketli": me,
            "uyarilar": warns, "ozet": ozet}


def insights(res: dict) -> List[tuple]:
    """Veriden otomatik üretilen bulgular: [(tür 'ok'|'info'|'warn', başlık, metin)]. Sayı değiştikçe metin de değişir."""
    o, out = res.get("ozet") or {}, []
    if not o:
        return out
    out.append(("info", "Kapsam",
                f"Tanımlı {o['n_tanimli']} noktanın {o['n_etiketli']} tanesi (%{o['kapsam']:.0f}) etiketli; "
                f"toplam {o['n_segment']} segment. Etiketsiz noktalar 'tür yok' anlamına gelmez, henüz etiketlenmemiştir."))
    tp, bl = res["tipler"], res["bolgeler"]
    if len(tp):
        none = tp[tp["etiketli"] == 0]["habitat_tipi"].tolist()
        low = tp[(tp["etiketli"] > 0) & (tp["etiketli"] < 3)]
        if none:
            out.append(("warn", "Etiketsiz habitat tipi", f"Hiç etiketli noktası olmayan tip: {', '.join(none)}."))
        if len(low):
            out.append(("warn", "Az noktalı habitat tipi",
                        "Etiketli nokta sayısı 3'ten az olan tip: " + ", ".join(
                            f"{r.habitat_tipi} ({r.etiketli})" for r in low.itertuples())
                        + ". Bu tipler için karşılaştırma yapılamaz."))
        top = tp.iloc[0]
        out.append(("info", "En çok segment", f"{top.habitat_tipi}: {top.segment} segment, {top.etiketli} etiketli nokta."))
    if len(bl):
        none = bl[bl["etiketli"] == 0]["konum"].tolist()
        if none:
            out.append(("warn", "Etiketsiz bölge", f"Hiç etiketli noktası olmayan bölge: {', '.join(none)}."))
        lo = bl[(bl["etiketli"] > 0) & (bl["etiketli"] < 3)]
        if len(lo):
            out.append(("warn", "Az noktalı bölge", "Etiketli nokta sayısı 3'ten az: " + ", ".join(
                f"{r.konum} ({r.etiketli})" for r in lo.itertuples()) + "."))
    if o["n_dusuk_cabali"]:
        out.append(("warn", "Düşük çaba",
                    f"{o['n_dusuk_cabali']} etiketli noktada {LOW_EFFORT_SEGMENTS}'ten az segment var; bu noktalar "
                    "doğrudan karşılaştırılamaz."))
    return out
