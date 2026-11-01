"""
src/data/coverage_report.py
===========================
"Olması gereken tasarım" ile etiketlenmiş veriyi karşılaştıran kapsam raporu.

Yazılım belirli bir veri setine göre uyarlanmaz. "Olması gereken tasarım" (örnekleme
noktaları, bölgeler, habitat tipleri) şu sırayla belirlenir:
  1. planned_points_csv (varsa)  -> açık nokta listesi,
  2. nokta kataloğu (habitat_data.json; use_catalog_design=true ise) -> çalışmanın tasarımı,
  3. expected_* alanları (varsa).
Hiçbiri yoksa beklenti varsayılmaz; yalnızca veriden çıkarılabilen genel kontroller yapılır.
Başka bir çalışmaya (örn. tek bölge, 50 nokta) geçmek için nokta kataloğunu değiştirmek yeterlidir.

Yapılandırma alanları (hepsi isteğe bağlı):
  use_catalog_design      : true                               tasarımı nokta kataloğundan al
  expected_locations      : ["A", "B"]                         katalogda olmayan beklenen bölgeler
  expected_habitat_types  : [...]                              beklenen habitat tipleri
  expected_points_total   : null                               tasarımdaki toplam nokta sayısı (katalog varsa gerekmez)
  min_points_per_group    : 10                                 grup başına önerilen en az nokta
  planned_points_csv      : "planli_noktalar.csv"              planlı nokta listesi (; ayraçlı;
                            sütunlar: Habitat_ID, Location, Habitat_Type) -> etiketsiz noktalar sayılır
  check_time_fields       : false                              true ise kayıt tarih/saat alanları raporlanır ve denetlenir
                                                               (varsayılan kapalı: tür sayıları zamana bağlı değildir)
  expected_months         : [4, 5, 6]                          kayıtların beklendiği aylar (1-12); eksik ay uyarılır
  expected_hours          : [5, 10]                            kayıtların beklendiği saat aralığı [başlangıç, bitiş]
  habitat_groups          : {"Makilik": "Açık alan", ...}      habitat tiplerinin ekolojik grupları
                            (karar araştırmacıya aittir; veriye bakarak değil, önceden belirlenmelidir)
  min_verified_fraction   : 0.0                                doğrulanmış etiket oranı uyarı eşiği
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

CONFIG_FILENAME = "zenginlik_ayarlar.json"

DEFAULT_CONFIG: Dict = {
    "use_catalog_design": True,
    "expected_locations": [],
    "expected_habitat_types": [],
    "expected_points_total": None,
    "min_points_per_group": 10,
    "check_time_fields": False,
    "expected_months": [],
    "expected_hours": [],
    "planned_points_csv": None,
    "habitat_groups": {},
    "min_verified_fraction": 0.0,
}


def load_config(path: Optional[str]) -> Dict:
    """Yapılandırmayı okur; dosya yoksa/bozuksa varsayılan (beklentisiz) yapılandırmayı döndürür."""
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                user = json.load(f)
            if isinstance(user, dict):
                cfg.update({k: v for k, v in user.items() if k in DEFAULT_CONFIG})
        except (OSError, ValueError):
            pass
    return cfg


def load_project_config() -> Dict:
    """Proje kökündeki zenginlik_ayarlar.json dosyasını okur (yoksa varsayılan)."""
    try:
        from .csv_handler import get_project_root
        root = get_project_root()
    except Exception:  # noqa: BLE001
        root = os.getcwd()
    cfg = load_config(os.path.join(root, CONFIG_FILENAME))
    pl = cfg.get("planned_points_csv")
    if pl and not os.path.isabs(pl):
        cfg["planned_points_csv"] = os.path.join(root, pl)
    return cfg


def catalog_design(catalog: Optional[List[dict]] = None) -> pd.DataFrame:
    """Nokta kataloğundan (habitat_data.json) tasarım tablosu: Habitat_ID, Location, Habitat_Type."""
    if catalog is None:
        try:
            from .config import HABITAT_DATA
            catalog = HABITAT_DATA
        except Exception:  # noqa: BLE001
            catalog = []
    rows = []
    for h in catalog or []:
        hid = h.get("habitat_ID", h.get("Habitat_ID"))
        if hid in (None, ""):
            continue
        rows.append({
            "Habitat_ID": _norm_id(hid),
            "Location": h.get("konum", h.get("location", h.get("Location"))),
            "Habitat_Type": h.get("habitat_tipi", h.get("habitat_type", h.get("Habitat_Type"))),
        })
    return pd.DataFrame(rows, columns=["Habitat_ID", "Location", "Habitat_Type"]).drop_duplicates("Habitat_ID")


def _read_plan(path: str) -> pd.DataFrame:
    plan = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    if "Habitat_ID" not in plan.columns:
        raise KeyError("Habitat_ID sütunu yok")
    return plan.drop_duplicates("Habitat_ID")


def apply_habitat_groups(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Habitat_Type -> Habitat_Group sütunu ekler (eşlemesi olmayan tip kendi adıyla kalır)."""
    out = df.copy()
    if "Habitat_Type" in out.columns:
        out["Habitat_Group"] = out["Habitat_Type"].map(lambda t: mapping.get(t, t))
    return out


@dataclass
class CoverageReport:
    lines: List[str] = field(default_factory=list)       # kullanıcıya gösterilecek satırlar
    problems: List[str] = field(default_factory=list)    # yalnızca uyarılar (⚠ ile başlayanlar)

    def format(self) -> str:
        return "\n".join(self.lines)


def _norm_id(x) -> str:
    """Habitat_ID'yi karşılaştırılabilir metne çevirir ('12', 12, 12.0 -> '12')."""
    t = str(x).strip()
    if t.endswith(".0") and t[:-2].isdigit():
        t = t[:-2]
    return t


def _points(df: pd.DataFrame) -> pd.DataFrame:
    """Nokta başına tek satır: Habitat_ID, Location, Habitat_Type."""
    cols = [c for c in ("Habitat_ID", "Location", "Habitat_Type") if c in df.columns]
    out = df.dropna(subset=["Habitat_ID"])[cols].copy()
    out["Habitat_ID"] = out["Habitat_ID"].map(_norm_id)
    return out.drop_duplicates("Habitat_ID")


def build_coverage_report(df: pd.DataFrame, config: Optional[Dict] = None,
                          design: Optional[pd.DataFrame] = None) -> CoverageReport:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config or {})
    rep = CoverageReport()
    add = rep.lines.append

    def warn(msg: str):
        rep.problems.append(msg)
        add(f"⚠ {msg}")

    if "Habitat_ID" not in df.columns or df["Habitat_ID"].dropna().empty:
        warn("Habitat_ID alanı boş: nokta temelli kapsam raporu hazırlanamıyor.")
        return rep
    pts = _points(df)
    n = len(pts)
    min_pts = int(cfg.get("min_points_per_group") or 10)

    total = cfg.get("expected_points_total")
    add(f"Etiketlenmiş nokta: {n}" + (f" / tasarımdaki {total} (%{100 * n / total:.0f})" if total else ""))

    # --- Bölgeler ---
    if "Location" in pts.columns:
        per_loc = pts["Location"].fillna("(belirtilmemiş)").value_counts()
        for loc in cfg.get("expected_locations") or []:
            if loc not in per_loc.index:
                warn(f"{loc}: hiç etiketli nokta yok; bu bölgenin ses kayıtlarını etiketleyin.")
        for loc, k in per_loc.items():
            if k < min_pts:
                warn(f"{loc}: yalnızca {k} etiketli nokta (en az {min_pts} önerilir).")
        add("Bölge başına nokta: " + ", ".join(f"{l}={k}" for l, k in per_loc.items()))
    # --- Habitat tipleri ---
    if "Habitat_Type" in pts.columns:
        per_h = pts["Habitat_Type"].fillna("(belirtilmemiş)").value_counts()
        for h in cfg.get("expected_habitat_types") or []:
            if h not in per_h.index:
                warn(f"Habitat tipi '{h}': hiç etiketli nokta yok.")
        small = [f"{h} ({k})" for h, k in per_h.items() if k < min_pts]
        if small:
            warn(f"{len(small)} habitat tipinde etiketli nokta sayısı {min_pts} noktanın altında: " + ", ".join(small)
                 + ". Bu gruplar nokta sayısı az olduğundan güvenilir karşılaştırılamaz; daha geniş habitat gruplarında "
                   "birleştirmeyi düşünün.")

    # --- Tasarım (planlı noktalar): açık liste > katalog ---
    plan = None
    planned_path = cfg.get("planned_points_csv")
    if planned_path:
        if os.path.isfile(planned_path):
            try:
                plan = _read_plan(planned_path)
            except (OSError, KeyError, ValueError) as exc:
                warn(f"Planlı nokta listesi okunamadı: {exc}")
        else:
            warn(f"Planlı nokta listesi bulunamadı: {planned_path}")
    if plan is None and design is not None and len(design):
        plan = design
    if plan is None and cfg.get("use_catalog_design"):
        cat = catalog_design()
        if len(cat):
            plan = cat
    if plan is not None and len(plan):
        plan = plan.copy()
        plan["Habitat_ID"] = plan["Habitat_ID"].map(_norm_id)
        have = set(pts["Habitat_ID"])
        planned_ids = set(plan["Habitat_ID"])
        done = plan[plan["Habitat_ID"].isin(have)]
        miss = plan[~plan["Habitat_ID"].isin(have)]
        add(f"Tasarım: {len(plan)} nokta; etiketli {len(done)} (%{100 * len(done) / len(plan):.0f}), "
            f"etiketsiz {len(miss)}.")
        unknown = sorted(have - planned_ids)
        if unknown:
            warn(f"{len(unknown)} etiketli Habitat_ID tasarım listesinde yok: "
                 + ", ".join(unknown[:8]) + (" …" if len(unknown) > 8 else "")
                 + ". Habitat_ID girişini kontrol edin.")
        if "Location" in plan.columns and plan["Location"].notna().any():
            tot = plan["Location"].fillna("(belirtilmemiş)").value_counts()
            dn = done["Location"].fillna("(belirtilmemiş)").value_counts()
            for loc, k in tot.items():
                got = int(dn.get(loc, 0))
                if got == 0:
                    warn(f"{loc}: tasarımdaki {k} noktadan hiçbiri etiketli değil; bu bölgenin kayıtlarını etiketleyin.")
                elif got < min_pts and got < k:
                    warn(f"{loc}: tasarımdaki {k} noktadan yalnızca {got} tanesi etiketli (en az {min_pts} önerilir).")
            add("Bölge ilerlemesi (etiketli/tasarım): "
                + ", ".join(f"{l}={int(dn.get(l, 0))}/{k}" for l, k in tot.items()))
        if "Habitat_Type" in plan.columns and plan["Habitat_Type"].notna().any() and len(miss):
            cnt = miss["Habitat_Type"].fillna("(belirtilmemiş)").value_counts()
            add("Etiketsiz noktalar (habitat tipi): " + ", ".join(f"{a}={b}" for a, b in cnt.items()))
        if len(miss) and "Location" in miss.columns:
            ids = sorted(miss["Habitat_ID"], key=lambda x: (len(x), x))
            add("Etiketlenmesi gereken Habitat_ID'ler: " + ", ".join(ids[:25]) + (" …" if len(ids) > 25 else ""))

    # --- Zaman alanları güvenilir mi? ---
    if cfg.get("check_time_fields"):
        dates = times = None
        for col, ad in (("Recording_Date", "tarih"), ("Recording_Time", "saat")):
            if col in df.columns:
                v = df[col].dropna().astype(str)
                if not len(v):
                    continue
                vc = v.value_counts()
                top_share = float(vc.iloc[0]) / len(v)
                if vc.size <= 2 or top_share >= 0.8:
                    warn(f"Kayıt {ad} değerlerinin %{100 * top_share:.0f}'i tek değerde ({vc.index[0]}); "
                         "bu alan gerçek kayıt zamanını yansıtmıyor olabilir (dosya kopyalama zamanı?). "
                         "Belirlenen tür sayılarını etkilemez; yalnızca mevsimsellik / gün içi "
                         "analizi ve gruplar arası zaman karışıklığı kontrolünü engeller. Sonuçlar 'bu çalışmanın "
                         "örnekleme penceresinde belirlenen tür sayısı' olarak yorumlanmalıdır.")
                elif col == "Recording_Date":
                    dates = pd.to_datetime(df[col], errors="coerce").dropna()
                else:
                    times = pd.to_datetime(df[col].astype(str), format="%H:%M:%S", errors="coerce").dropna()
                    if times.empty:
                        times = pd.to_datetime(df[col].astype(str), errors="coerce").dropna()
        if dates is not None and len(dates):
            add(f"Kayıt tarihi aralığı: {dates.min().date()} – {dates.max().date()}; aylara göre kayıt: "
                + ", ".join(f"{int(m)}. ay={int(k)}" for m, k in dates.dt.month.value_counts().sort_index().items()))
            for m in cfg.get("expected_months") or []:
                if int(m) not in set(dates.dt.month):
                    warn(f"{int(m)}. ayda hiç kayıt yok; bu dönemde kaydedilen sesleri etiketleyin/ekleyin.")
        if times is not None and len(times):
            hrs = times.dt.hour
            add(f"Kayıt saati aralığı: {int(hrs.min()):02d}:00 – {int(hrs.max()):02d}:59")
            eh = cfg.get("expected_hours") or []
            if len(eh) == 2:
                absent = [h for h in range(int(eh[0]), int(eh[1]) + 1) if h not in set(hrs)]
                if absent:
                    warn("Beklenen saat aralığında kaydı olmayan saatler: " + ", ".join(f"{h:02d}" for h in absent) + ".")

    # --- Etiket doğrulama durumu ---
    if "Verification_Status" in df.columns and cfg.get("min_verified_fraction"):
        vs = df["Verification_Status"].fillna("").astype(str).str.lower()
        negative = vs.str.contains("doğrulanmamış|not verified|not_verified")
        positive = vs.str.contains("doğrulamalı|doğrulan|verified|onaylan|etiketleyen")
        frac = float((positive & ~negative).mean())
        if frac < float(cfg["min_verified_fraction"]):
            warn(f"Doğrulanmış etiket oranı %{100 * frac:.0f} (eşik %{100 * float(cfg['min_verified_fraction']):.0f}).")

    if not rep.problems:
        add("✓ Tanımlı beklentilere göre kapsam eksiği bulunmadı.")
    return rep


def unlabeled_points(df: pd.DataFrame, config: Optional[Dict] = None,
                     design: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Tasarımda olup henüz etiketli olmayan noktalar (etiketleme iş listesi)."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config or {})
    plan = None
    p = cfg.get("planned_points_csv")
    if p and os.path.isfile(p):
        try:
            plan = _read_plan(p)
        except (OSError, KeyError, ValueError):
            plan = None
    if plan is None and design is not None and len(design):
        plan = design
    if plan is None and cfg.get("use_catalog_design"):
        plan = catalog_design()
    if plan is None or plan.empty:
        return pd.DataFrame(columns=["Habitat_ID", "Location", "Habitat_Type"])
    plan = plan.copy()
    plan["Habitat_ID"] = plan["Habitat_ID"].map(_norm_id)
    have = set(_points(df)["Habitat_ID"]) if "Habitat_ID" in df.columns else set()
    out = plan[~plan["Habitat_ID"].isin(have)]
    return out.sort_values(["Location", "Habitat_ID"], key=lambda c: c.astype(str).str.zfill(6)).reset_index(drop=True)
