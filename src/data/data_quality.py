"""
Veri kalitesi denetimi. `audit()` SALT-OKUNURDUR: CSV'yi ve dosyaları değiştirmez, yalnızca raporlar.
Değişiklik yapan tek yer `apply_duplicate_removal()`'dır; yalnızca kullanıcı iki kez onaylarsa
arayüzden çağrılır, CSV'nin yedeğini alır ve silinen ses dosyalarını yedek klasöre TAŞIR (geri alınabilir).

Kontroller (hepsi sütun varsa çalışır, yoksa atlanır; veri setinden bağımsızdır):
  1. Yinelenen segment: aynı kayıt + aynı başlangıç/bitiş + aynı tür.
  2. Örtüşen pencere: aynı kayıtta ardışık segmentlerin zaman aralıkları kesişiyor
     (meşru olabilir; "incelenmeli" olarak sunulur, hata sayılmaz).
  3. Boş zorunlu alanlar.
  4. Tutarsızlık: bir kayıt birden çok konuma/habitat noktasına atanmış; bir nokta birden
     çok habitat tipine atanmış; bitiş <= başlangıç.
  5. Ses dosyası diskte var mı (segment dosyaları; File_Path, yoksa verilen klasörlerde Filename);
     CSV satır sayısı ile klasördeki wav sayısı uyuşmuyorsa HANGİ satırların dosyasının eksik olduğu
     (Excel satır numarasıyla) ve CSV'de hiç geçmeyen (yetim) wav dosyaları listelenir.
  6. Etki: yinelenenler çıkarılsaydı eğitim süzgecini geçen tür/segment sayısı nasıl değişirdi.
"""
from __future__ import annotations

import csv
import datetime as _dt
import os
import unicodedata
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .classification_folders import CLASSIFICATION_FOLDER_NAME, _sanitize_folder_name

REQUIRED = ["Original_Recording_Name", "Filename", "Start_Time", "End_Time",
            "Scientific_Name", "Turkish_Name", "Location", "Habitat_ID", "Habitat_Type", "Call_Type"]
DUP_KEY = ["Original_Recording_Name", "Start_Time", "End_Time", "Scientific_Name"]
MIN_SEGMENT, MIN_RECORDINGS = 30, 2   # dataset_builder.DEFAULT_MIN_SEGMENTS / cv_runner / dataset_summary ile aynı (testle korunur)
# Yinelenen grup içinde farklı olması "aynı ötüş, farklı dosya adı" demek olan (önemsiz) sütunlar.
TRIVIAL_DIFF_COLS = {"Filename", "File_Path", "type_Count", "Background_Order", "Background_Family",
                     "Background_Scientific_Name", "Background_Turkish_Name", "Background_Noise_Type"}
MAX_LISTED = 40   # raporda tek tek yazılacak en fazla eksik satır / yetim dosya


def read_text(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return None
    return df if len(df) else None


@dataclass
class QualityReport:
    n_rows: int
    duplicate_rows: pd.DataFrame = field(default_factory=pd.DataFrame)     # tüm üyeler (kopya grupları)
    n_duplicate_extra: int = 0                                              # fazladan (silinecek sayıda) satır
    n_duplicate_recordings: int = 0
    overlap_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_overlap_pairs: int = 0
    empty_counts: Dict[str, int] = field(default_factory=dict)
    inconsistencies: List[str] = field(default_factory=list)
    bad_interval_rows: int = 0
    files_checked: Optional[int] = None
    files_missing: Optional[int] = None
    missing_rows: pd.DataFrame = field(default_factory=pd.DataFrame)       # dosyası bulunamayan CSV satırları
    orphan_files: List[str] = field(default_factory=list)                  # klasörde olup CSV'de geçmeyen wav'lar
    n_files_on_disk: Optional[int] = None
    n_unique_filenames: Optional[int] = None
    scanned_dirs: List[str] = field(default_factory=list)
    impact: Optional[dict] = None
    skipped: List[str] = field(default_factory=list)
    # --- ayrıntılar (hepsi veriden hesaplanır) ---
    n_recordings: int = 0
    n_species: int = 0
    n_dup_groups: int = 0
    dup_group_sizes: Dict[int, int] = field(default_factory=dict)          # grup boyutu -> grup sayısı
    dup_species: List[tuple] = field(default_factory=list)                 # (tür, fazla, toplam)
    dup_recordings: List[tuple] = field(default_factory=list)              # (kayıt, fazla)
    dup_differing_cols: List[str] = field(default_factory=list)            # yinelenen grup içinde değeri farklı sütunlar
    n_overlap_rows: int = 0
    n_overlap_recordings: int = 0
    max_overlap_s: float = 0.0
    impact_lost: List[tuple] = field(default_factory=list)                 # (tür, segment_şimdi, segment_sonra, kayıt_sonra)

    def suspicious_rows(self) -> pd.DataFrame:
        """Dışa aktarılacak şüpheli satırlar (neden sütunuyla)."""
        parts = []
        if len(self.duplicate_rows):
            parts.append(self.duplicate_rows.assign(Neden="Yinelenen segment"))
        if len(self.overlap_rows):
            parts.append(self.overlap_rows.assign(Neden="Örtüşen pencere (incelenmeli)"))
        if len(self.missing_rows):
            parts.append(self.missing_rows.assign(Neden="Ses dosyası eksik"))
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts)
        out.insert(0, "CSV_Satir", out.index + 2)   # Excel satır numarası (başlık = 1)
        return out

    def format_text(self) -> str:
        L = []
        # ---- Özet ----
        L.append(f"Denetlenen veri: {self.n_rows} segment, {self.n_recordings} kayıt, {self.n_species} tür")
        flags = []
        if self.n_duplicate_extra:
            flags.append(f"{self.n_duplicate_extra} yinelenen segment")
        if self.n_overlap_pairs:
            flags.append(f"{self.n_overlap_pairs} örtüşen pencere çifti (incelenmeli)")
        bad_empty = {k: v for k, v in self.empty_counts.items() if v}
        if bad_empty:
            flags.append(f"{sum(bad_empty.values())} boş zorunlu alan")
        flags += [x.lstrip("⚠ ") for x in self.inconsistencies]
        if self.bad_interval_rows:
            flags.append(f"{self.bad_interval_rows} geçersiz zaman aralığı")
        if self.files_missing:
            flags.append(f"{self.files_missing} satırın ses dosyası eksik")
        if self.orphan_files:
            flags.append(f"{len(self.orphan_files)} wav dosyası CSV'de yok")
        L.append("Özet: " + ("; ".join(flags) + "." if flags else "✓ Denetlenen kontrollerde sorun bulunmadı."))
        L.append("")

        # ---- 1 ----
        L.append("1) Yinelenen segment (aynı kayıt + aynı zaman aralığı + aynı tür)")
        if "yinelenen" in self.skipped:
            L.append("   Atlandı (gerekli sütunlar yok).")
        elif not self.n_duplicate_extra:
            L.append("   ✓ Yok.")
        else:
            pct = 100 * self.n_duplicate_extra / self.n_rows
            sizes = ", ".join(f"{v} grup × {k} satır" for k, v in self.dup_group_sizes.items())
            L.append(f"   {self.n_dup_groups} yinelenen grup bulundu ({sizes}); {self.n_duplicate_recordings} kayıtta.")
            L.append(f"   Her grupta 1 satır asıl, kalanlar fazlalık sayılır: toplam {self.n_duplicate_extra} "
                     f"fazlalık satır (%{pct:.1f}). Dışa aktarılan dosyada gruplar eksiksiz (asıl + kopya) yer alır.")
            if self.dup_differing_cols:
                L.append("   Grup içinde değeri farklı olan sütunlar: " + ", ".join(self.dup_differing_cols)
                         + (". Diğer tüm sütunlar aynı: aynı ötüş farklı dosya adıyla tekrar kaydedilmiş görünüyor."
                            if set(self.dup_differing_cols) <= TRIVIAL_DIFF_COLS
                            else ". Bu farklar bilinçli ikinci bir etiketi gösteriyor olabilir."))
            L.append("   En çok etkilenen türler (fazlalık / tür toplamı):")
            for sp, n, tot in self.dup_species[:6]:
                L.append(f"     • {sp}: {n} / {tot} (%{100 * n / tot:.0f})")
            L.append("   En çok etkilenen kayıtlar: " + ", ".join(f"{r} ({n})" for r, n in self.dup_recordings[:5]))
        L.append("")

        # ---- 2 ----
        L.append("2) Örtüşen pencereler (aynı kayıtta ardışık segmentler kesişiyor; yinelenenler hariç)")
        if "ortusen" in self.skipped:
            L.append("   Atlandı.")
        elif not self.n_overlap_pairs:
            L.append("   ✓ Yok.")
        else:
            L.append(f"   {self.n_overlap_pairs} çift ({self.n_overlap_rows} satır), {self.n_overlap_recordings} kayıtta; "
                     f"en büyük örtüşme {self.max_overlap_s:.1f} sn.")
            L.append("   Hata olmayabilir: aynı ötüşün kaydırılmış pencereleri olabilir. Aynı ötüş iki kez sayılıyorsa "
                     "tür başına segment sayısı hafif şişer; kararı dinleyerek siz verin.")
        L.append("")

        # ---- 3 ----
        L.append("3) Boş zorunlu alanlar")
        if not self.empty_counts:
            L.append("   Atlandı.")
        elif bad_empty:
            L.append("   ⚠ " + ", ".join(f"{k}: {v}" for k, v in bad_empty.items()))
        else:
            L.append(f"   ✓ {len(self.empty_counts)} zorunlu alanın hiçbirinde boş değer yok.")
        L.append("")

        # ---- 4 ----
        L.append("4) Tutarlılık (kayıt→konum/nokta, nokta→habitat tipi, bilimsel↔Türkçe ad)")
        L += [f"   {x}" for x in self.inconsistencies] or ["   ✓ Tutarsızlık bulunmadı."]
        if self.bad_interval_rows:
            L.append(f"   ⚠ {self.bad_interval_rows} satırda bitiş ≤ başlangıç.")
        L.append("")

        # ---- 5 ----
        L.append("5) Ses dosyası (CSV satırı ↔ segment klasörü)")
        if self.files_checked is None:
            L.append("   Kontrol edilmedi.")
        elif self.files_missing == self.files_checked:
            L.append(f"   Hiçbir segment dosyası bulunamadı ({self.files_checked}). CSV başka bilgisayarın yollarını "
                     "içeriyor olabilir; proje içindeki 'segments' klasörü de arandı.")
        else:
            L.append(f"   CSV'de {self.files_checked} satır, segment klasöründe "
                     f"{self.n_files_on_disk if self.n_files_on_disk is not None else '?'} wav dosyası var"
                     + (f" (fark: {self.files_checked - self.n_files_on_disk:+d})."
                        if self.n_files_on_disk is not None and self.n_files_on_disk != self.files_checked else "  ✓"))
            L.append(f"   {self.files_checked - self.files_missing} / {self.files_checked} satırın dosyası bulundu"
                     + (f"; {self.files_missing} satırınki eksik ⚠" if self.files_missing else "  ✓"))
            if self.n_files_on_disk is not None and self.n_unique_filenames is not None \
                    and self.n_unique_filenames < self.files_checked:
                L.append(f"   {self.files_checked - self.n_unique_filenames} satır, başka bir satırla AYNI dosya adını "
                         "paylaşıyor (iki satır tek wav'a bakıyor); bu da sayı farkına yol açar.")
            if len(self.missing_rows):
                L.append("   Dosyası bulunamayan satırlar (Excel satır no · tür · kayıt · zaman · dosya adı):")
                for idx, r in self.missing_rows.head(MAX_LISTED).iterrows():
                    L.append(f"     • satır {idx + 2} · {r.get('Scientific_Name', '')} · "
                             f"{r.get('Original_Recording_Name', '')} · {r.get('Start_Time', '')}–"
                             f"{r.get('End_Time', '')} sn · {r.get('Filename', '')}")
                if len(self.missing_rows) > MAX_LISTED:
                    L.append(f"     … ve {len(self.missing_rows) - MAX_LISTED} satır daha "
                             "(hepsi 'Şüpheli Satırları CSV'ye Aktar' dosyasında, Neden = Ses dosyası eksik).")
            if self.orphan_files:
                L.append(f"   Klasörde olup CSV'de hiçbir satırda geçmeyen {len(self.orphan_files)} wav dosyası "
                         "(yetim; otomatik silinmez):")
                for n in self.orphan_files[:MAX_LISTED]:
                    L.append(f"     • {n}")
                if len(self.orphan_files) > MAX_LISTED:
                    L.append(f"     … ve {len(self.orphan_files) - MAX_LISTED} dosya daha.")
            if self.scanned_dirs:
                L.append("   Taranan klasörler: " + "; ".join(self.scanned_dirs))
        L.append("")

        # ---- 6 ----
        L.append("6) Yinelenenlerin sınıflandırmaya etkisi (yalnızca bilgi; hiçbir şey silinmedi)")
        if self.impact:
            i = self.impact
            L.append(f"   Eğitim süzgecini (≥{MIN_SEGMENT} segment ve ≥{MIN_RECORDINGS} kayıt) geçen: "
                     f"şimdi {i['species_now']} tür / {i['segments_now']} segment; "
                     f"yinelenenler çıkarılırsa {i['species_dedup']} tür / {i['segments_dedup']} segment.")
            if self.impact_lost:
                L.append("   Eşiğin altına düşecek türler (segment şimdi → sonra, kayıt sonra):")
                for sp, now, after, rec in self.impact_lost:
                    L.append(f"     • {sp}: {now} → {after} segment, {rec} kayıt")
            else:
                L.append("   Hiçbir tür eşiğin altına düşmüyor.")
            L.append("   Yinelenenlerin gerçek tekrar mı bilinçli kesim mi olduğuna veriyi etiketleyen karar vermelidir.")
        elif self.n_duplicate_extra:
            L.append("   Hesaplanamadı (gerekli sütunlar yok).")
        else:
            L.append("   Yinelenen olmadığı için etki yok.")
        return "\n".join(L)


def _trainable(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Scientific_Name").agg(n=("Filename", "size"), r=("Original_Recording_Name", "nunique"))
    return g[(g.n >= MIN_SEGMENT) & (g.r >= MIN_RECORDINGS)]


def audit(df: pd.DataFrame, search_dirs: Optional[List[str]] = None, check_files: bool = True) -> QualityReport:
    """df: dtype=str okunmuş CSV (read_text). Hiçbir şeyi değiştirmez."""
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    rep = QualityReport(n_rows=len(df))
    if "Original_Recording_Name" in df.columns:
        rep.n_recordings = int(df.loc[df.Original_Recording_Name != "", "Original_Recording_Name"].nunique())
    if "Scientific_Name" in df.columns:
        rep.n_species = int(df.loc[df.Scientific_Name != "", "Scientific_Name"].nunique())

    if set(DUP_KEY).issubset(df.columns):
        mask = df.duplicated(DUP_KEY, keep=False)
        rep.duplicate_rows = df[mask].sort_values(DUP_KEY)
        rep.n_duplicate_extra = int(df.duplicated(DUP_KEY, keep="first").sum())
        rep.n_duplicate_recordings = int(df[mask]["Original_Recording_Name"].nunique())
        if mask.any():
            grp = df[mask].groupby(DUP_KEY)
            sizes = grp.size()
            rep.n_dup_groups = int(len(sizes))
            rep.dup_group_sizes = {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()}
            extra = df[df.duplicated(DUP_KEY, keep="first")]
            tot = df["Scientific_Name"].value_counts()
            rep.dup_species = [(sp, int(n), int(tot[sp])) for sp, n in
                               extra["Scientific_Name"].value_counts().items()]
            rep.dup_recordings = [(r, int(n)) for r, n in
                                  extra["Original_Recording_Name"].value_counts().items()]
            rep.dup_differing_cols = [c for c in df.columns
                                      if (df[mask].groupby(DUP_KEY)[c].nunique() > 1).any()]
    else:
        rep.skipped.append("yinelenen")

    if {"Original_Recording_Name", "Start_Time", "End_Time"}.issubset(df.columns):
        # Yinelenenler zaten (1)'de sayıldı; örtüşme, onlar çıkarılmış hâl üzerinden hesaplanır.
        base = df.drop_duplicates(DUP_KEY) if set(DUP_KEY).issubset(df.columns) else df
        t = base.assign(_s=pd.to_numeric(df["Start_Time"].str.replace(",", "."), errors="coerce"),
                      _e=pd.to_numeric(df["End_Time"].str.replace(",", "."), errors="coerce"))
        rep.bad_interval_rows = int((t["_e"] <= t["_s"]).sum())
        idx, max_ov = [], 0.0
        for _, g in t.dropna(subset=["_s", "_e"]).sort_values(["Original_Recording_Name", "_s"]).groupby(
                "Original_Recording_Name"):
            s, e, ix = g["_s"].values, g["_e"].values, list(g.index)
            for k in range(1, len(g)):
                if s[k] < e[k - 1]:
                    idx += [ix[k - 1], ix[k]]
                    max_ov = max(max_ov, float(min(e[k], e[k - 1]) - s[k]))
        rep.n_overlap_pairs = len(idx) // 2
        rep.overlap_rows = df.loc[sorted(set(idx))]
        rep.n_overlap_rows = len(set(idx))
        rep.n_overlap_recordings = int(rep.overlap_rows["Original_Recording_Name"].nunique()) if idx else 0
        rep.max_overlap_s = max_ov
    else:
        rep.skipped.append("ortusen")

    rep.empty_counts = {c: int((df[c] == "").sum()) for c in REQUIRED if c in df.columns}

    if {"Original_Recording_Name", "Location"}.issubset(df.columns):
        n = int((df[df.Location != ""].groupby("Original_Recording_Name").Location.nunique() > 1).sum())
        if n:
            rep.inconsistencies.append(f"⚠ {n} kayıt birden çok konuma atanmış.")
    if {"Original_Recording_Name", "Habitat_ID"}.issubset(df.columns):
        n = int((df[df.Habitat_ID != ""].groupby("Original_Recording_Name").Habitat_ID.nunique() > 1).sum())
        if n:
            rep.inconsistencies.append(f"⚠ {n} kayıt birden çok habitat noktasına atanmış.")
    if {"Habitat_ID", "Habitat_Type"}.issubset(df.columns):
        sub = df[(df.Habitat_ID != "") & (df.Habitat_Type != "")]
        n = int((sub.groupby("Habitat_ID").Habitat_Type.nunique() > 1).sum())
        if n:
            rep.inconsistencies.append(f"⚠ {n} habitat noktası birden çok habitat tipine atanmış.")
    if {"Scientific_Name", "Turkish_Name"}.issubset(df.columns):
        sub = df[(df.Scientific_Name != "") & (df.Turkish_Name != "")]
        n = int((sub.groupby("Scientific_Name").Turkish_Name.nunique() > 1).sum()
                + (sub.groupby("Turkish_Name").Scientific_Name.nunique() > 1).sum())
        if n:
            rep.inconsistencies.append(f"⚠ {n} tür adı eşleşmesi tutarsız (bilimsel ↔ Türkçe ad).")

    if check_files and "Filename" in df.columns:
        dirs = [d for d in (search_dirs or []) if d]
        miss_idx = []
        for idx, r in df.iterrows():
            fp = r.get("File_Path", "")
            ok = bool(fp) and os.path.exists(fp)
            if not ok:
                ok = bool(r["Filename"]) and any(os.path.exists(os.path.join(d, r["Filename"])) for d in dirs)
            if not ok:
                miss_idx.append(idx)
        rep.files_checked, rep.files_missing = len(df), len(miss_idx)
        rep.missing_rows = df.loc[miss_idx]
        names = df.loc[df.Filename != "", "Filename"]
        rep.n_unique_filenames = int(names.map(_key).nunique()) + int((df.Filename == "").sum())
        on_disk = _wavs_on_disk(dirs)
        if on_disk:
            rep.scanned_dirs = [os.path.abspath(d) for d in dirs if os.path.isdir(d)]
            rep.n_files_on_disk = len(on_disk)
            csv_names = set(names.map(_key))
            rep.orphan_files = sorted(n for low, n in on_disk.items() if low not in csv_names)

    if rep.n_duplicate_extra and {"Scientific_Name", "Filename", "Original_Recording_Name"}.issubset(df.columns):
        ok = df[(df.Scientific_Name != "") & (df.Original_Recording_Name != "")]
        a = _trainable(ok)
        b = _trainable(ok.drop_duplicates(DUP_KEY))
        rep.impact = {"species_now": len(a), "segments_now": int(a.n.sum()),
                      "species_dedup": len(b), "segments_dedup": int(b.n.sum()),
                      "species_lost": sorted(set(a.index) - set(b.index))}
        rep.impact_lost = [(sp, int(a.loc[sp, "n"]), int(ok.drop_duplicates(DUP_KEY).query("Scientific_Name == @sp").shape[0]),
                            int(ok.drop_duplicates(DUP_KEY).query("Scientific_Name == @sp").Original_Recording_Name.nunique()))
                           for sp in rep.impact["species_lost"]]
    return rep


def _key(name) -> str:
    """Dosya adı karşılaştırma anahtarı: Unicode NFC + casefold.

    pandas 3 (`Series.str.lower`) 'İ' harfini 'i', Python `str.lower` ise 'i̇' (i + U+0307)
    yaptığından iki yöntem karıştırılınca 'İspinoz…' gibi adlar yanlışlıkla yetim görünüyordu.
    Hem CSV hem disk tarafı bu tek işlevi kullanır.
    """
    return unicodedata.normalize("NFC", str(name)).casefold()


def _wavs_on_disk(dirs: List[str]) -> Dict[str, str]:
    """Verilen klasörlerin ÜST düzeyindeki .wav dosyaları: {küçük harfli ad: gerçek ad}."""
    out: Dict[str, str] = {}
    for d in dict.fromkeys(os.path.abspath(x) for x in dirs):
        if not os.path.isdir(d):
            continue
        try:
            for n in os.listdir(d):
                if _key(n).endswith(".wav") and os.path.isfile(os.path.join(d, n)):
                    out.setdefault(_key(n), n)
        except OSError:
            continue
    return out


# ---------------------------------------------------------------------------
# Yinelenen satırların silinmesi (YALNIZCA kullanıcı onayıyla çağrılır)
# ---------------------------------------------------------------------------

@dataclass
class RemovalPlan:
    row_numbers: List[int] = field(default_factory=list)       # silinecek satırların Excel satır numaraları
    files_to_move: List[str] = field(default_factory=list)     # yedek klasöre taşınacak wav yolları
    files_kept_shared: List[str] = field(default_factory=list)  # kalan bir satır da kullandığı için DOKUNULMAYAN dosya adları
    n_groups: int = 0

    @property
    def n_rows(self) -> int:
        return len(self.row_numbers)


@dataclass
class RemovalResult:
    rows_removed: int = 0
    files_moved: List[str] = field(default_factory=list)
    files_kept_shared: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    backup_csv: str = ""
    backup_dir: str = ""


def _read_raw(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f, delimiter=";") if r]    # boş satırlar pandas'ta da atlanır
    return (rows[0], rows[1:]) if rows else ([], [])


def _cell(row: List[str], col: Dict[str, int], name: str) -> str:
    i = col.get(name)
    return row[i].strip() if i is not None and i < len(row) else ""


def plan_duplicate_removal(csv_path: str, search_dirs: Optional[List[str]] = None) -> RemovalPlan:
    """
    Hangi satırların ve hangi dosyaların kaldırılacağını HESAPLAR; hiçbir şeyi değiştirmez.
    Her yinelenen grupta ilk satır korunur, sonrakiler silinir (audit() ile aynı kural).
    Bir wav, KALAN başka bir satır tarafından kullanılıyorsa dosyaya dokunulmaz.
    """
    plan = RemovalPlan()
    header, rows = _read_raw(csv_path)
    col = {h.strip(): i for i, h in enumerate(header)}
    if not set(DUP_KEY).issubset(col):
        return plan
    seen, remove_idx, groups = set(), [], set()
    for i, r in enumerate(rows):
        key = tuple(_cell(r, col, c) for c in DUP_KEY)
        if key in seen:
            remove_idx.append(i)
            groups.add(key)
        else:
            seen.add(key)
    plan.n_groups = len(groups)
    plan.row_numbers = [i + 2 for i in remove_idx]
    drop = set(remove_idx)
    kept_names = {_key(_cell(r, col, "Filename")) for i, r in enumerate(rows) if i not in drop}
    dirs = [d for d in (search_dirs or []) if d]
    to_move: Dict[str, str] = {}
    for i in remove_idx:
        r = rows[i]
        fn = _cell(r, col, "Filename")
        if not fn or not _key(fn).endswith(".wav"):
            continue
        if _key(fn) in kept_names:
            if fn not in plan.files_kept_shared:
                plan.files_kept_shared.append(fn)
            continue
        cands = []
        fp = _cell(r, col, "File_Path")
        if fp and _key(os.path.basename(fp)) == _key(fn):
            cands.append(fp)
        folder = _sanitize_folder_name(_cell(r, col, "Scientific_Name"))
        for d in dirs:
            cands.append(os.path.join(d, fn))
            cands.append(os.path.join(d, CLASSIFICATION_FOLDER_NAME, folder, fn))
        for c in cands:
            if os.path.isfile(c):
                to_move.setdefault(os.path.normcase(os.path.abspath(c)), os.path.abspath(c))
    plan.files_to_move = sorted(to_move.values())
    return plan


def apply_duplicate_removal(csv_path: str, search_dirs: Optional[List[str]] = None,
                            now: Optional[_dt.datetime] = None) -> RemovalResult:
    """
    Yinelenen (fazlalık) satırları CSV'den siler ve ses dosyalarını segment klasöründen kaldırır.
    GÜVENLİK: (1) önce CSV'nin yedeği alınır; (2) CSV geçici dosyaya yazılıp atomik olarak değiştirilir;
    (3) ses dosyaları kalıcı silinmez, `<segments>/_silinen_yedek_<zaman>/` altına taşınır (geri alınabilir).
    """
    res = RemovalResult()
    plan = plan_duplicate_removal(csv_path, search_dirs)
    res.files_kept_shared = plan.files_kept_shared
    if plan.n_rows == 0:
        return res
    ts = (now or _dt.datetime.now()).strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(csv_path)
    res.backup_csv = f"{base}.yedek_{ts}{ext or '.csv'}"
    shutil.copy2(csv_path, res.backup_csv)

    header, rows = _read_raw(csv_path)
    drop = {n - 2 for n in plan.row_numbers}
    tmp = csv_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(r for i, r in enumerate(rows) if i not in drop)
    os.replace(tmp, csv_path)
    res.rows_removed = len(drop)

    dirs = [os.path.abspath(d) for d in (search_dirs or []) if d and os.path.isdir(d)]
    if plan.files_to_move:
        root = dirs[0] if dirs else os.path.dirname(os.path.abspath(csv_path))
        res.backup_dir = os.path.join(root, f"_silinen_yedek_{ts}")
        for src in plan.files_to_move:
            try:
                rel = next((os.path.relpath(src, d) for d in dirs
                            if os.path.normcase(src).startswith(os.path.normcase(d) + os.sep)), os.path.basename(src))
                dst = os.path.join(res.backup_dir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                k = 1
                while os.path.exists(dst):
                    stem, e = os.path.splitext(os.path.join(res.backup_dir, rel))
                    dst, k = f"{stem}_{k}{e}", k + 1
                shutil.move(src, dst)
                res.files_moved.append(src)
            except OSError as exc:
                res.errors.append(f"{os.path.basename(src)}: {exc}")
    return res
