"""
src/data/dataset_summary.py
================
`bird_sound_dataset.csv` dosyasından özet istatistikleri (toplam kayıt, tür başına
kayıt sayısı, bölge başına kayıt sayısı) hesaplar. Veri setinin olgunluğuna (tür
çeşitliliği ve örnek sayısı yeterli mi) hızlıca bakmak için tasarlanmıştır.

Tür adı alanı `Turkish_Name`'dir (veri seti uyumluluğu için BioSeg v1.0'ın CSV
şemasıyla aynıdır); uygulamanın etiketleme formuyla tutarlı olarak Türkçe
ortak adı tutar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from . import richness_core as _re
from . import coverage_report as _cov

# Makine öğrenmesi uygulamalarında yaygın kullanılan kaba bir alt sınır: bir türün
# "yeterince örneklenmiş" sayılması için gereken kayıt sayısı (daha önceki
# çalışmanızdaki "≥30 kayıt/tür" filtresine bakın). Katı bir bilimsel kural değil,
# veri setinin olgunluğunu hızlıca gözden geçirmek için pratik bir eşiktir.
LOW_SAMPLE_THRESHOLD = 10
GOOD_SAMPLE_THRESHOLD = 30
# Sınıflandırma eğitimine/çapraz doğrulamaya girmek için varsayılan süzgeç
# (src/ml/cv_runner.py ile aynı varsayılanlar): ≥30 segment ve ≥2 farklı kayıt.
MIN_SEGMENT_FOR_TRAINING = 30
MIN_RECORDINGS_FOR_TRAINING = 2


@dataclass
class DatasetSummary:
    total_records: int
    unique_species: int
    unique_locations: int
    unique_habitats: int
    species_counts: pd.Series  # tür adı -> kayıt sayısı, azalan sırada
    location_counts: pd.Series
    habitat_counts: pd.Series
    verification_counts: pd.Series
    low_sample_species: List[Tuple[str, int]] = field(default_factory=list)
    well_sampled_species: List[Tuple[str, int]] = field(default_factory=list)
    habitat_richness: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    location_richness: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    habitat_group_richness: List["GroupRichness"] = field(default_factory=list)
    habitat_accumulation_curves: List["AccumulationCurve"] = field(default_factory=list)
    # --- Nokta temelli (örnekleme birimi = Habitat_ID) analizler; src/data/richness_core.py ---
    n_points: int = 0
    n_species_observed: int = 0                            # tüm noktalarda belirlenen farklı tür sayısı
    type_richness: List["PointGroupRichness"] = field(default_factory=list)   # Habitat_Type (veya yapılandırmadaki grup) başına
    group_label: str = "Habitat tipine"                    # type_richness'in grup türü (metin için)
    coverage_lines: List[str] = field(default_factory=list)  # kapsam raporu satırları (yapılandırma verildiyse)
    location_rarefaction: Optional[pd.DataFrame] = None
    pooled_curve: Optional[pd.DataFrame] = None            # seyreltme: k, beklenen_tur, ss
    # --- Ek özet alanları (hepsi CSV'den türetilir; yoksa None/boş) ---
    n_recordings: Optional[int] = None            # farklı Original_Recording_Name (uzun kayıt dosyası)
    median_per_species: Optional[float] = None    # tür başına segment medyanı
    min_per_species: Optional[int] = None
    trainable_species: Optional[int] = None       # süzgeci (≥30 segment, ≥2 kayıt) geçen tür sayısı
    trainable_segments: Optional[int] = None
    labeled_points: Optional[int] = None          # Habitat_ID dolu farklı nokta sayısı
    call_type_counts: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    # Nokta başına segment, kayıt ve belirlenen tür sayıları
    point_effort: Optional[pd.DataFrame] = None
    point_incidence: Optional[pd.DataFrame] = None   # nokta x tür varlık/yokluk (bool); görselleştirme için
    point_types: Optional[pd.Series] = None          # nokta -> habitat tipi (grup)

    def format_text_summary(self) -> str:
        lines = [f"Toplam segment: {self.total_records}"]
        if self.n_recordings is not None:
            lines.append(f"Farklı kayıt (uzun kayıt dosyası) sayısı: {self.n_recordings}")
        lines += [
            f"Farklı tür sayısı: {self.unique_species}",
            f"Farklı konum sayısı: {self.unique_locations}",
            f"Farklı habitat tipi sayısı: {self.unique_habitats}",
        ]
        if self.labeled_points:
            lines.append(f"Etiketlenmiş nokta (Habitat_ID) sayısı: {self.labeled_points}")
        if self.unique_species > 0:
            avg = self.total_records / self.unique_species
            med = f", medyan {self.median_per_species:.0f}" if self.median_per_species is not None else ""
            mn = f", en az {self.min_per_species}" if self.min_per_species is not None else ""
            lines.append(f"Tür başına segment: ortalama {avg:.1f}{med}{mn}")
        if self.low_sample_species:
            lines.append(
                f"⚠ {len(self.low_sample_species)} türün segment sayısı "
                f"{LOW_SAMPLE_THRESHOLD} değerinden AZ (bu türlerde başarı ölçümü güvenilir değildir)"
            )
        if self.well_sampled_species:
            lines.append(
                f"✓ {len(self.well_sampled_species)} tür, "
                f"{GOOD_SAMPLE_THRESHOLD}+ segmentle yeterince örneklenmiş"
            )
        if self.trainable_species is not None:
            lines.append(
                f"Sınıflandırma süzgecini (≥{MIN_SEGMENT_FOR_TRAINING} segment ve "
                f"≥{MIN_RECORDINGS_FOR_TRAINING} kayıt) geçen: {self.trainable_species} tür, "
                f"{self.trainable_segments} segment"
            )
        if len(self.call_type_counts) > 1:
            top = ", ".join(f"{k}: {v}" for k, v in self.call_type_counts.head(4).items())
            lines.append(f"Çağrı tipi dağılımı: {top}")
        return "\n".join(lines)

    def format_richness_summary(self) -> str:
        """Tür zenginliği BELİRLEME özeti: belirlenen tür sayısı ve nokta başına segment/kayıt sayıları."""
        lines = ["Tür zenginliği belirleme: her nokta/bölge/habitatta belirlenen FARKLI tür sayısı "
                 "(tekrarlayan kayıtlar bir kez sayılır)."]
        # Habitat/bölge karşılaştırması (en zengin / en fakir) bilerek verilmez:
        # sıralama etiketlenen veri kümesine göre değişir.
        lines.append("")
        lines.append("BELİRLENEN TÜR ZENGİNLİĞİ")
        if self.n_points:
            lines.append(f"Etiketlenmiş nokta (Habitat_ID) sayısı: {self.n_points}; "
                         f"belirlenen toplam tür: {self.n_species_observed}.")
        elif len(self.habitat_richness) > 0:
            lines.append(f"Etiketlenmiş nokta (Habitat_ID) sayısı: {len(self.habitat_richness)}")
        if len(self.location_richness) > 0:
            lines.append(f"Etiketli bölge sayısı: {len(self.location_richness)}")
        if len(self.habitat_richness) == 0 and len(self.location_richness) == 0:
            lines.append("(Henüz Habitat_ID/Location alanı doldurulmuş segment yok.)")
        lines.append(
            "Belirlenen tür sayısı, etiketlenmiş noktalarda GÖRÜLEN türlerdir; etiketlenmemiş "
            "noktalar hakkında bilgi vermez."
        )

        pe = self.point_effort
        if pe is not None and len(pe) > 0:
            lines.append("")
            lines.append("NOKTA BAŞINA SEGMENT VE KAYIT")
            lines.append(
                f"Nokta başına segment: {int(pe['segment'].min())}–{int(pe['segment'].max())} "
                f"(medyan {pe['segment'].median():.0f}); kayıt: {int(pe['kayit'].min())}–{int(pe['kayit'].max())}; "
                f"belirlenen tür: {int(pe['tur'].min())}–{int(pe['tur'].max())}."
            )
            lines.append("Noktalar ham tür sayısıyla sıralanmaz; karşılaştırma için seyreltilmiş değer kullanılır.")
        if self.type_richness:
            lines.append("")
            lines.append(f"{self.group_label} göre (birim = nokta):")
            for g in self.type_richness:
                lines.append(f"  {g.group}: {g.m} nokta, belirlenen {g.s_obs} tür")
        if self.coverage_lines:
            lines.append("")
            lines.append("Kapsam raporu:")
            lines.extend(f"  {l}" for l in self.coverage_lines)
        return "\n".join(lines)


def load_dataset(csv_path: str) -> Optional[pd.DataFrame]:
    """
    CSV dosyasını gerçekte yazıldığı biçimle tutarlı okur (noktalı virgül
    ayırıcı, utf-8-sig kodlaması). Dosya yoksa ya da boşsa None döndürür.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    return df


def compute_richness_by_group(
    df: pd.DataFrame,
    group_column: str,
    label_column: Optional[str] = None,
    species_column: str = "Turkish_Name",
) -> pd.Series:
    """
    Her grup (örn. her Habitat_ID veya her Location) için o grupta kaydedilen
    FARKLI (benzersiz) tür sayısını (tür zenginliği) hesaplar; aynı türün birden
    çok kaydı zenginliğe yalnızca bir kez sayılır.

    label_column verilirse (örn. Habitat_Type), okunabilirlik için, yalın sayısal
    Habitat_ID yerine o gruptaki en sık değer parantez içinde grup etiketine
    eklenir (örn. "12 (Orman)").

    Boş/belirtilmemiş grup veya tür değerleri zenginlik hesabının dışında
    bırakılır (yanıltıcı "0 kayıt = 0 tür" satırlarını önlemek için).
    """
    if group_column not in df.columns or species_column not in df.columns:
        return pd.Series(dtype=int)

    sub = df[[group_column, species_column] + ([label_column] if label_column and label_column in df.columns else [])].copy()
    sub[group_column] = sub[group_column].astype(str).str.strip()
    sub[species_column] = sub[species_column].astype(str).str.strip()

    sub = sub[(sub[group_column] != "") & (sub[group_column].str.lower() != "nan")]
    sub = sub[(sub[species_column] != "") & (sub[species_column].str.lower() != "nan")]

    if sub.empty:
        return pd.Series(dtype=int)

    richness = sub.groupby(group_column)[species_column].nunique()

    if label_column and label_column in sub.columns:
        # Her grup için en sık etiket değerini bul (mod)
        most_common_label = sub.groupby(group_column)[label_column].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else ""
        )

        def _build_label(group_val):
            label_val = most_common_label.get(group_val, "")
            if label_val:
                return f"{group_val} ({label_val})"
            return str(group_val)

        richness.index = [_build_label(g) for g in richness.index]

    return richness.sort_values(ascending=False)


@dataclass
class PointGroupRichness:
    """Bir grubun (habitat tipi vb.) nokta sayısı ve belirlenen tür sayısı."""
    group: str
    m: int        # gruptaki nokta sayısı
    s_obs: int    # belirlenen farklı tür sayısı


@dataclass
class GroupRichness:
    """Bir grup (habitat/nokta) için belirlenen tür sayısı ve örnekleme eforu."""
    group: str
    raw_group: str  # etiketsiz ham grup değeri (örn. "12 (Orman)" değil yalnızca "12"); eğri hesabı için gerekli
    observed_richness: int  # belirlenen farklı tür sayısı
    n_recordings: int  # bu gruptaki farklı kayıt sayısı
    total_duration_hours: float  # bu gruptaki toplam örnekleme süresi (benzersiz kayıtlar üzerinden)
    richness_per_hour: Optional[float]  # efora göre normalize edilmiş zenginlik (tür / saat)


def _parse_duration_seconds(value) -> Optional[float]:
    """CSV'deki Original_Duration alanını saniye cinsinden float'a çevirir (Türkçe
    ondalık virgülünü de kabul eder); çevrilemezse None döndürür (sessizce
    atlanır, istisna fırlatılmaz)."""
    try:
        f = float(str(value).strip().replace(",", "."))
        if f != f or f < 0:  # NaN veya negatif
            return None
        return f
    except (ValueError, TypeError):
        return None


def compute_group_richness(
    df: pd.DataFrame,
    group_column: str,
    label_column: Optional[str] = None,
    species_column: str = "Turkish_Name",
    recording_column: str = "Original_Recording_Name",
    duration_column: str = "Original_Duration",
) -> List[GroupRichness]:
    """
    Her grup (örn. her Habitat_ID) için şunları hesaplar:
      1. Belirlenen tür zenginliği (S_obs),
      2. Efora göre NORMALİZE edilmiş zenginlik (tür/saat), kayıt sayısı ve toplam süre.

    Kayıt sayısı, aynı kayıttan kesilen segmentler bağımsız örnekler olmadığı için
    segment değil BENZERSİZ kayıt (Original_Recording_Name) sayılarak verilir.
    """
    required = {group_column, species_column, recording_column}
    if not required.issubset(df.columns):
        return []

    cols = [group_column, species_column, recording_column]
    has_duration = duration_column in df.columns
    if has_duration:
        cols.append(duration_column)
    has_label = label_column and label_column in df.columns
    if has_label:
        cols.append(label_column)

    sub = df[cols].copy()
    sub[group_column] = sub[group_column].astype(str).str.strip()
    sub[species_column] = sub[species_column].astype(str).str.strip()
    sub[recording_column] = sub[recording_column].astype(str).str.strip()

    sub = sub[(sub[group_column] != "") & (sub[group_column].str.lower() != "nan")]
    sub = sub[(sub[species_column] != "") & (sub[species_column].str.lower() != "nan")]
    # Kayıt kimliği boşsa insidans hesabı için gerçek bir kayıt kimliği gerekir;
    # eksikse o satır bu hesaptan çıkarılır.
    sub = sub[(sub[recording_column] != "") & (sub[recording_column].str.lower() != "nan")]

    if sub.empty:
        return []

    results: List[GroupRichness] = []

    for group_val, group_df in sub.groupby(group_column):
        species_to_recordings: dict = {}
        for _, row in group_df.iterrows():
            species_to_recordings.setdefault(row[species_column], set()).add(row[recording_column])

        s_obs = len(species_to_recordings)
        unique_recordings = set()
        for recs in species_to_recordings.values():
            unique_recordings |= recs
        m = len(unique_recordings)

        total_duration_hours = None
        richness_per_hour = None
        if has_duration:
            # Süre, her BENZERSİZ kayıt için bir kez sayılır (aynı kaydın birden çok
            # segmentinde çift sayılmamalıdır).
            recording_durations = {}
            for _, row in group_df.iterrows():
                rec_id = row[recording_column]
                if rec_id not in recording_durations:
                    dur = _parse_duration_seconds(row[duration_column])
                    if dur is not None:
                        recording_durations[rec_id] = dur
            if recording_durations:
                total_duration_hours = sum(recording_durations.values()) / 3600.0
                if total_duration_hours > 0:
                    richness_per_hour = s_obs / total_duration_hours

        label_val = group_val
        if has_label:
            most_common = group_df[label_column].mode()
            if not most_common.empty and str(most_common.iloc[0]).strip():
                label_val = f"{group_val} ({most_common.iloc[0]})"

        results.append(GroupRichness(
            group=label_val,
            raw_group=str(group_val),
            observed_richness=s_obs,
            n_recordings=m,
            total_duration_hours=total_duration_hours or 0.0,
            richness_per_hour=richness_per_hour,
        ))

    results.sort(key=lambda r: r.observed_richness, reverse=True)
    return results


@dataclass
class AccumulationCurve:
    """
    Bir grup için, kayıt sayısı arttıkça kümülatif tür sayısının nasıl
    büyüdüğünü gösteren eğri; "örnekleme eforu yeterli miydi" sorusunun
    görsel yanıtı.
    """
    group: str
    x: List[int]  # kayıt sayısı (1..m)
    y_mean: List[float]  # o kadar kayıtta ortalama kümülatif tür sayısı
    y_std: List[float]  # standart sapma (belirsizlik göstergesi)

    @property
    def is_leveling_off(self) -> bool:
        """
        Eğrinin son %20'sindeki ortalama artış, ilk %20'sindeki ortalama artışın
        küçük bir kesrinin (kaba, sezgisel bir eşik) altındaysa eğri "düzleşiyor"
        sayılır; bu, örneklemenin büyük olasılıkla yeterli olduğunu düşündürür.
        Katı bir istatistiksel test değil, hızlı bir görsel/sezgisel özettir.
        """
        n = len(self.y_mean)
        if n < 5:
            return False
        head = max(1, n // 5)
        early_slope = (self.y_mean[head] - self.y_mean[0]) / head
        late_slope = (self.y_mean[-1] - self.y_mean[-1 - head]) / head
        if early_slope <= 0:
            return True
        return late_slope < 0.25 * early_slope


def compute_species_accumulation_curve(
    df: pd.DataFrame,
    group_column: str,
    group_value: str,
    species_column: str = "Turkish_Name",
    recording_column: str = "Original_Recording_Name",
    n_permutations: int = 100,
    random_state: int = 42,
) -> Optional[AccumulationCurve]:
    """
    Belirli bir grup (örn. tek bir Habitat_ID) için örnek bazlı seyreltme
    (rarefaction) eğrisini Monte Carlo yöntemiyle hesaplar: kayıtların sırası
    `n_permutations` kez rastgele karıştırılır, her karıştırmada "ilk k kayıtta
    kümülatif kaç farklı tür var" hesaplanır ve tüm karıştırmaların ORTALAMASI
    alınır; bu, analitik "Mao Tau" formülünün pratik bir yaklaşımıdır.

    Eğri düzleşiyorsa (marjinal kazanç küçükse) o grup için örnekleme büyük
    olasılıkla yeterlidir; hâlâ dikse daha fazla kayıt gerekir.
    """
    required = {group_column, species_column, recording_column}
    if not required.issubset(df.columns):
        return None

    sub = df[df[group_column].astype(str).str.strip() == str(group_value)]
    if sub.empty:
        return None

    recording_to_species: dict = {}
    for _, row in sub.iterrows():
        rec = str(row[recording_column]).strip()
        sp = str(row[species_column]).strip()
        if not rec or rec.lower() == "nan" or not sp or sp.lower() == "nan":
            continue
        recording_to_species.setdefault(rec, set()).add(sp)

    recordings = list(recording_to_species.keys())
    m = len(recordings)
    if m == 0:
        return None

    rng = np.random.default_rng(random_state)
    all_curves = np.zeros((n_permutations, m))

    for p in range(n_permutations):
        order = rng.permutation(recordings)
        seen: set = set()
        for i, rec in enumerate(order):
            seen |= recording_to_species[rec]
            all_curves[p, i] = len(seen)

    return AccumulationCurve(
        group=str(group_value),
        x=list(range(1, m + 1)),
        y_mean=all_curves.mean(axis=0).tolist(),
        y_std=all_curves.std(axis=0).tolist(),
    )


def _compute_point_based_analyses(df: pd.DataFrame, group_column: str = "Habitat_Type") -> dict:
    """
    Nokta (Habitat_ID) temelli zenginlik analizleri. Gerekli sütunlar yoksa ya
    da veri analiz için uygun değilse ilgili alanlar boş bırakılır (sessizce
    yanlış sonuç üretmek yerine).
    """
    out: dict = {}
    if "Habitat_ID" not in df.columns or "Turkish_Name" not in df.columns:
        return out
    try:
        inc = _re.build_incidence_matrix(df, "Habitat_ID", "Turkish_Name")
    except KeyError:
        return out
    if inc.empty:
        return out
    out["n_points"] = int(inc.shape[0])
    out["n_species_observed"] = int(_re.incidence_stats(inc).s_obs)
    out["point_incidence"] = inc
    if group_column in df.columns:
        tmp = df[["Habitat_ID", group_column]].astype(str)
        tmp["Habitat_ID"] = tmp["Habitat_ID"].str.strip()
        tmp = tmp[(tmp["Habitat_ID"] != "") & (tmp["Habitat_ID"].str.lower() != "nan")]
        if not tmp.empty:
            out["point_types"] = tmp.groupby("Habitat_ID")[group_column].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else "")
    out["pooled_curve"] = _re.rarefaction_curve(inc)
    if group_column in df.columns:
        multi = df.groupby("Habitat_ID")[group_column].nunique()
        if not (multi > 1).any():   # bir nokta birden çok gruba atanmışsa gruplama uygun değil
            groups = []
            for g, sub in df.groupby(group_column):
                gi = _re.build_incidence_matrix(sub, "Habitat_ID", "Turkish_Name")
                if not gi.empty:
                    groups.append(PointGroupRichness(group=str(g), m=int(gi.shape[0]),
                                                     s_obs=int(_re.incidence_stats(gi).s_obs)))
            groups.sort(key=lambda r: r.group)
            out["type_richness"] = groups
    if "Location" in df.columns:
        try:
            out["location_rarefaction"] = _re.rarefy_groups(df, "Location", unit_column="Habitat_ID")
        except (KeyError, ValueError):
            pass
    return out


def _extra_stats(df: pd.DataFrame, species_counts: pd.Series) -> dict:
    """Kayıt sayısı, tür başına medyan, eğitim süzgecini geçen türler, nokta ve çağrı tipi özeti."""
    def _clean(col):
        return df[col].astype(str).str.strip().replace({"nan": ""}) if col in df.columns else None

    out: dict = {}
    sp = species_counts[species_counts.index != "(belirtilmemiş)"]
    if len(sp):
        out["median_per_species"] = float(sp.median())
        out["min_per_species"] = int(sp.min())
    rec, tur = _clean("Original_Recording_Name"), _clean("Turkish_Name")
    if rec is not None:
        out["n_recordings"] = int(rec[rec != ""].nunique())
    if rec is not None and tur is not None:
        ok = (rec != "") & (tur != "")
        g = pd.DataFrame({"t": tur[ok], "r": rec[ok]}).groupby("t")["r"].agg(["size", "nunique"])
        good = g[(g["size"] >= MIN_SEGMENT_FOR_TRAINING) & (g["nunique"] >= MIN_RECORDINGS_FOR_TRAINING)]
        out["trainable_species"] = int(len(good))
        out["trainable_segments"] = int(good["size"].sum())
    hid = _clean("Habitat_ID")
    if hid is not None:
        out["labeled_points"] = int(hid[hid != ""].nunique())
        ok = hid != ""
        if ok.any():
            eff = pd.DataFrame({"nokta": hid[ok]})
            eff["tur"] = tur[ok] if tur is not None else ""
            eff["kayit"] = rec[ok] if rec is not None else ""
            eff = eff[eff["tur"] != ""] if tur is not None else eff
            out["point_effort"] = eff.groupby("nokta").agg(
                segment=("nokta", "size"),
                kayit=("kayit", lambda x: int(x[x != ""].nunique())),
                tur=("tur", "nunique"),
            )
    ct = _clean("Call_Type")
    if ct is not None and (ct != "").any():
        out["call_type_counts"] = ct[ct != ""].value_counts()
    return out


def compute_summary(df: pd.DataFrame, config: Optional[dict] = None) -> DatasetSummary:
    """
    Bir DataFrame'den (load_dataset çıktısı) özet istatistikleri hesaplar.
    config verilirse (bkz. coverage_report): habitat tipleri yapılandırmadaki ekolojik gruplara
    birleştirilir ve kapsam raporu eklenir. config None ise davranış değişmez.
    """

    def _value_counts(column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(dtype=int)
        # Boş/NaN değerleri sessizce atmak yerine "(belirtilmemiş)" olarak grupla
        col = df[column].fillna("(belirtilmemiş)").replace("", "(belirtilmemiş)")
        return col.value_counts().sort_values(ascending=False)

    species_counts = _value_counts("Turkish_Name")
    location_counts = _value_counts("Location")
    habitat_counts = _value_counts("Habitat_Type")
    verification_counts = _value_counts("Verification_Status")

    low_sample = [
        (name, int(count)) for name, count in species_counts.items()
        if count < LOW_SAMPLE_THRESHOLD and name != "(belirtilmemiş)"
    ]
    well_sampled = [
        (name, int(count)) for name, count in species_counts.items()
        if count >= GOOD_SAMPLE_THRESHOLD
    ]

    habitat_richness = compute_richness_by_group(df, "Habitat_ID", label_column="Habitat_Type")
    location_richness = compute_richness_by_group(df, "Location")
    habitat_group_richness = compute_group_richness(df, "Habitat_ID", label_column="Habitat_Type")

    # Birikim eğrileri yalnızca en bilgilendirici (en çok kayıt yapılan) 5 habitat
    # için hesaplanır; hepsi için hesaplamak hem yavaş olur hem grafiği okunmaz kılar.
    top_habitats = sorted(habitat_group_richness, key=lambda e: e.n_recordings, reverse=True)[:5]
    habitat_accumulation_curves = []
    for est in top_habitats:
        curve = compute_species_accumulation_curve(df, "Habitat_ID", est.raw_group)
        if curve is not None:
            curve.group = est.group  # okunabilir (etiketli) adı kullan
            habitat_accumulation_curves.append(curve)

    point: dict
    cov_lines: List[str] = []
    group_label = "Habitat tipine"
    if config is not None:
        mapping = config.get("habitat_groups") or {}
        group_col = "Habitat_Type"
        if mapping and "Habitat_Type" in df.columns:
            df_g = _cov.apply_habitat_groups(df, mapping)
            group_col, group_label = "Habitat_Group", "Habitat grubuna"
        else:
            df_g = df
        point = _compute_point_based_analyses(df_g, group_col)
        cov_lines = _cov.build_coverage_report(df, config).lines
    else:
        point = _compute_point_based_analyses(df)

    extra = _extra_stats(df, species_counts)

    return DatasetSummary(
        total_records=len(df),
        unique_species=int((species_counts.index != "(belirtilmemiş)").sum()),
        unique_locations=int((location_counts.index != "(belirtilmemiş)").sum()),
        unique_habitats=int((habitat_counts.index != "(belirtilmemiş)").sum()),
        species_counts=species_counts,
        location_counts=location_counts,
        habitat_counts=habitat_counts,
        verification_counts=verification_counts,
        low_sample_species=low_sample,
        well_sampled_species=well_sampled,
        habitat_richness=habitat_richness,
        location_richness=location_richness,
        habitat_group_richness=habitat_group_richness,
        habitat_accumulation_curves=habitat_accumulation_curves,
        group_label=group_label,
        coverage_lines=cov_lines,
        **extra,
        **point,
    )