"""
Tür zenginliği BELİRLEME: örnekleme NOKTASI temelli, gözlenen (belirlenen) tür sayıları.

Bu modül yalnızca belirlenen (gözlenen) türleri sayar ve gruplar arası karşılaştırma için
aynı sayıda noktaya seyreltilmiş (rarefaction) değerler üretir.

Temel ilkeler
-------------
* Örnekleme birimi = bir örnekleme noktası (ör. ``Habitat_ID`` / ``Code``).
  Girdi, nokta x tür varlık/yokluk (insidans) matrisidir. Aynı noktadaki
  birden çok kayıt/segment tek birimin içinde birleştirilir (bağımsız
  örnekler değildir).
* Seyreltme (rarefaction): örnek-bazlı analitik beklenen değer (Coleman 1982;
  Colwell vd. 2004) ve koşullu varyans.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import gammaln


# --------------------------------------------------------------------------- #
# Girdi: insidans matrisi
# --------------------------------------------------------------------------- #
def build_incidence_matrix(
    df: pd.DataFrame,
    unit_column: str = "Habitat_ID",
    species_column: str = "Turkish_Name",
) -> pd.DataFrame:
    """
    Veri çerçevesinden nokta x tür varlık/yokluk matrisi (bool) üretir.
    Boş / 'nan' birim ya da tür değerleri atılır. Satırlar birimler, sütunlar
    türlerdir.
    """
    if unit_column not in df.columns or species_column not in df.columns:
        raise KeyError(f"'{unit_column}' ve '{species_column}' sütunları gerekli.")
    sub = df[[unit_column, species_column]].copy()
    for col in (unit_column, species_column):
        sub[col] = sub[col].astype(str).str.strip()
        sub = sub[(sub[col] != "") & (sub[col].str.lower() != "nan")]
    if sub.empty:
        return pd.DataFrame(dtype=bool)
    return pd.crosstab(sub[unit_column], sub[species_column]).astype(bool)


def _as_array(incidence) -> np.ndarray:
    arr = np.asarray(incidence)
    if arr.ndim != 2:
        raise ValueError("İnsidans matrisi 2 boyutlu (birim x tür) olmalıdır.")
    arr = arr.astype(bool)
    # Hiçbir birimde görülmeyen sütunlar zenginliğe katkı vermez
    return arr[:, arr.any(axis=0)]


@dataclass
class IncidenceStats:
    m: int                 # örnekleme birimi sayısı
    s_obs: int             # belirlenen (gözlenen) tür sayısı
    freq: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=int))  # tür başına nokta sayısı


def incidence_stats(incidence) -> IncidenceStats:
    arr = _as_array(incidence)
    return IncidenceStats(m=arr.shape[0], s_obs=int(arr.shape[1]),
                          freq=arr.sum(axis=0).astype(int))


# --------------------------------------------------------------------------- #
# Seyreltme (rarefaction)
# --------------------------------------------------------------------------- #
def _log_comb(n: np.ndarray, k: int) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    out = np.full(n.shape, -np.inf)
    ok = n >= k
    out[ok] = gammaln(n[ok] + 1) - gammaln(k + 1) - gammaln(n[ok] - k + 1)
    return out


def rarefaction(incidence, k: int) -> tuple:
    """
    k birimlik rastgele alt örneklemde beklenen tür sayısı ve koşullu standart
    sapması (veriye koşullu; Colwell vd. 2004). 1 <= k <= m olmalı.
    """
    arr = _as_array(incidence)
    m = arr.shape[0]
    if not (1 <= k <= m):
        raise ValueError(f"k, 1 ile m={m} arasında olmalı (k={k}).")
    if arr.shape[1] == 0:
        return 0.0, 0.0
    n = arr.sum(axis=0).astype(float)
    log_total = float(_log_comb(np.array([m]), k)[0])
    p_absent = np.exp(_log_comb(m - n, k) - log_total)   # P(tür i hiç seçilmez)
    mean = float((1 - p_absent).sum())

    # Varyans: Var(Σ A_i) = Σ pa_i (1-pa_i) + 2 Σ_{i<j} (P(A_i ∧ A_j) − pa_i pa_j)
    a = arr.astype(float)
    co = a.T @ a                                          # ortak birim sayısı
    union = n[:, None] + n[None, :] - co
    p_both = np.exp(_log_comb(m - union, k) - log_total)
    cov = p_both - np.outer(p_absent, p_absent)
    np.fill_diagonal(cov, 0.0)
    var = float((p_absent * (1 - p_absent)).sum() + cov.sum())
    return mean, math.sqrt(max(var, 0.0))




def rarefaction_curve(incidence, k_max: Optional[int] = None) -> pd.DataFrame:
    """k = 1..k_max için beklenen tür sayısı ve ±1 s.s. (k_max varsayılan m)."""
    arr = _as_array(incidence)
    m = arr.shape[0]
    k_max = m if k_max is None else min(k_max, m)
    rows = []
    for k in range(1, k_max + 1):
        mean, sd = rarefaction(arr, k)
        rows.append({"k": k, "beklenen_tur": mean, "ss": sd})
    return pd.DataFrame(rows)




def rarefy_groups(
    df: pd.DataFrame,
    group_column: str,
    unit_column: str = "Habitat_ID",
    species_column: str = "Turkish_Name",
    k: Optional[int] = None,
) -> pd.DataFrame:
    """
    Gruplar (ör. Location) arasında AYNI sayıda birimle karşılaştırma: her grup k
    birime seyreltilir (varsayılan k = en az birimli grubun birim sayısı).
    Ham tür sayıları örnek sayısından etkilendiği için gruplar arası
    karşılaştırma bu değerlerle yapılmalıdır.
    Sütunlar: grup, m, s_obs, k, seyreltilmis_tur, ss
    """
    for col in (group_column, unit_column, species_column):
        if col not in df.columns:
            raise KeyError(f"'{col}' sütunu gerekli.")
    mats = {}
    for g, sub in df.groupby(group_column):
        inc = build_incidence_matrix(sub, unit_column, species_column)
        if not inc.empty:
            mats[str(g)] = inc
    if not mats:
        return pd.DataFrame(columns=["grup", "m", "s_obs", "k", "seyreltilmis_tur", "ss"])
    k_common = min(m.shape[0] for m in mats.values()) if k is None else int(k)
    rows = []
    for g, inc in sorted(mats.items()):
        m = inc.shape[0]
        kk = min(k_common, m)
        mean, sd = rarefaction(inc, kk)
        rows.append({"grup": g, "m": m, "s_obs": int(inc.shape[1]),
                     "k": kk, "seyreltilmis_tur": mean, "ss": sd})
    return pd.DataFrame(rows)


def species_at_frequency(incidence, k: int) -> list:
    """Tam ``k`` noktada belirlenen türler: [(tür, [nokta, ...]), ...] (tür adına göre sıralı).

    ``incidence`` nokta x tür bool DataFrame'idir (index = nokta, columns = tür). Tam ``k`` noktada belirlenen tür yoksa
    boş liste döner (tür-sıklığı grafiğinde o değerin çubuğu boştur)."""
    if incidence is None or getattr(incidence, "empty", True):
        return []
    inc = incidence.astype(bool)
    counts = inc.sum(axis=0)
    out = []
    for sp in sorted(counts[counts == int(k)].index, key=str):
        pts = [i for i in inc.index[inc[sp].to_numpy()]]
        try:
            pts = sorted(pts, key=lambda v: int(str(v)))
        except ValueError:
            pts = sorted(pts, key=str)
        out.append((str(sp), [str(p) for p in pts]))
    return out
