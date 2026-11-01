"""
Tür Zenginliği Belirleme panosu için otomatik bulgu cümleleri.

Yalnızca özetteki (DatasetSummary) sayılardan türetilir; veri setinden bağımsızdır.
Yalnızca belirlenen türler, seyreltilmiş
karşılaştırmalar ve betimleyici ölçüler raporlanır.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import richness_core as rc

LOW_EFFORT_SEGMENTS = 5   # bu değerin altındaki segmentli noktalar karşılaştırma için zayıf kabul edilir


@dataclass
class Insight:
    kind: str    # "ok" | "info" | "warn"
    title: str
    text: str


def rarefy_types(inc, point_types) -> list:
    """Habitat tiplerini aynı nokta sayısına seyreltir. [(tip, m, s_obs, ort, ss, k)] (ort'a göre azalan).
    En az 2 noktalı en az iki tip yoksa boş liste."""
    if inc is None or getattr(inc, "shape", (0, 0))[0] == 0 or point_types is None:
        return []
    pt = point_types.reindex(inc.index)
    groups = {}
    for g in pt.dropna().unique():
        sub = inc.loc[pt[pt == g].index]
        sub = sub.loc[:, sub.any(axis=0)]
        if len(sub) >= 2:
            groups[str(g)] = sub
    if len(groups) < 2:
        return []
    k = min(len(m) for m in groups.values())
    out = []
    for g, m in groups.items():
        mean, sd = rc.rarefaction(m, k)
        out.append((g, len(m), int(m.shape[1]), mean, 0.0 if len(m) == k else sd, k))
    out.sort(key=lambda r: -r[3])
    return out


def compute_insights(s) -> List[Insight]:
    out: List[Insight] = []
    if not s.n_points:
        return [Insight("warn", "Veri yok", "Habitat_ID ile etiketlenmiş nokta bulunamadı.")]

    out.append(Insight(
        "ok", "Belirlenen tür zenginliği",
        f"{s.n_points} etiketli noktada toplam {s.n_species_observed} farklı tür belirlendi."))

    # En zengin nokta
    eff = s.point_effort
    if eff is not None and len(eff):
        top = eff["tur"].idxmax()
        seg = int(eff.loc[top, "segment"])
        typ = ""
        if s.point_types is not None and top in s.point_types.index:
            typ = f" ({s.point_types[top]})"
        out.append(Insight(
            "info", "En çok tür belirlenen nokta",
            f"{top}{typ}: {int(eff.loc[top, 'tur'])} tür, {seg} segment. "
            "Noktalar ham tür sayısıyla sıralanmaz."))

        out.append(Insight(
            "warn", "Nokta başına segment sayısı",
            f"Nokta başına etiketli segment {int(eff['segment'].min())}–{int(eff['segment'].max())} arasında değişiyor "
            f"(medyan {eff['segment'].median():.0f}). Noktalar ham tür sayısıyla "
            "sıralanmaz."))
        low = int((eff["segment"] < LOW_EFFORT_SEGMENTS).sum())
        if low:
            out.append(Insight(
                "warn", "Az segmentli noktalar",
                f"{low} noktada {LOW_EFFORT_SEGMENTS}'ten az etiketli segment var; bu noktaların tür sayıları "
                "tek başına yorumlanmamalıdır."))

    # Tek noktada görülen türler
    inc = s.point_incidence
    if inc is not None and getattr(inc, "size", 0) > 0:
        freq = np.asarray(inc).astype(bool).sum(axis=0)
        freq = freq[freq > 0]
        if len(freq):
            tek = int((freq == 1).sum())
            out.append(Insight(
                "info", "Tek noktada belirlenen türler",
                f"Türlerin {tek} tanesi ({100 * tek / len(freq):.0f}%) yalnızca bir noktada belirlendi; "
                f"en yaygın tür {int(freq.max())} noktada görüldü."))

    # Birikim eğrisi
    pc = s.pooled_curve
    if pc is not None and len(pc) >= 2:
        y = pc["beklenen_tur"].values
        tail = min(5, len(y) - 1)
        slope = (y[-1] - y[-1 - tail]) / tail
        out.append(Insight(
            "info", "Tür birikim eğrisi",
            f"Son {tail} noktada nokta başına ortalama +{slope:.2f} yeni tür belirlendi "
            "(seyreltme eğrisinin son kısmının eğimi)."))

    # Habitat tipleri (aynı nokta sayısına seyreltilmiş)
    rt = rarefy_types(inc, s.point_types)
    if rt:
        k = rt[0][5]
        hi, lo = rt[0], rt[-1]
        overlap = len(rt) > 1 and (rt[0][3] - rt[0][4]) <= (rt[1][3] + rt[1][4])
        note = " En üstteki tiplerin ±1 s.s. aralıkları örtüşüyor; aralarındaki fark belirgin değil." if overlap else ""
        out.append(Insight(
            "info", "Habitat tipleri",
            f"{k} noktaya seyreltildiğinde en yüksek beklenen tür sayısı {hi[0]} ({hi[3]:.1f}), "
            f"en düşük {lo[0]} ({lo[3]:.1f}).{note}"))

    # Bölgeler
    loc = s.location_rarefaction
    if loc is not None and len(loc) > 1:
        k = int(loc["k"].min())
        parts = ", ".join(f"{r.grup}: {r.seyreltilmis_tur:.1f}" for r in loc.itertuples())
        small = loc.loc[loc["m"].idxmin()]
        out.append(Insight(
            "info", "Bölgeler",
            f"{k} noktaya seyreltilmiş beklenen tür sayısı — {parts}. "
            f"{small['grup']} yalnızca {int(small['m'])} noktaya sahip olduğundan karşılaştırma sınırlıdır."))
    return out
