"""src/visualization/habitat_plots.py — Habitat Bilgileri penceresi grafikleri (matplotlib, Figure tabanlı)."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .plot_style import RENK, style_context

LOW_EFFORT = 5   # data.habitat_info.LOW_EFFORT_SEGMENTS ile aynı


def _empty(title):
    fig = Figure(figsize=(7, 2.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, "(veri yok)", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.axis("off")
    return fig


def coverage_by_type(tipler, dpi=100):
    """Habitat tipi başına tanımlı ve etiketli nokta sayısı (yan yana yatay çubuklar)."""
    title = "Habitat tipi başına nokta: tanımlı ve etiketli"
    if tipler is None or len(tipler) == 0:
        return _empty(title)
    with style_context():
        d = tipler.sort_values("tanimli", ascending=True)
        n = len(d)
        fig = Figure(figsize=(8.4, max(3.2, 0.52 * n + 1.5)), dpi=dpi)
        ax = fig.add_subplot(111)
        y = np.arange(n)
        h = 0.38
        ax.barh(y + h / 2, d["tanimli"], h, color="#c9d1dc", label="Tanımlı nokta")
        ax.barh(y - h / 2, d["etiketli"], h, color=RENK["mavi"], label="Etiketli nokta")
        for yi, (t, e) in enumerate(zip(d["tanimli"], d["etiketli"])):
            ax.text(t + 0.4, yi + h / 2, str(int(t)), va="center", fontsize=9, color="#4b5563")
            ax.text(e + 0.4, yi - h / 2, str(int(e)), va="center", fontsize=9, color=RENK["mavi"], fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(d["habitat_tipi"])
        ax.set_xlabel("Nokta sayısı")
        ax.set_title(title)
        ax.set_xlim(0, max(d["tanimli"].max() * 1.12, 1))
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
        fig.tight_layout()
        return fig


def segments_per_point(noktalar, dpi=100):
    """Etiketli noktaların segment sayısı, habitat tipine göre (her nokta bir daire)."""
    title = "Etiketli noktalarda segment sayısı (habitat tipine göre)"
    if noktalar is None or len(noktalar) == 0 or not (noktalar["segment"] > 0).any():
        return _empty(title)
    with style_context():
        lab = noktalar[noktalar["segment"] > 0]
        order = (lab.groupby("habitat_tipi")["segment"].sum().sort_values(ascending=True)).index.tolist()
        fig = Figure(figsize=(8.4, max(3.2, 0.52 * len(order) + 1.6)), dpi=dpi)
        ax = fig.add_subplot(111)
        rng = np.random.default_rng(0)
        for yi, t in enumerate(order):
            v = lab.loc[lab["habitat_tipi"] == t, "segment"].to_numpy()
            jit = rng.uniform(-0.18, 0.18, len(v))
            col = [RENK["kirmizi"] if x < LOW_EFFORT else RENK["mavi"] for x in v]
            ax.scatter(v, yi + jit, s=46, c=col, edgecolors="white", linewidths=0.8, zorder=3)
        ax.axvline(LOW_EFFORT, color=RENK["kirmizi"], ls="--", lw=1)
        ax.text(LOW_EFFORT + 0.6, len(order) - 0.45, f"< {LOW_EFFORT} segment: düşük çaba", color=RENK["kirmizi"],
                fontsize=9, va="bottom")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_ylim(-0.6, len(order) - 0.2)
        ax.set_xlabel("Noktadaki segment sayısı")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        return fig


def region_type_matrix(mt, me, dpi=100):
    """Bölge × habitat tipi: hücre metni 'etiketli / tanımlı', renk etiketli oranı."""
    title = "Bölge ve habitat tipine göre nokta kapsamı"
    if mt is None or mt.empty:
        return _empty(title)
    with style_context():
        t = mt.to_numpy(dtype=float)
        e = me.to_numpy(dtype=float)
        frac = np.divide(e, t, out=np.zeros_like(t), where=t > 0)
        fig = Figure(figsize=(max(7.6, 0.95 * t.shape[1] + 2.6), max(3.0, 0.8 * t.shape[0] + 2.4)), dpi=dpi)
        ax = fig.add_subplot(111)
        im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=max(frac.max(), 0.25), aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                if t[i, j] == 0:
                    ax.text(j, i, "—", ha="center", va="center", color="#9ca3af", fontsize=10)
                else:
                    r_, g_, b_, _a = im.cmap(im.norm(frac[i, j]))
                    dark = (0.299 * r_ + 0.587 * g_ + 0.114 * b_) < 0.55     # koyu zemin -> beyaz yazı
                    ax.text(j, i, f"{int(e[i, j])} / {int(t[i, j])}", ha="center", va="center", fontsize=10,
                            color="white" if dark else "#1f2a37", fontweight="bold" if e[i, j] else "normal")
        ax.set_xticks(range(t.shape[1]))
        ax.set_xticklabels(list(mt.columns), rotation=40, ha="right")
        ax.set_yticks(range(t.shape[0]))
        ax.set_yticklabels(list(mt.index))
        ax.set_title(title)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("Etiketli nokta oranı")
        fig.tight_layout()
        return fig
