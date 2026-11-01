"""
Tür zenginliği grafikleri (saf matplotlib; Qt'ye bağımlı değil).

Hem uygulama penceresi (SpeciesRichnessDialog) hem de tez şekli betiği
(tez_sekilleri.py) aynı fonksiyonları kullanır; böylece ekranda görülen ile
tezde basılan grafik aynıdır. Her fonksiyon bir matplotlib Figure döndürür.
"""
import re

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .plot_style import RENK, style_context


MAX_BARS = 40   # bundan fazla kategori varsa en büyük MAX_BARS tanesi gösterilir (okunabilirlik)


def bar_chart(counts, title, xlabel, colors=None, bar_color=None, dpi=100, max_bars=MAX_BARS):
    """Yatay çubuk grafiği; counts bir pandas Series (en büyük değer üstte).

    Kategori sayısı çok olduğunda (örn. 300 nokta) en büyük ``max_bars`` kategori
    gösterilir (``max_bars=None`` ise hepsi) ve başlığa toplam sayı yazılır; böylece şekil okunur boyutta kalır.
    """
    with style_context():
        total = len(counts)
        if max_bars is not None and total > max_bars:
            counts = counts.iloc[:max_bars]
            title = f"{title}\n(en yüksek {max_bars} / {total})"
        n = max(len(counts), 1)
        fig = Figure(figsize=(9, max(2.8, 0.38 * n + 1.3)), dpi=dpi)
        ax = fig.add_subplot(111)
        if len(counts) == 0:
            ax.text(0.5, 0.5, "(veri yok)", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        else:
            ordered = counts.iloc[::-1]
            labels = [str(x) for x in ordered.index]
            values = ordered.values
            col = colors(values) if callable(colors) else (bar_color or RENK["mavi"])
            ax.barh(labels, values, color=col)
            ax.set_xlabel(xlabel)
            ax.grid(axis="y", visible=False)
            ax.set_ylim(-0.7, len(values) - 0.3)   # üstte/altta gereksiz boşluk kalmasın
            xmax = float(np.max(values)) if len(values) else 1.0
            ax.set_xlim(0, xmax * 1.12 if xmax > 0 else 1)
            for i, v in enumerate(values):
                txt = f" {v:.2f}" if isinstance(v, (float, np.floating)) else f" {v}"
                ax.text(v, i, txt, va="center", fontsize=9)
        ax.set_title(title)
        fig.tight_layout()
    return fig


def point_richness_chart(counts, title="Nokta Başına Belirlenen Tür Sayısı",
                         xlabel="Farklı tür sayısı", max_bars=MAX_BARS, dpi=100, all_types=None):
    """Nokta başına tür sayısı: az nokta varsa çubuk (renk = habitat tipi), çok nokta varsa dağılım (histogram)."""
    if len(counts) <= max_bars:
        def _tip(label):
            m = re.search(r"\(([^()]*)\)\s*$", str(label))
            return m.group(1) if m else "?"
        types = [_tip(i) for i in counts.index]
        cmap, named, has_other = type_colors(all_types if all_types is not None else types)
        named = [t for t in named if t in set(types)]          # legend: yalnızca çizilen tipler; renkler tüm noktalardan
        fig = bar_chart(counts, title, xlabel,
                        colors=lambda vals: [cmap[t] for t in types[::-1]][:len(vals)], dpi=dpi, max_bars=max_bars)
        fig.axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        if named:
            n_other = len(set(types)) - len(named)
            fig.axes[0].legend(handles=_legend_handles(cmap, named, has_other, n_other), loc="lower right", fontsize=8.5)
        return fig
    with style_context():
        fig = Figure(figsize=(9, 4.8), dpi=dpi)
        ax = fig.add_subplot(111)
        v = np.asarray(counts.values, dtype=float)
        bins = np.arange(v.min() - 0.5, v.max() + 1.5, max(1, int(np.ceil((v.max() - v.min() + 1) / 25))))
        ax.hist(v, bins=bins, color=RENK["mor"], edgecolor="white")
        ax.axvline(np.median(v), color=RENK["kirmizi"], linestyle="--", label=f"medyan = {np.median(v):.0f}")
        ax.axvline(v.mean(), color=RENK["mavi"], linestyle=":", label=f"ortalama = {v.mean():.1f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Nokta sayısı")
        ax.set_title(f"{title} — dağılım ({len(v)} nokta)")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        fig.tight_layout()
    return fig


def location_rarefaction(loc, dpi=100):
    """Bölgeler için aynı sayıda noktaya seyreltilmiş beklenen tür sayısı (nokta ± 1 s.s.).
    Tüm noktaları kullanılan (k = m) bölgede seyreltme belirsizliği yoktur; hata çubuğu çizilmez."""
    with style_context():
        n = len(loc)
        fig = Figure(figsize=(9.5, max(2.8, 0.95 * n + 1.5)), dpi=dpi)
        ax = fig.add_subplot(111)
        labels = [f"{r.grup}  (belirlenen {int(r.s_obs)} tür, {int(r.m)} nokta)"
                  for r in loc.itertuples()][::-1]
        vals = loc["seyreltilmis_tur"].values[::-1]
        sds = loc["ss"].values[::-1]
        ms = loc["m"].values[::-1]
        ks = loc["k"].values[::-1]
        y = np.arange(n)
        for i in range(n):
            full = int(ks[i]) >= int(ms[i])
            sd = 0.0 if (sds[i] != sds[i] or full) else float(sds[i])
            if sd > 0:
                ax.errorbar(vals[i], y[i], xerr=sd, fmt="none", ecolor="#555555", elinewidth=1.6, capsize=4, zorder=2)
            ax.plot(vals[i], y[i], "o", markersize=11, color="#4F6D8F", markeredgecolor="white",
                    markeredgewidth=1.5, zorder=3)
            tail = "  (tüm noktalar, s.s. yok)" if full else ""
            ax.annotate(f"{vals[i]:.1f}{tail}", (vals[i] + sd, y[i]), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=9)
        top = max(v + (0 if (sd != sd) else sd) for v, sd in zip(vals, sds))
        ax.set_xlim(0, top * 1.9)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.grid(axis="y", visible=False)
        k = int(loc["k"].min())
        ax.set_xlabel(f"Beklenen tür sayısı ({k} noktaya seyreltilmiş, ±1 s.s.)")
        ax.set_title("Bölge Başına Tür Zenginliği (seyreltilmiş)")
        fig.tight_layout()
    return fig


def accumulation(curve, dpi=100, golge=False, cubuk=False):
    """Tür birikim (seyreltme) eğrisi: etiketlenen nokta sayısı arttıkça belirlenen tür sayısı.

    Varsayılan (sade) görünüm: gölgesiz tek çizgi, üzerinde yorum metni yok. ``cubuk=True`` altına "her yeni nokta kaç yeni tür getirdi" çubuklarını ekler,
    ``golge=True`` eski görünümü verir (tek panel, ±1 s.s. gölgeli: nokta sırasına bağlı değişkenlik)."""
    x = curve["k"].values
    y = curve["beklenen_tur"].values
    if golge:
        return _accumulation_shaded(curve, dpi)
    tail = min(5, len(y) - 1)
    gain = np.diff(y)
    with style_context():
        if cubuk:
            fig = Figure(figsize=(9, 6.6), dpi=dpi)
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.6], hspace=0.12)
            ax = fig.add_subplot(gs[0])
        else:
            fig = Figure(figsize=(9, 5.0), dpi=dpi)
            ax = fig.add_subplot(111)
        ax.plot(x, y, color=RENK["mavi"], lw=2.2)
        ax.scatter([x[-1]], [y[-1]], color=RENK["mavi"], zorder=3, s=36)
        ax.set_ylabel("Belirlenen tür sayısı")
        ax.set_title("Etiketlenen nokta sayısı arttıkça belirlenen tür sayısı")
        ax.set_xlim(0.5, x.max() + 0.5)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if cubuk:
            ax.tick_params(labelbottom=False)
            ax2 = fig.add_subplot(gs[1], sharex=ax)
            if len(y) >= 2:
                ax2.bar(x[1:], gain, color=RENK["acik_mavi"], width=0.8)
                ax2.axhline(float(np.mean(gain[-tail:])), color=RENK["kirmizi"], ls="--", lw=1)
            ax2.set_ylabel("Yeni nokta başına\nyeni tür")
            ax2.set_xlabel("Etiketlenmiş nokta sayısı")
            ax2.grid(axis="x", visible=False)
            fig.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.11)
        else:
            ax.set_xlabel("Etiketlenmiş nokta sayısı")
            fig.tight_layout()
    fig.acc_points = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    fig.acc_label = None
    return fig


def accumulation_label_at(fig, px, py, slack_px=12.0):
    """Birikim grafiğinde ekran pikseli (px, py) çizginin bir noktasına (özellikle uçtaki yuvarlağa) yakınsa o noktanın
    'N nokta → T tür' etiketini yazar ve True döner. Çizgiden uzağa tıklanırsa etiketi kaldırır ve False döner.
    Çağıran, değişiklik sonrası tuvali yeniden çizmelidir."""
    pts = getattr(fig, "acc_points", None)
    if pts is None or not fig.axes:
        return False
    ax = fig.axes[0]
    x, y = pts
    scr = ax.transData.transform(np.column_stack([x, y]))
    d = np.hypot(scr[:, 0] - px, scr[:, 1] - py)
    old = getattr(fig, "acc_label", None)
    if old is not None:
        old.remove()
        fig.acc_label = None
    i = int(np.argmin(d)) if len(d) else -1
    if i < 0 or d[i] > slack_px:
        return False
    at_end = i == len(x) - 1
    fig.acc_label = ax.annotate(
        f"{int(x[i])} nokta → {y[i]:.0f} tür", (x[i], y[i]), xytext=(-6 if at_end else 12, -48 if at_end else 14),
        textcoords="offset points", ha="right" if at_end else "left", va="top" if at_end else "bottom", fontsize=11,
        fontweight="bold", color=RENK["mavi"], zorder=6,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": RENK["mavi"], "lw": 1})
    return True


def _accumulation_shaded(curve, dpi=100):
    with style_context():
        fig = Figure(figsize=(9, 5.6), dpi=dpi)
        ax = fig.add_subplot(111)
        x = curve["k"].values
        y = curve["beklenen_tur"].values
        sd = curve["ss"].values
        slope_txt = ""
        if len(y) >= 2:
            tail = min(5, len(y) - 1)
            slope = (y[-1] - y[-1 - tail]) / tail
            slope_txt = f" (son {tail} noktada nokta başına ≈ +{slope:.2f} tür)"
        ax.plot(x, y, color=RENK["mavi"], label="Beklenen tür sayısı (seyreltme)" + slope_txt)
        ax.fill_between(x, y - sd, y + sd, color=RENK["mavi"], alpha=0.15, linewidth=0,
                        label="±1 s.s. (nokta sırasına bağlı değişkenlik)")
        ax.set_xlabel("Etiketlenmiş nokta sayısı")
        ax.set_xlim(0.5, x.max() + 0.5)
        ax.set_ylabel("Belirlenen tür sayısı (beklenen)")
        ax.set_title("Tür Birikim Eğrisi (belirlenen türler, tüm noktalar)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="lower right")
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Görsel panel grafikleri (Tür Zenginliği Belirleme penceresi)
# --------------------------------------------------------------------------- #
_KAT_SIRA = ["mavi", "turuncu", "yesil", "kirmizi", "mor", "acik_mavi"]   # gri yalnızca "Diğer" içindir   # sabit sıra, döngü yok
# 10 habitat tipinin hepsi adlı renk alsın diye ilk 6 renk korunur, 4 renk eklenir; renk tek başına ayırt etmeye
# yetmeyebileceğinden varlık haritasında şerit üzerine tip numarası da yazılır.
_EK_RENKLER = ["#5E3C99", "#8C510A", "#B5BD00", "#2F2F2F"]
MAX_ADLI_TIP = len(_KAT_SIRA) + len(_EK_RENKLER)


def _kat_renk(i):
    return RENK[_KAT_SIRA[i]] if i < len(_KAT_SIRA) else _EK_RENKLER[i - len(_KAT_SIRA)]
_DIGER_RENK = "#B8B8B8"


def type_colors(types, max_named=MAX_ADLI_TIP):
    """Habitat tipi -> renk. En çok noktası olan max_named (varsayılan 10) tip adlı renk alır; ondan fazla tip varsa kalanı 'Diğer' gridir."""
    counts = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    order = sorted(counts, key=lambda k: (-counts[k], str(k)))
    cmap = {t: _kat_renk(i) for i, t in enumerate(order[:max_named])}
    for t in order[max_named:]:
        cmap[t] = _DIGER_RENK
    return cmap, order[:max_named], len(order) > max_named


def _legend_handles(cmap, named, has_other, n_other=0, numbered=False):
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor=cmap[t],
                markeredgecolor="white", label=(f"{i} · {t}" if numbered else str(t)))
         for i, t in enumerate(named, start=1)]
    if has_other:
        h.append(Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor=_DIGER_RENK,
                        markeredgecolor="white",
                        label=f"Diğer ({n_other} tip)" if n_other else "Diğer"))
    return h


def habitat_type_richness(type_richness, inc=None, point_types=None, dpi=100):
    """Habitat tipi başına tür zenginliği. inc + point_types verilirse tipler AYNI NOKTA SAYISINA
    seyreltilir (nokta ± 1 s.s.); aksi halde ham sayı çubuğu (nokta sayıları farklı: karşılaştırılamaz)."""
    from ..data import richness_core as rc
    with style_context():
        rows = []
        if inc is not None and point_types is not None and getattr(inc, "shape", (0, 0))[0] > 0:
            pt = point_types.reindex(inc.index)
            groups = {str(g): inc.loc[pt[pt == g].index] for g in pt.dropna().unique()}
            groups = {g: m.loc[:, m.any(axis=0)] for g, m in groups.items() if len(m) >= 2}
            if len(groups) >= 2:
                k = min(len(m) for m in groups.values())
                for g, m in groups.items():
                    mean, sd = rc.rarefaction(m, k)
                    rows.append((g, len(m), int(m.shape[1]), mean, 0.0 if len(m) == k else sd))
                rows.sort(key=lambda r: r[3])
                fig = Figure(figsize=(7.4, max(3.2, 0.5 * len(rows) + 1.8)), dpi=dpi)
                ax = fig.add_subplot(111)
                for i, (g, m, so, mean, sd) in enumerate(rows):
                    if sd > 0:
                        ax.errorbar(mean, i, xerr=sd, fmt="none", ecolor="#555555", elinewidth=1.4, capsize=3, zorder=2)
                    ax.plot(mean, i, "o", markersize=10, color="#4F6D8F", markeredgecolor="white",
                            markeredgewidth=1.4, zorder=3)
                    ax.annotate(f"{mean:.1f}   (belirlenen {so} tür, {m} nokta)", (mean + sd, i),
                                xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
                ax.set_yticks(range(len(rows)))
                ax.set_yticklabels([r[0] for r in rows])
                ax.set_ylim(-0.6, len(rows) - 0.4)
                top = max(r[3] + r[4] for r in rows)
                ax.set_xlim(0, top * 1.75)
                ax.grid(axis="y", visible=False)
                ax.set_xlabel(f"Beklenen tür sayısı ({k} noktaya seyreltilmiş, ±1 s.s.)")
                excl = len(pt.dropna().unique()) - len(rows)
                title = "Habitat Tipine Göre Tür Zenginliği (seyreltilmiş)"
                ax.set_title(title)
                if excl > 0:
                    ax.text(0.99, 0.02, f"{excl} tip yalnızca 1 noktalı olduğu için dışarıda", transform=ax.transAxes,
                            ha="right", va="bottom", fontsize=8.5, color="#555555")
                fig.tight_layout()
                return fig
        n = max(len(type_richness), 1)
        fig = Figure(figsize=(7.4, max(3.2, 0.5 * n + 1.8)), dpi=dpi)
        ax = fig.add_subplot(111)
        rows = sorted(type_richness, key=lambda r: r.s_obs)
        if not rows:
            ax.text(0.5, 0.5, "(veri yok)", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        else:
            ax.barh([r.group for r in rows], [r.s_obs for r in rows], color="#4F6D8F", height=0.62)
            xmax = max(r.s_obs for r in rows)
            for i, r in enumerate(rows):
                ax.text(r.s_obs + 0.01 * xmax, i, f" {r.s_obs} tür · {r.m} nokta", va="center", fontsize=9)
            ax.set_xlim(0, xmax * 1.35)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("Belirlenen tür sayısı (nokta sayıları farklı: ham sayı karşılaştırılamaz)")
        ax.set_title("Habitat Tipine Göre Belirlenen Tür")
        fig.tight_layout()
    return fig


def species_incidence_histogram(inc, dpi=100):
    """Her tür kaç noktada görüldü: nokta sayısı - tür sayısı dağılımı (betimleyici)."""
    with style_context():
        fig = Figure(figsize=(7.4, 4.6), dpi=dpi)
        ax = fig.add_subplot(111)
        freq = np.asarray(inc).astype(bool).sum(axis=0)
        freq = freq[freq > 0]
        if len(freq) == 0:
            ax.text(0.5, 0.5, "(veri yok)", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        else:
            vals = np.arange(1, int(freq.max()) + 1)          # x ekseni 1'den en yaygın türün nokta sayısına kadar sürer; çubuğu olmayan değerde tür yoktur
            cnt = np.bincount(freq, minlength=int(freq.max()) + 1)[1:]
            cols = [RENK["turuncu"] if v == 1 else RENK["mavi"] for v in vals]
            ax.bar(vals, cnt, color=cols, width=0.75)
            for v, c in zip(vals, cnt):
                if c > 0:                                          # boş değerde (tür yok) çubuk ve sayı gösterilmez
                    ax.text(v, c, f"{c}", ha="center", va="bottom", fontsize=9)
            ax.set_xlim(0.4, vals.max() + 0.6)
            tek = int(cnt[0])
            ax.text(0.97, 0.95,
                    f"{tek} tür (%{100 * tek / len(freq):.0f}) yalnızca 1 noktada belirlendi\n"
                    f"En yaygın tür {int(freq.max())} noktada belirlendi",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color="#333333", linespacing=1.5)
            if len(vals) <= 20:
                ax.set_xticks(vals)                                # her değer etiketli: 1, 2, 3, ...
            else:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Türün belirlendiği nokta sayısı")
            ax.set_ylabel("Tür sayısı")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="x", visible=False)
        ax.set_title("Her tür kaç noktada belirlendi?")
        fig.tight_layout()
    return fig


def presence_heatmap(inc, point_types=None, max_points=120, max_species=None, dpi=100, all_types=None):
    """Tür x nokta varlık haritası: TÜM türler (tek noktada görülenler dahil) satırda, noktalar sütunda.
    Satırlar yaygınlığa (azalan), sütunlar nokta başına tür sayısına (azalan) göre sıralıdır;
    sütunların üstündeki renkli şerit habitat tipini gösterir."""
    from matplotlib.gridspec import GridSpec
    with style_context():
        arr = np.asarray(inc).astype(bool)
        if arr.size == 0:
            fig = Figure(figsize=(9, 3), dpi=dpi)
            fig.add_subplot(111).text(0.5, 0.5, "(veri yok)", ha="center", va="center")
            return fig
        points = list(inc.index) if hasattr(inc, "index") else list(range(arr.shape[0]))
        species = list(inc.columns) if hasattr(inc, "columns") else list(range(arr.shape[1]))
        keep = np.where(arr.any(axis=0))[0]                      # hiç belirlenmemiş sütunlar dışarıda
        freq = arr.sum(axis=0)
        s_ord = keep[np.argsort(-freq[keep], kind="stable")]
        if max_species:
            s_ord = s_ord[:max_species]
        p_ord = np.argsort(-arr.sum(axis=1), kind="stable")[:max_points]
        sub = arr[np.ix_(p_ord, s_ord)].T                         # satır = tür, sütun = nokta
        n_s, n_p = sub.shape
        xlab = [str(points[i]) for i in p_ord]
        ylab = [str(species[j]) for j in s_ord]
        fs = 8 if n_s <= 90 else 6.5
        W = max(7.0, 0.3 * n_p + 2.4)
        H = 0.2 * n_s + 3.0
        fig = Figure(figsize=(W, H), dpi=dpi)
        gs = GridSpec(2, 1, height_ratios=[0.45, 0.2 * n_s], hspace=0.03, figure=fig,
                      left=2.0 / W, right=0.99, top=1 - 0.55 / H, bottom=1.6 / H)
        ax_t = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1])
        if point_types is not None:
            types = [str(point_types.get(points[i], "?")) for i in p_ord]
            if all_types is None and hasattr(point_types, "values"):
                all_types = [str(v) for v in point_types.values]           # renk/numara tüm noktalardan: grafikler arası aynı
            cmap, named, has_other = type_colors(all_types if all_types is not None else types)
            num = {t: i for i, t in enumerate(named, start=1)}
            for x, t in enumerate(types):
                ax_t.add_patch(plt_rect(x, cmap[t]))
                if t in num:                  # renk tek başına yetmesin: şeritte tip numarası da yazılır (legend ile aynı)
                    r_, g_, b_ = mcolors.to_rgb(cmap[t])
                    ax_t.text(x, 0.5, str(num[t]), ha="center", va="center", fontsize=7,
                              color="white" if (0.299 * r_ + 0.587 * g_ + 0.114 * b_) < 0.6 else "black")
            fig.legend(handles=_legend_handles(cmap, named, has_other, len(set(types)) - len(named), numbered=True),
                       loc="upper left", bbox_to_anchor=(2.0 / W, 1.0), ncol=5, fontsize=8, frameon=False)
        ax_t.set_xlim(-0.5, n_p - 0.5)
        ax_t.set_ylim(0, 1)
        ax_t.axis("off")
        cm = LinearSegmentedColormap.from_list("var_yok", ["#F2F4F7", RENK["mavi"]])
        ax.imshow(sub, aspect="auto", cmap=cm, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(n_p))
        ax.set_xticklabels(xlab, rotation=90, fontsize=8)
        ax.set_yticks(range(n_s))
        ax.set_yticklabels(ylab, fontsize=fs)
        ax.set_xticks(np.arange(-0.5, n_p), minor=True)
        ax.set_yticks(np.arange(-0.5, n_s), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        tek = int((freq[keep] == 1).sum())
        note = f"{n_s} tür · {tek} tanesi yalnızca 1 noktada · mavi = belirlendi"
        if arr.shape[0] > len(p_ord):
            note += f" · en zengin {len(p_ord)}/{arr.shape[0]} nokta"
        ax.set_xlabel(note, fontsize=9)
    return fig


def plt_rect(x, color):
    from matplotlib.patches import Rectangle
    return Rectangle((x - 0.5, 0.1), 1.0, 0.8, facecolor=color, edgecolor="white", linewidth=1.0)
