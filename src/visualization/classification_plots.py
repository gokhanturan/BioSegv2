"""Başarı sınırı analizi grafikleri (uygulama ve tez için ortak stil)."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

from .plot_style import RENK, style_context

MAX_BARS = 80          # tüm türler (77) tek grafikte; daha fazlaysa en az segmentliler


def _kisa_ad(etiket):
    """'Türkçe ad (Latince ad)' -> 'Türkçe ad' (grafik etiketi için)."""
    return str(etiket).split(" (")[0]


def data_sufficiency(species_table, min_segment=30, min_kayit=2, dpi=100):
    """Tür başına segment sayısı; en az segmentli MAX_BARS tür gösterilir. Renkler açıklamalıdır (efsane)."""
    with style_context():
        t = species_table.sort_values("segment")
        total = len(t)
        gizli = t.iloc[MAX_BARS:]
        t = t.head(MAX_BARS)
        fig = Figure(figsize=(9, max(3.2, 0.24 * len(t) + 2.0)), dpi=dpi)
        ax = fig.add_subplot(111)
        cols = [RENK["yesil"] if u else RENK["kirmizi"] for u in t["uygun"]]
        ax.barh(t["etiket"] if "etiket" in t.columns else t["tur"], t["segment"], color=cols)
        ax.tick_params(axis="y", labelsize=8)
        ax.axvline(min_segment, color=RENK["siyah"], lw=1, ls="--")
        for i, (s_, k) in enumerate(zip(t["segment"], t["kayit"])):
            ax.text(s_, i, f" {s_} seg / {k} kayıt", va="center", fontsize=8)
        ax.set_xlim(0, max(min_segment * 1.05, float(t["segment"].max()) * 1.35))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Segment sayısı")
        handles = [Patch(color=RENK["kirmizi"], label="Süzgeç dışı (sınıflandırılamaz)")]
        if bool(t["uygun"].any()):
            handles.append(Patch(color=RENK["yesil"], label="Süzgeci geçen"))
        handles += [Line2D([0], [0], color=RENK["siyah"], lw=1, ls="--",
                          label=f"Eğitim süzgeci: ≥{min_segment} segment ve ≥{min_kayit} kayıt")]
        ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True, framealpha=0.9)
        title = "Veri Yeterliliği"
        if total > len(t):
            n_gec = int(gizli["uygun"].sum())
            title += f" (en az segmentli {len(t)} / {total} tür)"
            sub = (f"Gösterilmeyen {len(gizli)} türün {n_gec} tanesi süzgeci geçiyor "
                   f"(en az {int(gizli['segment'].min())} segment)")
        else:
            sub = f"{int(species_table['uygun'].sum())} / {total} tür süzgeci geçiyor"
        ax.set_title(f"{title}\n{sub}", fontsize=11)
        fig.tight_layout()
    return fig


def methods_accuracy_f1(methods, dpi=100):
    """Yöntemlere göre doğruluk ve makro F1 (ortalama ± s.s.), yan yana; üstte doğruluk, altta makro F1."""
    from ..ml.classification_diagnosis import METHOD_NAMES
    with style_context():
        from ..ml.classification_diagnosis import ORDER
        order = [k for k in ORDER if k in methods]
        fig = Figure(figsize=(8.5, 1.1 * len(order) + 2.6), dpi=dpi)
        ax = fig.add_subplot(111)
        y = np.arange(len(order))
        h = 0.38
        for j, (met, lab, col) in enumerate((("dogruluk", "Doğruluk", RENK["mavi"]),
                                             ("makro_f1", "Makro F1", RENK["turuncu"]))):
            v = [methods[k][met] for k in order]
            e = [methods[k][met + "_ss"] for k in order]
            yy = y + (0.5 - j) * h            # doğruluk üstte: efsane sırasıyla aynı
            ax.barh(yy, v, h, xerr=e, capsize=3, color=col, label=lab, error_kw={"lw": 1.2})
            for yi, vi, ei in zip(yy, v, e):
                ax.text(vi + ei + 0.015, yi, f"{vi:.2f}", va="center", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_NAMES[k] for k in order])
        ax.set_xlim(0, 1.12)                  # etiketler çubuk/hata çizgisinin dışında kalsın
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Çapraz doğrulama ortalaması (hata çizgisi: ± 1 s.s.)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        ax.set_title("Yöntem Karşılaştırması (kayıt bazlı çapraz doğrulama)", fontsize=12)
        fig.tight_layout()
    return fig


def _place_labels(ax, fig, items, fontsize=7.5):
    """Nokta etiketlerini çakışmadan yerleştirir: aday konumları sırayla dener, dolu olanı atlar (açgözlü)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig.tight_layout()
    canvas = fig.canvas if hasattr(fig.canvas, 'get_renderer') else FigureCanvasAgg(fig)
    canvas.draw()
    rend = canvas.get_renderer()
    cands = [(0, 7, "center", "bottom"), (0, -7, "center", "top"), (8, 0, "left", "center"),
             (-8, 0, "right", "center"), (0, 22, "center", "bottom"), (0, -22, "center", "top"),
             (18, 14, "left", "bottom"), (-18, 14, "right", "bottom"), (18, -14, "left", "top"),
             (-18, -14, "right", "top"), (0, 38, "center", "bottom"), (0, -38, "center", "top")]
    pts = [ax.transData.transform((x, y)) for x, y, _ in items]
    placed = []

    def hit(a, b, pad=4.0):
        return not (a.x1 + pad < b.x0 or b.x1 + pad < a.x0 or a.y1 + pad < b.y0 or b.y1 + pad < a.y0)

    from matplotlib.transforms import Bbox
    dots = [Bbox([[px - 5, py - 5], [px + 5, py + 5]]) for px, py in pts]
    ax_bb = ax.get_window_extent(rend)
    for (x, y, ad), (px, py) in zip(items, pts):
        best = None
        for dx, dy, ha, va in cands:
            a = ax.annotate(ad, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fontsize,
                            arrowprops=dict(arrowstyle="-", lw=0.5, color="#888888") if abs(dy) > 14 or abs(dx) > 14 else None)
            bb = a.get_window_extent(rend)
            ok = (ax_bb.x0 <= bb.x0 and bb.x1 <= ax_bb.x1 and ax_bb.y0 <= bb.y0 and bb.y1 <= ax_bb.y1
                  and not any(hit(bb, o) for o in placed) and not any(hit(bb, d) for d in dots))
            if ok:
                best = bb
                break
            a.remove()
        if best is None:                                  # hiçbiri uymadıysa ilk adayı kullan
            dx, dy, ha, va = cands[0]
            a = ax.annotate(ad, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fontsize)
            best = a.get_window_extent(rend)
        placed.append(best)


def support_vs_f1(species_table, rho=None, dpi=100, p=None):
    """Tür başına segment sayısı (log) - F1 serpme grafiği; küçük n'de tüm türler adlandırılır."""
    from ..ml.classification_diagnosis import WEAK_F1
    with style_context():
        t = species_table.dropna(subset=["f1"]).sort_values("segment")
        fig = Figure(figsize=(8.5, 5.4), dpi=dpi)
        ax = fig.add_subplot(111)
        aralik = "f1_min" in t.columns and "f1_max" in t.columns and bool((t["f1_max"] > t["f1_min"]).any())
        zayif = t["f1_min"] if "f1_min" in t.columns else t["f1"]      # zayıflık: en kötü modele göre
        cols = [RENK["kirmizi"] if v < WEAK_F1 else RENK["mavi"] for v in zayif]
        if aralik:                                                       # modeller arası en düşük–en yüksek F1
            ax.vlines(t["segment"], t["f1_min"], t["f1_max"], color=RENK["gri"], lw=1.4, alpha=0.7, zorder=2)
        ax.scatter(t["segment"], t["f1"], c=cols, s=40, zorder=3)
        ax.set_xscale("log")
        ax.axhline(WEAK_F1, color=RENK["gri"], lw=1, ls="--")
        # Y ekseni 0'dan başlamaz (tüm türler yüksek olabilir); eşik çizgisi her zaman görünür kalır.
        lo = min(float(zayif.min()), WEAK_F1) - 0.12 if len(t) else 0.0
        ax.set_ylim(max(0.0, np.floor(lo * 10) / 10), 1.03)
        ax.text(0.995, WEAK_F1 + 0.008, f"zayıf eşiği: F1 = {WEAK_F1}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color=RENK["gri"])
        label_all = len(t) <= 15
        items = [(float(r["segment"]), float(r["f1"]),
                  _kisa_ad(r["etiket"] if "etiket" in r.index else r["tur"]))
                 for (_, r), zv in zip(t.iterrows(), zayif) if label_all or zv < WEAK_F1]
        if len(t):
            lo_x, hi_x = float(t["segment"].min()), float(t["segment"].max())
            adaylar = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 1000]
            tk = [a for a in adaylar if lo_x * 0.85 <= a <= hi_x * 1.15] or [int(lo_x), int(hi_x)]
            ax.xaxis.set_major_locator(FixedLocator(tk))
            ax.xaxis.set_major_formatter(FixedFormatter([str(a) for a in tk]))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(lo_x * 0.85, hi_x * 1.15)
        ax.set_xlabel("Tür başına gerçek segment sayısı (log ölçek)")
        ax.set_ylabel("F1 skoru (çapraz doğrulama" + (", model ortalaması)" if aralik else ")"))
        if aralik:
            ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=RENK["mavi"], markersize=7,
                                      label="model ortalaması"),
                               Line2D([0], [0], color=RENK["gri"], lw=1.4, label="modeller arası en düşük–en yüksek")],
                      loc="lower right", fontsize=8)
        ek = ""
        if rho is not None:
            ek = f"  (Spearman ρ = {rho:.2f}" + (f", p = {p:.3f}" if p is not None else "") + f", n = {len(t)})"
        ax.set_title("Örnek Sayısı ve Başarı" + ek, fontsize=12)
        _place_labels(ax, fig, items)   # eksenler/başlık kesinleştikten sonra
    return fig
