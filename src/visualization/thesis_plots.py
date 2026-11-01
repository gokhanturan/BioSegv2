"""Tez şekilleri için çapraz doğrulama ve uzman–model zenginlik grafikleri (saf matplotlib)."""
import numpy as np
from matplotlib.figure import Figure

from .plot_style import RENK, style_context

_YONTEM_AD = {
    "birdnet": "BirdNET 2.4",
    "perch": "Perch 2.0",
    "birlesik_birdnet_perch": "BirdNET + Perch (birleşik)",
    "MLP_BirdNET": "BirdNET + MLP",
    "LR_BirdNET": "BirdNET + Lojistik regresyon",
    "LR_MFCC": "MFCC + Lojistik regresyon",
    "SifirAtis_Perch": "Perch hazır çıktı (sıfır-atış)",
}
_YONTEM_RENK = {"birdnet": RENK["mavi"], "perch": RENK["yesil"], "birlesik_birdnet_perch": RENK["mor"],
                "MLP_BirdNET": RENK["mavi"], "LR_BirdNET": RENK["acik_mavi"],
                "LR_MFCC": RENK["turuncu"], "SifirAtis_Perch": RENK["gri"]}


def cv_methods(summary_rows, metric="makro_f1", dpi=100):
    """
    Yöntem karşılaştırması (kat ortalaması ± s.s.). summary_rows: cv_yontem_ozeti.csv satırları
    (yontem, sinif_agirligi, <metric>_ort, <metric>_ss). Sınıf ağırlığı varsa MLP/LR için
    ağırlıklı satır, yoksa ağırlıksız satır kullanılır; sıfır-atış tek satırdır.
    """
    with style_context():
        best = {}
        for r in summary_rows:
            k = r["yontem"]
            cw = str(r["sinif_agirligi"]).lower() in ("true", "1")
            if k not in best or cw:        # ağırlıklı satır varsa onu tercih et
                best[k] = r
        order = [k for k in ("birlesik_birdnet_perch", "perch", "birdnet",
                             "SifirAtis_Perch", "LR_MFCC", "LR_BirdNET", "MLP_BirdNET") if k in best]
        fig = Figure(figsize=(8.5, 1.0 * len(order) + 2.2), dpi=dpi)
        ax = fig.add_subplot(111)
        vals = [float(best[k][f"{metric}_ort"]) for k in order]
        errs = [float(best[k][f"{metric}_ss"]) for k in order]
        ax.barh([_YONTEM_AD[k] for k in order], vals, xerr=errs, capsize=4,
                color=[_YONTEM_RENK[k] for k in order])
        for i, v in enumerate(vals):
            ax.text(v + errs[i] + 0.015, i, f"{v:.3f}", va="center", fontsize=10)
        ax.set_xlim(0, 1.12)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(("Makro F1" if metric == "makro_f1" else "Doğruluk") + " (ortalama ± 1 s.s.)")
        ax.set_title("Yöntem Karşılaştırması (kayıt bazlı tekrarlı çapraz doğrulama)", fontsize=12)
        fig.tight_layout()
    return fig


def cv_per_class(rows, n=25, dpi=100):
    """Tür bazlı F1 (birleştirilmiş test tahminleri): en düşük n tür. rows: cv_tur_bazli.csv."""
    with style_context():
        use = [r for r in rows if str(r["sinif_agirligi"]).lower() in ("true", "1")] or list(rows)
        use = sorted(use, key=lambda r: float(r["f1"]))[:n]
        fig = Figure(figsize=(9, max(3, 0.36 * len(use) + 1.4)), dpi=dpi)
        ax = fig.add_subplot(111)
        names = [f"{r['tur']} (n={r['destek_segment']})" for r in use]
        f1 = [float(r["f1"]) for r in use]
        cols = [RENK["kirmizi"] if v < 0.5 else RENK["turuncu"] if v < 0.8 else RENK["yesil"] for v in f1]
        ax.barh(names, f1, color=cols)
        for i, v in enumerate(f1):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
        ax.set_xlim(0, 1.08)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("F1 skoru")
        ax.set_title(f"En Düşük F1'li {len(use)} Tür (çapraz doğrulama)", fontsize=12)
        fig.tight_layout()
    return fig


def richness_agreement(point_rows, summary_rows, dpi=100):
    """
    Uzman ve model zenginliği: (a) nokta başına tür sayısı dağılım grafiği (y = x çizgisi),
    (b) güven eşiğine göre nokta düzeyi var/yok uyumu (kesinlik, duyarlılık, Jaccard).
    point_rows: zenginlik_nokta.csv; summary_rows: zenginlik_ozet.csv.
    """
    with style_context():
        fig = Figure(figsize=(11, 4.8), dpi=dpi)
        ax = fig.add_subplot(1, 2, 1)
        e = np.array([float(r["tur_uzman"]) for r in point_rows])
        m = np.array([float(r["tur_model"]) for r in point_rows])
        rng = np.random.default_rng(0)
        ax.scatter(e + rng.uniform(-0.12, 0.12, len(e)), m + rng.uniform(-0.12, 0.12, len(m)),
                   color=RENK["mavi"], alpha=0.75, s=36, edgecolor="white", linewidth=0.5)
        top = max(e.max(), m.max()) + 1
        ax.plot([0, top], [0, top], color="black", linewidth=1, linestyle="--", label="y = x")
        ax.set_xlim(0, top); ax.set_ylim(0, top)
        ax.set_xlabel("Uzman etiketiyle nokta başına tür sayısı")
        ax.set_ylabel("Model çıktısıyla nokta başına tür sayısı")
        ax.set_title("Nokta Başına Tür Sayısı", fontsize=12)
        ax.legend(loc="upper left")

        ax2 = fig.add_subplot(1, 2, 2)
        thr = [float(r["guven_esigi"]) for r in summary_rows]
        for key, ad, c in (("var_yok_kesinlik_ort", "Kesinlik", RENK["turuncu"]),
                           ("var_yok_duyarlilik_ort", "Duyarlılık", RENK["yesil"]),
                           ("var_yok_jaccard_ort", "Jaccard", RENK["mavi"])):
            ax2.plot(thr, [float(r[key]) for r in summary_rows], marker="o", color=c, label=ad)
        ax2.plot(thr, [float(r["tespit_orani_ort"]) for r in summary_rows], marker="s",
                 color=RENK["gri"], linestyle=":", label="Tespit oranı (segment)")
        ax2.set_ylim(0, 1.02)
        ax2.set_xlabel("Model güven eşiği")
        ax2.set_ylabel("Nokta düzeyi var/yok uyumu")
        ax2.set_title("Güven Eşiğinin Etkisi", fontsize=12)
        ax2.legend(loc="lower left", fontsize=9)
        fig.suptitle("Uzman ve Model Tür Zenginliğinin Karşılaştırması", fontweight="bold", fontsize=13)
        fig.tight_layout()
    return fig
