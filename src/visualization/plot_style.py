"""
Ortak grafik stili (uygulama içi grafikler ve tez şekilleri aynı görünümü paylaşır).

Renkler renk körlüğüne dayanıklı (Okabe-Ito) paletten seçilmiştir; yazı tipi
boyutları basılı tezde okunur olacak şekilde ayarlanmıştır. Başlıklar Türkçedir.
"""
from contextlib import contextmanager

import matplotlib as mpl

# Okabe-Ito paleti
RENK = {
    "mavi": "#0072B2",
    "turuncu": "#E69F00",
    "yesil": "#009E73",
    "kirmizi": "#D55E00",
    "mor": "#CC79A7",
    "acik_mavi": "#56B4E9",
    "gri": "#7F7F7F",
    "siyah": "#000000",
}

RCPARAMS = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 2.0,
    "axes.unicode_minus": True,
}


def apply_style():
    """Stili genel olarak uygular (uygulama başlangıcında bir kez çağrılır)."""
    mpl.rcParams.update(RCPARAMS)


@contextmanager
def style_context():
    """Stili yalnızca blok içinde uygular."""
    with mpl.rc_context(RCPARAMS):
        yield
