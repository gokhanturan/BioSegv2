"""Tür birikim grafiği: sade görünüm (gölgesiz çizgi + yeni tür çubukları) ve isteğe bağlı gölgeli görünüm."""
import numpy as np
import pandas as pd

from src.visualization import richness_plots as rp


def _curve(n=10):
    k = np.arange(1, n + 1)
    return pd.DataFrame({"k": k, "beklenen_tur": 20 * (1 - np.exp(-k / 4)), "ss": np.full(n, 1.0)})


def test_simple_view_is_single_panel_without_shading_or_comment():
    c = _curve()
    fig = rp.accumulation(c)
    assert len(fig.axes) == 1
    assert all(type(x).__name__ != "PolyCollection" for x in fig.axes[0].collections)
    assert not fig.axes[0].texts                      # çizgi üzerinde yorum metni yok


def test_optional_bar_panel_matches_curve_differences():
    c = _curve()
    top, bottom = rp.accumulation(c, cubuk=True).axes
    gains = [p.get_height() for p in bottom.patches]
    assert len(gains) == len(c) - 1 and np.allclose(gains, np.diff(c["beklenen_tur"]))


def test_shaded_view_and_tiny_curves():
    assert len(rp.accumulation(_curve(), golge=True).axes) == 1
    for n in (1, 2):
        assert rp.accumulation(_curve(n)) is not None


def test_species_at_frequency_lists_species_and_points_and_handles_empty():
    from src.data import richness_core as rc
    inc = pd.DataFrame({"A": [1, 1, 1], "B": [1, 0, 0], "C": [1, 1, 0]}, index=["10", "2", "7"]).astype(bool)
    assert rc.species_at_frequency(inc, 3) == [("A", ["2", "7", "10"])]
    assert rc.species_at_frequency(inc, 1) == [("B", ["10"])]
    assert rc.species_at_frequency(inc, 2) == [("C", ["2", "10"])]
    assert rc.species_at_frequency(inc, 9) == [] and rc.species_at_frequency(None, 1) == []


def test_dialog_click_helper_shows_list_or_explains_gap(monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import src.ui.main_window as mw
    from types import SimpleNamespace
    inc = pd.DataFrame({"A": [1, 1], "B": [1, 0]}, index=["1", "2"]).astype(bool)
    msgs = []
    monkeypatch.setattr(mw.QMessageBox, "information", staticmethod(lambda *a, **k: msgs.append(a[1])))
    d = mw.SpeciesRichnessDialog.__new__(mw.SpeciesRichnessDialog)
    mw.QDialog.__init__(d)
    d.summary = SimpleNamespace(point_incidence=inc)
    d._show_species_at_frequency(2)
    assert d._freq_dialog.windowTitle() == "Tam 2 noktada belirlenen türler" and not msgs
    d._show_species_at_frequency(5)
    assert msgs == ["Bu değerde tür yok"]


def test_click_on_end_marker_writes_point_and_species_count_and_click_away_clears():
    c = _curve(12)
    fig = rp.accumulation(c)
    fig.canvas.draw()
    ax = fig.axes[0]
    x, y = fig.acc_points
    px, py = ax.transData.transform((x[-1], y[-1]))
    assert rp.accumulation_label_at(fig, px + 3, py - 2)
    assert fig.acc_label.get_text() == f"12 nokta → {y[-1]:.0f} tür"
    assert not rp.accumulation_label_at(fig, px - 150, py + 150) and fig.acc_label is None
    mx, my = ax.transData.transform((x[3], y[3]))                    # çizgi üzerindeki başka bir nokta da çalışır
    assert rp.accumulation_label_at(fig, mx, my) and fig.acc_label.get_text().startswith("4 nokta")


def test_habitat_colors_identical_in_points_chart_and_presence_heatmap():
    rng = np.random.default_rng(0)
    types = pd.Series(["Koru a"] * 5 + ["Makilik"] * 4 + ["Taşlık"] * 2 + ["Baltalık"] * 3, index=[str(i) for i in range(14)])
    inc = pd.DataFrame(rng.random((14, 9)) > 0.6, index=types.index, columns=[f"s{i}" for i in range(9)])
    counts = inc.sum(axis=1).sort_values(ascending=False)
    counts.index = [f"{i} ({types[i]})" for i in counts.index]
    f1 = rp.point_richness_chart(counts, all_types=list(types.values))
    f2 = rp.presence_heatmap(inc, types)
    l1 = f1.axes[0].get_legend()
    l2 = f2.legends[0]
    a = {t.get_text(): h.get_markerfacecolor() for h, t in zip(l1.legend_handles, l1.get_texts())}
    b = {t.get_text().split(" · ")[1]: h.get_markerfacecolor() for h, t in zip(l2.legend_handles, l2.get_texts())}
    assert a == b and len(a) == 4
    # grafikte yalnızca bir kısım nokta çizilse bile renkler değişmez
    f3 = rp.point_richness_chart(counts.iloc[:5], all_types=list(types.values))
    l3 = f3.axes[0].get_legend()
    c3 = {t.get_text(): h.get_markerfacecolor() for h, t in zip(l3.legend_handles, l3.get_texts())}
    assert all(a[k] == v for k, v in c3.items())
