"""Yazılımın veri setinin boyutundan/bölge sayısından bağımsız çalıştığını doğrular
(tek bölge / az nokta ve çok bölge / yüzlerce nokta)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from src.data import coverage_report as C
from src.data import dataset_summary as D
from src.visualization import richness_plots as R


def _make(n_points, regions, n_species=120, seed=0):
    rng = np.random.default_rng(seed)
    w = 1 / np.arange(1, n_species + 1)
    w /= w.sum()
    rows = []
    for p in range(1, n_points + 1):
        loc = regions[p % len(regions)]
        for r in range(2):
            for _ in range(5):
                sp = int(rng.choice(n_species, p=w))
                rows.append({"Habitat_ID": p, "Location": loc, "Habitat_Type": f"T{p % 4}",
                             "Original_Recording_Name": f"r{p}_{r}", "Turkish_Name": f"Tur{sp}",
                             "Recording_Date": f"2026-0{4 + p % 3}-1{p % 9}", "Recording_Time": f"{6 + p % 8:02d}:00:00"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("n_points,regions", [(314, ["Il1", "Il2", "Il3"]), (50, ["Tek"]), (12, ["A", "B"]), (1, ["A"])])
def test_pipeline_runs_for_any_design(n_points, regions):
    df = _make(n_points, regions)
    design = pd.DataFrame({"Habitat_ID": [str(i) for i in range(1, n_points + 1)],
                           "Location": [regions[i % len(regions)] for i in range(1, n_points + 1)],
                           "Habitat_Type": [f"T{i % 4}" for i in range(1, n_points + 1)]})
    cfg = dict(C.DEFAULT_CONFIG, use_catalog_design=False)
    rep = C.build_coverage_report(df, cfg, design=design)
    assert f"Tasarım: {n_points} nokta; etiketli {n_points}" in rep.format()
    sm = D.compute_summary(df, config=cfg)
    txt = sm.format_richness_summary()
    assert f"Etiketlenmiş nokta (Habitat_ID) sayısı: {n_points}" in txt
    fig = R.point_richness_chart(sm.habitat_richness)
    assert fig.get_figheight() <= 20            # nokta sayısı ne olursa olsun şekil okunur boyutta


def test_bar_chart_caps_categories():
    s = pd.Series(np.arange(300, 0, -1), index=[f"k{i}" for i in range(300)])
    fig = R.bar_chart(s, "Başlık", "x")
    assert fig.get_figheight() <= 20
    assert "en yüksek 40 / 300" in fig.axes[0].get_title()


def test_point_richness_switches_to_histogram_for_many_points():
    many = pd.Series(np.random.default_rng(0).integers(2, 20, 200), index=[str(i) for i in range(200)])
    assert "dağılım" in R.point_richness_chart(many).axes[0].get_title()
    few = many.iloc[:10]
    assert "dağılım" not in R.point_richness_chart(few).axes[0].get_title()
