import os

import numpy as np
import pandas as pd
import pytest

from src.data import richness_insights as ri


class _S:
    pass


def _summary(n_points=12):
    rng = np.random.default_rng(1)
    inc = pd.DataFrame(rng.random((n_points, 25)) < 0.3, index=[str(i) for i in range(n_points)],
                       columns=[f"t{j}" for j in range(25)])
    inc.iloc[:, 0] = True
    s = _S()
    s.n_points = n_points
    s.n_species_observed = int(inc.any(axis=0).sum())
    s.point_incidence = inc
    s.point_types = pd.Series({str(i): ("A" if i % 2 else "B") for i in range(n_points)})
    s.point_effort = pd.DataFrame({"segment": np.arange(2, 2 + n_points) * 3, "kayit": 1,
                                   "tur": inc.sum(axis=1).values}, index=inc.index)
    s.pooled_curve = pd.DataFrame({"k": [1, 2, 3], "beklenen_tur": [5.0, 8.0, 9.0], "ss": [1, 1, 0]})
    s.location_rarefaction = None
    return s


def test_insights_basic_and_no_estimator_wording():
    items = ri.compute_insights(_summary())
    text = " ".join(i.title + " " + i.text for i in items)
    assert "Spearman" not in text and "Habitat tipleri" in text
    assert "Chao" not in text and "Jackknife" not in text
    assert items[0].kind == "ok"


def test_insights_empty():
    s = _summary()
    s.n_points = 0
    assert ri.compute_insights(s)[0].kind == "warn"


def test_rarefy_types_equal_k_and_no_sd_when_full():
    s = _summary(12)
    rows = ri.rarefy_types(s.point_incidence, s.point_types)
    assert len(rows) == 2 and rows[0][5] == 6
    assert all(r[4] == 0.0 for r in rows)   # her tip tüm 6 noktasıyla kullanıldı
    assert ri.rarefy_types(s.point_incidence, pd.Series({str(i): "A" for i in range(12)})) == []


def test_dashboard_dialog_builds_offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication
    from src.data.dataset_summary import compute_summary
    from src.ui.main_window import SpeciesRichnessDialog
    app = QApplication.instance() or QApplication([])
    rows = []
    for p in range(6):
        for k in range(3 + p):
            rows.append({"Turkish_Name": f"T{(p + k) % 9}", "Habitat_ID": str(p), "Habitat_Type": "A" if p % 2 else "B",
                         "Location": "X" if p < 4 else "Y"})
    df = pd.DataFrame(rows)
    d = SpeciesRichnessDialog(compute_summary(df))
    assert len(d._collect_figures()) >= 5
    d.close()
    app.processEvents()
