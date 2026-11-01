"""Yapay zekâ ile tür zenginliği sekmesi: kayıtlı zenginlik uyumlarının modeller arası tablosu."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.ml import ai_richness as ai, model_registry as mr, richness_validation as RV, last_result_store as lrs
from src.ml.dataset_builder import TrainingSample


def _zu(noise, seed=0):
    rng = np.random.default_rng(seed)
    classes = [f"Tur{i}" for i in range(4)]
    kept, pmap, pred, conf = [], {}, [], []
    for p in range(6):
        for r in range(2):
            rec = f"rec{p}_{r}"
            pmap[rec] = f"P{p}"
            for k in range(3):
                sp = classes[(p + k) % 4]
                kept.append(TrainingSample(f"{rec}_{k}.wav", sp, rec))
                wrong = rng.random() < noise
                pred.append(rng.integers(0, 4) if wrong else classes.index(sp))
                conf.append(rng.uniform(0.4, 1.0))
    preds = [(np.array(pred), np.array(conf))] * 2
    return RV.agreement_from_oof(kept, pmap, classes, preds), classes


def _stored(model_id, zu, classes):
    from src.ml.pipeline import PipelineResult
    from src.ml.train_classifier import TrainingResult
    tr = TrainingResult(test_accuracy=0.9, test_macro_f1=0.9, per_class_report={}, confusion_matrix=np.eye(4, dtype=int),
                        class_names=classes, n_train=10, n_val=2, n_test=3, n_train_recordings=5, n_val_recordings=1,
                        n_test_recordings=2, excluded_species=[], cv_summary=None, zenginlik_uyumu=zu)
    return mr.get_model(model_id), SimpleNamespace(result=PipelineResult(training_result=tr, save_dir="x", raw_species_counts={}))


def test_build_rows_per_model_and_threshold_and_warnings():
    zu1, cl = _zu(0.1)
    zu2, _ = _zu(0.4, seed=1)
    res = ai.build_rows([_stored("birdnet", zu1, cl), _stored("perch", zu2, cl),
                         _stored("birlesik_birdnet_perch", None, cl),
                         _stored("birlesik_birdnet_perch", {"hata": "yok"}, cl)])
    assert len(res["satirlar"]) == 2 * len(zu1["esikler"])
    assert all(len(r["hucreler"]) == len(ai.COLUMNS) for r in res["satirlar"])
    assert len(res["uyarilar"]) == 2 and "yeniden çalıştırın" in res["uyarilar"][0]
    a = next(r for r in res["satirlar"] if r["model_id"] == "birdnet" and r["esik"] == 0.0)
    b = next(r for r in res["satirlar"] if r["model_id"] == "perch" and r["esik"] == 0.0)
    assert a["hucreler"][5] != "—"                            # nokta başı |fark| doldurulur
    assert a["jaccard"] > b["jaccard"]                       # daha az gürültülü model daha uyumlu
    assert ai.to_csv(res).splitlines()[0].startswith("Model;Güven eşiği")
    assert "Perch 2.0: 4/77" in ai.coverage_text(res["modeller"], 77)


def test_build_rows_empty():
    r = ai.build_rows([])
    assert r["satirlar"] == [] and r["uyarilar"] == [] and ai.coverage_text([], 77) == ""


def test_dialog_has_ai_tab_with_table(tmp_path, monkeypatch):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from src.data.dataset_summary import compute_summary
    from src.ui import main_window as mw
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    zu, cl = _zu(0.2)
    for mid in ("birdnet", "perch"):
        spec, st = _stored(mid, zu, cl)
        lrs.save_last_result(mr.result_dir(mid, str(tmp_path)), st.result, "x", {}, {"hash": "h"})
    rows = [{"Turkish_Name": f"T{(p + k) % 9}", "Habitat_ID": str(p), "Habitat_Type": "A", "Location": "X"}
            for p in range(6) for k in range(3 + p)]
    d = mw.SpeciesRichnessDialog(compute_summary(pd.DataFrame(rows)))
    tabs = d.findChildren(QtWidgets.QTabWidget)[0]
    names = [tabs.tabText(i) for i in range(tabs.count())]
    assert "Yapay Zekâ" in names
    page = tabs.widget(names.index("Yapay Zekâ"))
    tables = page.findChildren(QtWidgets.QTableWidget)
    assert len(tables) == 1 and tables[0].rowCount() == 2 * len(zu["esikler"])
    d.close()
    app.processEvents()


def test_dialog_ai_tab_without_results(tmp_path, monkeypatch):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from src.data.dataset_summary import compute_summary
    from src.ui import main_window as mw
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    rows = [{"Turkish_Name": f"T{(p + k) % 9}", "Habitat_ID": str(p), "Habitat_Type": "A", "Location": "X"}
            for p in range(6) for k in range(3 + p)]
    d = mw.SpeciesRichnessDialog(compute_summary(pd.DataFrame(rows)))
    tabs = d.findChildren(QtWidgets.QTabWidget)[0]
    page = tabs.widget([tabs.tabText(i) for i in range(tabs.count())].index("Yapay Zekâ"))
    assert not page.findChildren(QtWidgets.QTableWidget)
    d.close()


def test_habitat_tab_states_its_limits(tmp_path, monkeypatch):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from src.data.dataset_summary import compute_summary
    from src.ui import main_window as mw
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    rows = [{"Turkish_Name": f"T{(p + k) % 9}", "Habitat_ID": str(p), "Habitat_Type": "A" if p < 5 else "B",
             "Location": "X" if p < 4 else "Y"} for p in range(6) for k in range(3 + p)]
    d = mw.SpeciesRichnessDialog(compute_summary(pd.DataFrame(rows)))
    texts = " ".join(l.text() for l in d.findChildren(QtWidgets.QLabel))
    assert "Bu sekmenin yorum sınırı" in texts and "Habitat tipi başına nokta" in texts
    assert "tezde bir fark iddiasının dayanağı olarak kullanılmamalıdır" in texts      # B tipinde 1 nokta
    d.close()


def test_interactive_map_has_legend_and_marks_low_effort():
    pytest.importorskip("folium")
    from src.data import location_map as lm
    pts = [lm.LocationPoint(37.5, 30.5, "A", 40, 12), lm.LocationPoint(37.6, 30.6, "B", 2, 1)]
    html = lm.build_interactive_map_html(pts)
    assert "Nasıl okunur" in html and "doğrudan karşılaştırılamaz" in html
    assert "güvenilir değil" in html and "#b45309" in html
