"""Çapraz doğrulama ek ölçütleri: ağırlıklı F1, makro PR-AUC, havuzlanmış değerler."""
import os
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _proba(rng, n, k):
    p = rng.random((n, k))
    return p / p.sum(axis=1, keepdims=True)


def test_fold_extra_metrics_match_sklearn():
    from src.ml.pipeline import fold_extra_metrics
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 60)
    proba = _proba(rng, 60, 4)
    w, ap = fold_extra_metrics(y, proba, 4)
    assert w == pytest.approx(f1_score(y, proba.argmax(1), average="weighted", labels=range(4), zero_division=0))
    assert ap == pytest.approx(np.mean([average_precision_score(y == c, proba[:, c]) for c in range(4)]))


def test_fold_extra_metrics_perfect_and_absent_class():
    from src.ml.pipeline import fold_extra_metrics
    y = np.array([0, 0, 1, 1])
    proba = np.array([[.9, .05, .05], [.8, .1, .1], [.1, .8, .1], [.1, .9, 0.0]])
    w, ap = fold_extra_metrics(y, proba, 3)
    assert w == pytest.approx(1.0) and ap == pytest.approx(1.0)     # 3. sınıf testte yok: ortalamaya girmez


def test_pooled_metrics_match_sklearn():
    from sklearn.metrics import confusion_matrix
    from src.ml.pipeline import pooled_metrics
    rng = np.random.default_rng(2)
    yt = rng.integers(0, 5, 200)
    yp = np.where(rng.random(200) < 0.7, yt, rng.integers(0, 5, 200))
    m = pooled_metrics(confusion_matrix(yt, yp, labels=range(5)))
    assert m["dogruluk"] == pytest.approx(accuracy_score(yt, yp))
    assert m["makro_f1"] == pytest.approx(f1_score(yt, yp, average="macro", labels=range(5), zero_division=0))
    assert m["agirlikli_f1"] == pytest.approx(f1_score(yt, yp, average="weighted", labels=range(5), zero_division=0))


def _res(rows, confusion=None):
    from src.ml.pipeline import CrossValidationResult
    pc = [{"sinif_agirligi": False, "tur": "a", "f1": 0.5}, {"sinif_agirligi": False, "tur": "b", "f1": 0.8}]
    return CrossValidationResult(rows, pc, {}, ["a", "b"], 4, 2, confusion=confusion)


def test_summary_has_new_metrics_and_old_rows_still_work():
    from src.ml.pipeline import summarize_cv
    base = {"sinif_agirligi": False, "dogruluk": 0.8, "makro_f1": 0.7}
    rows = [dict(base, agirlikli_f1=0.8, makro_pr_auc=0.9), dict(base, agirlikli_f1=0.6, makro_pr_auc=0.7)]
    out = summarize_cv(_res(rows, {False: np.array([[3, 1], [0, 4]])}), False, 5, 1)
    assert out["agirlikli_f1"][0] == pytest.approx(0.7) and out["makro_pr_auc"][0] == pytest.approx(0.8)
    assert out["havuzlanmis"]["dogruluk"] == pytest.approx(7 / 8)
    eski = summarize_cv(_res([base]), False, 5, 1)
    assert "agirlikli_f1" not in eski and "makro_pr_auc" not in eski and "havuzlanmis" not in eski
    assert eski["dogruluk"] == (0.8, 0.0)


class _CentroidModel:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.c = np.stack([X[y == i].mean(0) if np.any(y == i) else np.full(X.shape[1], 1e3) for i in range(self.k)])
        return self

    def predict(self, X, verbose=0):
        d = ((X[:, None, :] - self.c[None]) ** 2).sum(-1)
        e = np.exp(-d / 4.0)
        return e / e.sum(1, keepdims=True)


def _fake_train(X_train, y_train, X_val, y_val, n_classes, **kw):
    return _CentroidModel(n_classes).fit(X_train, y_train), {}


def test_run_cross_validation_adds_metrics_without_changing_old_ones():
    from src.ml import pipeline as pl
    from src.ml.dataset_builder import TrainingSample
    rng = np.random.default_rng(3)
    species = ["A", "B", "C"]
    samples, feats = [], []
    for si, sp in enumerate(species):
        for r in range(8):
            for _ in range(3):
                samples.append(TrainingSample(filepath=f"{sp}{r}.wav", species=sp, recording_id=f"{sp}{r}"))
                feats.append(rng.standard_normal(6) * 1.5 + si * 2.0)
    X = np.array(feats)
    with patch.object(pl, "build_and_train_mlp", side_effect=_fake_train):
        res = pl.run_cross_validation(X, samples, n_splits=4, n_repeats=2, class_weight_options=(False,), max_epochs=1)
    rows = res.fold_rows
    assert len(rows) == 8
    for r in rows:
        assert 0 <= r["agirlikli_f1"] <= 1 and 0 <= r["makro_pr_auc"] <= 1 and 0 <= r["dogruluk"] <= 1
    out = pl.summarize_cv(res, False, 4, 2)
    toplam = sum(r["n_test"] for r in rows)
    agirlikli_dogruluk = sum(r["dogruluk"] * r["n_test"] for r in rows) / toplam
    assert out["havuzlanmis"]["dogruluk"] == pytest.approx(agirlikli_dogruluk)
    assert out["dogruluk"][0] == pytest.approx(np.mean([r["dogruluk"] for r in rows]))


def test_dialog_shows_extra_cards_and_hint_for_old_results(tmp_path, monkeypatch):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from PyQt5.QtWidgets import QLabel
    from src.ui import main_window as mw
    from tests.test_classifier_dialog import _result
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))

    def texts(d):
        return " ".join(l.text() for l in d._result_layout.parentWidget().findChildren(QLabel))

    d = mw.ClassifierTrainingDialog()
    d._on_finished(_result(True))
    assert "Ek ölçütler" in texts(d) and "bu kayıtlı sonuçta yok" in texts(d)

    r = _result(True)
    r.training_result.cv_summary.update({
        "agirlikli_f1": (0.905, 0.04), "makro_pr_auc": (0.951, 0.03),
        "havuzlanmis": {"dogruluk": 0.93, "makro_f1": 0.932, "agirlikli_f1": 0.93}})
    d2 = mw.ClassifierTrainingDialog()
    d2._on_finished(r)
    t = texts(d2)
    assert "Ağırlıklı F1" in t and "Makro PR-AUC" in t and "Havuzlanmış makro F1" in t
    assert "0.95" in t and "0.93" in t and "segment sayısıyla ağırlıklandırıldı" in t
    assert "bu kayıtlı sonuçta yok" not in t
    det = d2.details_view.toPlainText()
    assert "Ağırlıklı F1: 0.905" in det and "Makro PR-AUC: 0.951" in det and "Havuzlanmış" in det
