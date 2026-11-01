import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

from src.ml import olcutler as o


def test_makro_f1_ignores_classes_absent_from_test():
    yt = np.array([0, 0, 1, 1])
    assert o.makro_f1(yt, np.array([0, 0, 1, 1])) == pytest.approx(1.0)


def test_makro_f1_equals_sklearn_when_all_classes_present():
    yt = np.array([0, 1, 2, 0, 1, 2])
    yp = np.array([0, 1, 2, 0, 2, 1])
    assert o.makro_f1(yt, yp) == pytest.approx(f1_score(yt, yp, average="macro"))


def test_makro_f1_does_not_reward_or_punish_prediction_of_absent_class_through_a_zero_row():
    yt = np.array([0, 0, 1, 1])
    yp = np.array([0, 0, 1, 2])
    assert o.makro_f1(yt, yp) == pytest.approx(f1_score(yt, yp, average="macro", labels=[0, 1]))


def test_agirlikli_f1_matches_sklearn_and_empty_input():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 4, 80)
    yp = np.where(rng.random(80) < 0.7, yt, rng.integers(0, 4, 80))
    assert o.agirlikli_f1(yt, yp) == pytest.approx(f1_score(yt, yp, average="weighted"))
    assert o.agirlikli_f1([], []) == 0.0 and o.makro_f1([], []) == 0.0


def test_makro_pr_auc_matches_sklearn_and_skips_absent_classes():
    rng = np.random.default_rng(1)
    yt = rng.integers(0, 3, 50)
    p = rng.random((50, 4))
    p /= p.sum(1, keepdims=True)
    esas = np.mean([average_precision_score(yt == c, p[:, c]) for c in range(3)])
    assert o.makro_pr_auc(yt, p) == pytest.approx(esas)


def test_havuzlanmis_matches_sklearn_and_skips_empty_rows():
    from sklearn.metrics import confusion_matrix
    rng = np.random.default_rng(2)
    yt = rng.integers(0, 4, 150)
    yp = np.where(rng.random(150) < 0.75, yt, rng.integers(0, 4, 150))
    m = o.havuzlanmis(confusion_matrix(yt, yp, labels=range(5)))
    assert m["makro_f1"] == pytest.approx(f1_score(yt, yp, average="macro", labels=range(4)))
    assert m["agirlikli_f1"] == pytest.approx(f1_score(yt, yp, average="weighted"))
    assert m["dogruluk"] == pytest.approx((yt == yp).mean())
