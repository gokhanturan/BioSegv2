"""improvements.py testleri (TensorFlow gerektirmez; sentetik veri)."""
import numpy as np
import pytest

from src.ml import baselines, improvements as I
from tests.test_baselines import _synthetic


def test_prototype_and_lr_proba_shapes_and_rows():
    X, kept = _synthetic(n_species=4, rec_per_species=6)
    rows, sel = I.run_improvements(X, kept, n_splits=3, n_repeats=2)
    assert {r["yontem"] for r in rows} == {"Prototip_BirdNET", "Topluluk_BirdNET"}
    assert len(rows) == 2 * 3 * 2
    assert all(r["dogruluk"] > 0.8 for r in rows)
    assert {r["esik"] for r in sel} == set(I.THRESHOLDS)


def test_zero_shot_fusion_added_and_nan_scores_handled():
    X, kept = _synthetic()
    classes = sorted({s.species for s in kept})
    y = np.array([classes.index(s.species) for s in kept])
    zs = np.full((len(y), len(classes)), -np.inf)
    zs[np.arange(len(y)), y] = 1.0
    zs[:, 0] = np.nan                       # Perch etiket listesinde olmayan tür
    rows, _ = I.run_improvements(X, kept, zero_shot_scores=zs, n_splits=3, n_repeats=1)
    assert "LR_SifirAtis_Perch" in {r["yontem"] for r in rows}
    assert np.isfinite(I.sanitize_zero_shot(zs)).all()


def test_same_splits_as_baselines_enable_paired_test():
    X, kept = _synthetic()
    base = baselines.run_baselines(X, kept, n_splits=3, n_repeats=2)
    rows, _ = I.run_improvements(X, kept, n_splits=3, n_repeats=2)
    a = {(r["tekrar"], r["kat"]): r["n_test"] for r in base if r["yontem"] == "LR_BirdNET" and r["sinif_agirligi"]}
    b = {(r["tekrar"], r["kat"]): r["n_test"] for r in rows if r["yontem"] == "Prototip_BirdNET"}
    assert a == b


def test_selective_prediction_trades_coverage_for_accuracy():
    rng = np.random.default_rng(0)
    n, k = 600, 3
    y = rng.integers(0, k, n)
    conf = rng.uniform(0.34, 1.0, n)
    correct = rng.uniform(size=n) < conf              # iyi kalibre
    pred = np.where(correct, y, (y + 1) % k)
    P = np.full((n, k), 0.0)
    P[np.arange(n), pred] = conf
    P += ((1 - conf) / (k - 1))[:, None] * (P == 0)
    rows = I.selective_rows("m", y, P, k)
    kap = [r["kapsam"] for r in rows]; acc = [r["dogruluk_kabul"] for r in rows]
    assert kap == sorted(kap, reverse=True) and acc[-1] > acc[0]
    assert rows[0]["ece"] < 0.1


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    perfect = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    assert I.ece(y, perfect) == pytest.approx(0.0)
    wrong = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], float)
    assert I.ece(y, wrong) == pytest.approx(1.0)
