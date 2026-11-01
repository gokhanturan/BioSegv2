import numpy as np

from src.ml.pipeline import CrossValidationResult, summarize_cv


def _res(confusion):
    rows = [{"sinif_agirligi": False, "dogruluk": 0.75, "makro_f1": 0.7, "tekrar": 1, "kat": 1}]
    pc = [{"sinif_agirligi": False, "tur": "a", "f1": 0.5}, {"sinif_agirligi": False, "tur": "b", "f1": 0.8}]
    return CrossValidationResult(rows, pc, {}, ["a", "b"], 4, 2, confusion=confusion)


def test_summary_carries_pooled_confusion_and_matches_f1():
    cm = np.array([[1, 1], [0, 2]])
    out = summarize_cv(_res({False: cm}), False, 5, 1)
    assert out["karisiklik"] == [[1, 1], [0, 2]]
    assert out["siniflar"] == ["a", "b"]
    tp, fp, fn = cm[0, 0], cm[:, 0].sum() - cm[0, 0], cm[0].sum() - cm[0, 0]
    assert 2 * tp / (2 * tp + fp + fn) == 2 / 3


def test_summary_without_confusion_still_works():
    out = summarize_cv(_res(None), False, 5, 1)
    assert "karisiklik" not in out
