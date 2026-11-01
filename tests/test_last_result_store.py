"""Tür Sınıflandırma Eğitimi: son sonucu saklama / geri yükleme."""
import datetime as dt
import os

import numpy as np
import pytest

from src.ml import last_result_store as lrs

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _result(cv=True):
    from tests.test_classifier_dialog import _result as make
    return make(cv)


def test_roundtrip_keeps_everything_the_screen_needs(tmp_path):
    res = _result(True)
    when = dt.datetime(2026, 10, 13, 14, 32, 5)
    lrs.save_last_result(str(tmp_path), res, "ayrıntı metni", {"min_samples": 20, "cv": True}, {"hash": "ab", "n_segment": 9}, when)
    st = lrs.load_last_result(str(tmp_path))
    tr, tr0 = st.result.training_result, res.training_result
    assert st.saved_at == when and st.details == "ayrıntı metni" and st.settings["min_samples"] == 20
    assert st.fingerprint == {"hash": "ab", "n_segment": 9}
    assert tr.class_names == tr0.class_names and np.array_equal(tr.confusion_matrix, tr0.confusion_matrix)
    assert tr.test_accuracy == tr0.test_accuracy and tr.n_test == tr0.n_test and tr.excluded_species == ["A", "B", "C"]
    (a, asd), (f, fsd) = tr.cv_summary["dogruluk"], tr.cv_summary["makro_f1"]    # JSON'da liste, yine açılabilir
    assert (a, asd) == (0.912, 0.042) and tr.cv_summary["tur_f1"] == tr0.cv_summary["tur_f1"]
    assert st.result.save_dir == res.save_dir


def test_missing_corrupt_or_foreign_version_gives_none(tmp_path):
    assert lrs.load_last_result(str(tmp_path)) is None
    (tmp_path / lrs.FILE_NAME).write_text("{bozuk", encoding="utf-8")
    assert lrs.load_last_result(str(tmp_path)) is None
    (tmp_path / lrs.FILE_NAME).write_text('{"surum": 99}', encoding="utf-8")
    assert lrs.load_last_result(str(tmp_path)) is None


def test_numpy_values_are_serialisable(tmp_path):
    res = _result(False)
    res.training_result.per_class_report = {"Tur 0": {"f1-score": np.float32(0.5)}}
    res.training_result.test_predictions = [{"n": np.int64(3), "x": np.float64(0.25)}]
    lrs.save_last_result(str(tmp_path), res, "", {})
    st = lrs.load_last_result(str(tmp_path))
    assert st.result.training_result.test_predictions == [{"n": 3, "x": 0.25}]
