"""Eşleştirilmiş anlamlılık testi (Nadeau–Bengio + Holm), kat satırlarının kaydı ve karşılaştırma penceresi."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.ml import model_registry as mr, last_result_store as lrs, pipeline as pl
from src.ml import model_significance as ms


def _rows(f1s, n_train=80, n_val=20, n_test=25, n_rep=2):
    """f1s: sıralı makro F1 listesi; kat sayısı = len/n_rep."""
    k = len(f1s) // n_rep
    return [{"tekrar": i // k + 1, "kat": i % k + 1, "dogruluk": f, "makro_f1": f, "n_train": n_train,
             "n_val": n_val, "n_test": n_test} for i, f in enumerate(f1s)]


def test_paired_nb_matches_manual_formula():
    from scipy import stats
    rng = np.random.default_rng(0)
    base = rng.uniform(0.7, 0.9, 10)
    a, b = base + 0.03 + rng.normal(0, 0.01, 10), base
    res = ms.paired_nb_test(_rows(list(a)), _rows(list(b)))
    d = a - b
    J = 10
    se = np.sqrt((1 / J + 25 / 100) * d.var(ddof=1))
    p_manual = 2 * stats.t.sf(abs(d.mean() / se), J - 1)
    assert res["n_kat"] == 10 and res["fark"] == pytest.approx(d.mean())
    assert res["p_nb"] == pytest.approx(p_manual)
    assert res["p_nb"] > res["p_naif"]                       # düzeltme muhafazakârdır
    assert res["ga_alt"] < res["fark"] < res["ga_ust"] and res["kazanilan_kat"] == int((d > 0).sum())
    flipped = ms.paired_nb_test(_rows(list(b)), _rows(list(a)))
    assert flipped["fark"] == pytest.approx(-res["fark"]) and flipped["p_nb"] == pytest.approx(res["p_nb"])


def test_degenerate_cases_and_mismatched_folds():
    same = ms.paired_nb_test(_rows([0.8] * 6), _rows([0.8] * 6))
    assert same["p_nb"] == 1.0 and same["fark"] == 0.0
    shifted = ms.paired_nb_test(_rows([0.9] * 6), _rows([0.8] * 6))
    assert shifted["p_nb"] == 0.0 and shifted["kazanilan_kat"] == 6
    with pytest.raises(ValueError):
        ms.paired_nb_test(_rows([0.8] * 6), _rows([0.8] * 4))                  # farklı kat kümesi
    with pytest.raises(ValueError):
        ms.paired_nb_test(_rows([0.8, 0.7]), _rows([0.8, 0.7]))                 # < 3 kat
    with pytest.raises(ValueError):
        ms.paired_nb_test(_rows([0.8] * 6, n_test=25), _rows([0.7] * 6, n_test=26))   # test boyutu farklı


def test_holm_adjust():
    assert ms.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert ms.holm_adjust([0.5]) == [0.5]
    assert ms.holm_adjust([0.4, 0.9]) == pytest.approx([0.8, 0.9])
    assert max(ms.holm_adjust([0.6, 0.7, 0.8])) <= 1.0


def test_family_assignment():
    assert ms.family_of("perch", "birdnet") == "ana"
    assert ms.family_of(ms.COMBINED_ID, "birdnet") == "birlesik" == ms.family_of("perch", ms.COMBINED_ID)
    assert ms.family_of("birdnet", "baska_model") == "kesif"


def _stored(model_id, f1s, iz="abc", with_rows=True, with_cv=True, **kw):
    spec = mr.get_model(model_id)
    cv = None
    if with_cv:
        cv = {"dogruluk": (0.9, 0.02), "makro_f1": (float(np.mean(f1s)), 0.03)}
        if with_rows:
            cv["kat_satirlari"] = _rows(f1s, **kw)
            cv["bolme_izi"] = iz
    tr = SimpleNamespace(cv_summary=cv)
    return spec, SimpleNamespace(result=SimpleNamespace(training_result=tr))


def test_compare_stored_families_orientation_and_holm():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.80, 0.9, 10)
    stored = [
        _stored("birdnet", list(base)),
        _stored("perch", list(base + rng.normal(0, 0.01, 10))),
        _stored("birlesik_birdnet_perch", list(base + 0.02 + rng.normal(0, 0.005, 10))),
    ]
    res = ms.compare_stored(stored)
    rows = res["satirlar"]
    fams = [r["aile"] for r in rows]
    assert fams.count("ana") == 1 and fams.count("birlesik") == 2 and fams.count("kesif") == 0 and len(fams) == 3
    assert fams == sorted(fams, key=["ana", "birlesik", "kesif"].index)
    ana = next(r for r in rows if r["aile"] == "ana")
    assert (ana["a_id"], ana["b_id"]) == ("birdnet", "perch") and ana["p_duzeltilmis"] == ana["p_nb"]
    for r in (r for r in rows if r["aile"] == "birlesik"):
        assert r["a_id"] == "birlesik_birdnet_perch"                          # önceden belirlenmiş yön
        assert r["p_duzeltilmis"] >= r["p_nb"]
    cells = ms.table_cells(rows[0])
    assert len(cells) == len(ms.TABLE_COLUMNS)
    assert "ok" not in cells[-1]
    csv_text = ms.to_csv(res)
    assert csv_text.splitlines()[0].startswith("aile;A;B") and len(csv_text.splitlines()) == 1 + len(rows)


def test_compare_stored_refuses_different_splits_and_reports_missing_rows():
    f = [0.8, 0.82, 0.78, 0.81, 0.79, 0.8]
    res = ms.compare_stored([
        _stored("birdnet", f, iz="A"), _stored("perch", f, iz="B"),
        _stored("birlesik_birdnet_perch", f, with_rows=False)])
    assert len(res["satirlar"]) == 1
    r = res["satirlar"][0]
    assert r["durum"].startswith("bölmeler farklı") and r["p_nb"] is None and r["anlamli"] is None
    assert ms.table_cells(r)[2] == "—"
    text = " ".join(res["uyarilar"])
    assert "BirdNET + Perch" in text and "yeniden çalıştırın" in text


def _three_cvs(X, samples):
    kw = dict(n_splits=3, n_repeats=2, class_weight_options=(False,), max_epochs=1)
    return (pl.run_cross_validation(X, samples, **kw), pl.run_cross_validation(X * 2, samples, **kw),
            pl.run_cross_validation(X[:-3], samples[:-3], **kw))


def test_cv_stores_fold_rows_and_split_fingerprint_roundtrip(tmp_path):
    from src.ml.dataset_builder import TrainingSample
    rng = np.random.default_rng(0)
    samples, rows = [], []
    for si in range(3):
        for r in range(5):
            for j in range(3):
                samples.append(TrainingSample(f"s{si}_{r}_{j}.wav", f"Sp{si}", f"rec{si}_{r}"))
                v = rng.normal(size=6)
                v[si] += 3
                rows.append(v)
    X = np.array(rows, dtype=np.float32)
    from unittest.mock import patch
    from tests.test_cv_extra_metrics import _fake_train
    with patch.object(pl, "build_and_train_mlp", side_effect=_fake_train):
        cv1, cv2, cv3 = _three_cvs(X, samples)
    assert cv1.split_hash == cv2.split_hash                       # özdeş bölmeler, farklı gömme: eşleştirilebilir
    assert cv1.split_hash != cv3.split_hash                       # farklı örnek kümesi: eşleşmez
    s = pl.summarize_cv(cv1, False, 3, 2)
    assert len(s["kat_satirlari"]) == 6 and s["bolme_izi"] == cv1.split_hash
    assert {"tekrar", "kat", "makro_f1", "n_train", "n_val", "n_test"} <= set(s["kat_satirlari"][0])

    from tests.test_classifier_dialog import _result
    r = _result(True)
    r.training_result.cv_summary = s
    lrs.save_last_result(str(tmp_path), r, "x", {}, {"hash": "h"})
    back = lrs.load_last_result(str(tmp_path)).result.training_result.cv_summary
    assert back["bolme_izi"] == s["bolme_izi"] and back["kat_satirlari"] == s["kat_satirlari"]


def test_comparison_dialog_shows_significance_table(tmp_path, monkeypatch):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from src.ui import main_window as mw
    from tests.test_classifier_dialog import _result
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    rng = np.random.default_rng(3)
    base = rng.uniform(0.8, 0.9, 6)
    for mid, f in (("birdnet", base), ("perch", base - 0.05 + rng.normal(0, 0.005, 6))):
        r = _result(True)
        spec, st = _stored(mid, list(f))
        cv = st.result.training_result.cv_summary
        cv.update({"n_kat": 3, "n_tekrar": 2, "n_deger": 6, "dogruluk_aralik": (0.8, 0.9),
                   "makro_f1_aralik": (0.7, 0.9)})
        r.training_result.cv_summary = cv
        lrs.save_last_result(mr.result_dir(mid, str(tmp_path)), r, "x", {}, {"hash": "h"})
    d = mw.ModelComparisonDialog()
    assert d.table.rowCount() == 2 and d.sig_table.rowCount() == 1
    assert d.sig_table.columnCount() == len(ms.TABLE_COLUMNS)
    assert "BirdNET 2.4 − Perch 2.0" in d.sig_table.item(0, 1).text()
    assert d.sig_table.item(0, 7).text() == "anlamlı fark"
    assert d.table.height() < 200 and d.sig_table.height() < 200               # boş alan bırakılmaz
