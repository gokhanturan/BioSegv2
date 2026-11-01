"""Öğrenme eğrisi: alt örnekleme kuralları ve çalıştırma."""
import os
import numpy as np
import pytest

from src.ml.dataset_builder import TrainingSample
from src.ml.learning_curve import subsample_train_indices, run_learning_curve, summarize_learning_curve, write_csvs
from src.ml.train_classifier import group_kfold_splits


def _samples(rec_per=(4, 6, 8), seg=3):
    out, rid = [], 0
    for si, k in enumerate(rec_per):
        for _ in range(k):
            rid += 1
            out += [TrainingSample(f"s{si}_{rid}_{j}.wav", f"Sp{si}", f"rec{rid}") for j in range(seg)]
    return out


def test_subsample_is_nested_per_species_min_one_and_deterministic():
    s = _samples()
    idx = np.arange(len(s))
    sets = {f: subsample_train_indices(idx, s, f, 7) for f in (0.25, 0.5, 0.75, 1.0)}
    assert len(sets[1.0]) == len(idx)
    for a, b in ((0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        assert set(sets[a]) <= set(sets[b])
    recs = lambda f, sp: {s[i].recording_id for i in sets[f] if s[i].species == sp}   # noqa: E731
    assert len(recs(0.25, "Sp0")) == 1 and len(recs(0.5, "Sp1")) == 3 and len(recs(0.25, "Sp2")) == 2
    assert all(len({s[i].species for i in sets[0.25]}) == 3 for _ in [0])
    again = subsample_train_indices(idx, s, 0.5, 7)
    assert np.array_equal(again, sets[0.5])
    for f in sets:                                           # kayıt bütünlüğü: bir kaydın segmentleri ya hep ya hiç
        kept = {s[i].recording_id for i in sets[f]}
        assert all((i in set(sets[f])) == (s[i].recording_id in kept) for i in idx)


def test_run_learning_curve_same_test_and_no_leak():
    pytest.importorskip("tensorflow")
    s = _samples((6, 6, 6), seg=3)
    X = np.random.default_rng(0).normal(size=(len(s), 16)).astype("float32")
    for i, x in enumerate(s):
        X[i, int(x.species[-1]) * 4:int(x.species[-1]) * 4 + 3] += 2.0
    prog = []
    res = run_learning_curve(X, s, fractions=(0.5, 1.0), n_splits=3, n_repeats=1, max_epochs=3,
                             progress_callback=lambda d, t: prog.append((d, t)))
    rows = res["satirlar"]
    assert len(rows) == 3 * 2 and prog[-1] == (6, 6)
    for k in (1, 2, 3):
        a, b = [r for r in rows if r["kat"] == k and r["oran"] == 0.5][0], [r for r in rows if r["kat"] == k and r["oran"] == 1.0][0]
        assert a["n_test_segment"] == b["n_test_segment"] and a["n_egitim_segment"] < b["n_egitim_segment"]
    ozet = res["ozet"]
    assert [o["oran"] for o in ozet] == [0.5, 1.0] and abs(ozet[-1]["makro_f1_fark_tam"]) < 1e-12


def test_summary_and_csv(tmp_path):
    rows = [{"tekrar": 1, "kat": k, "oran": o, "dogruluk": 0.5 + 0.1 * o + 0.01 * k, "makro_f1": 0.4 + 0.1 * o,
             "agirlikli_f1": 0.45, "n_egitim_segment": int(20 * o), "n_egitim_kayit": int(8 * o), "n_test_segment": 5}
            for k in (1, 2) for o in (0.5, 1.0)]
    ozet = summarize_learning_curve(rows)
    assert abs(ozet[0]["makro_f1_fark_tam"] + 0.05) < 1e-9
    files = write_csvs(str(tmp_path), {"satirlar": rows, "ozet": ozet})
    assert files == ["ogrenme_egrisi.csv", "ogrenme_egrisi_ozet.csv"] and os.path.isfile(tmp_path / files[1])


def test_pipeline_learning_curve_reuses_cv_and_persists(tmp_path):
    pytest.importorskip("tensorflow")
    from unittest.mock import patch
    from src.ml import pipeline as pl, last_result_store as store
    s = _samples((6, 6, 6), seg=3)
    X = np.random.default_rng(2).normal(size=(len(s), 16)).astype("float32")
    for i, x in enumerate(s):
        X[i, int(x.species[-1]) * 4:int(x.species[-1]) * 4 + 3] += 2.0
    stages = []
    with patch.object(pl, "build_training_samples", return_value=(s, [], {})), \
         patch.object(pl, "_extract_with_cache", return_value=(X, s, 0, None)):
        res = pl.run_full_training_pipeline(save_dir=str(tmp_path / "m"), max_epochs=3, cv_folds=3, cv_repeats=1,
                                            learning_curve=True, progress_callback=lambda a, b, c: stages.append(a))
    tr = res.training_result
    lc = tr.learning_curve
    assert lc and "ozet" in lc and [o["oran"] for o in lc["ozet"]] == [0.25, 0.5, 0.75, 1.0]
    assert "lc" in stages and "ÖĞRENME EĞRİSİ" in tr.format_summary()
    full = [o for o in lc["ozet"] if o["oran"] == 1.0][0]
    assert abs(full["makro_f1_ort"] - tr.cv_summary["makro_f1"][0]) < 1e-9
    assert abs(full["dogruluk_ort"] - tr.cv_summary["dogruluk"][0]) < 1e-9
    store.save_last_result(str(tmp_path), res, "d", {})
    back = store.load_last_result(str(tmp_path)).result.training_result.learning_curve
    assert [o["oran"] for o in back["ozet"]] == [0.25, 0.5, 0.75, 1.0]
    write_csvs(str(tmp_path), back)
    removed = store.clear_saved_results(str(tmp_path))
    assert "ogrenme_egrisi.csv" in removed and "son_siniflandirma_sonucu.json" in removed


def test_pipeline_without_curve_unchanged(tmp_path):
    pytest.importorskip("tensorflow")
    from unittest.mock import patch
    from src.ml import pipeline as pl
    s = _samples((6, 6, 6), seg=3)
    X = np.random.default_rng(3).normal(size=(len(s), 16)).astype("float32")
    with patch.object(pl, "build_training_samples", return_value=(s, [], {})), \
         patch.object(pl, "_extract_with_cache", return_value=(X, s, 0, None)):
        res = pl.run_full_training_pipeline(save_dir=str(tmp_path / "m"), max_epochs=2, cv_folds=3, cv_repeats=1)
    assert res.training_result.learning_curve is None


def test_ui_shows_curve_in_charts_and_checkbox(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from datetime import datetime
    from PyQt5.QtWidgets import QApplication
    from src.ui import main_window as mw
    from src.ml.pipeline import PipelineResult
    from src.ml.train_classifier import TrainingResult
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    rows = [{"tekrar": 1, "kat": k, "oran": o, "dogruluk": 0.4 + 0.2 * o, "makro_f1": 0.3 + 0.2 * o,
             "agirlikli_f1": 0.4, "n_egitim_segment": int(20 * o), "n_egitim_kayit": int(8 * o), "n_test_segment": 5}
            for k in (1, 2) for o in (0.25, 0.5, 0.75, 1.0)]
    lc = {"satirlar": rows, "ozet": summarize_learning_curve(rows), "sinif_adlari": ["A", "B"]}
    esik = [{"guven_esigi": t, "tespit_orani_ort": 1.0, "tur_uzman_ort": 5.0, "tur_model_ort": 6.0,
             "var_yok_jaccard_ort": 0.8, "var_yok_kesinlik_ort": 0.9, "var_yok_duyarlilik_ort": 0.9,
             "sahte_tur_nokta_cifti_ort": 2.0, "kacirilan_tur_nokta_cifti_ort": 1.0} for t in (0.0, 0.5)]
    zu = {"esikler": esik, "nokta": [
        {"nokta": "P1", "habitat_tipi": "Orman", "konum": "X", "n_kayit": 3, "tur_uzman": 3, "tur_model": 4,
         "sahte_turler": "Parus major", "kacirilan_turler": ""},
        {"nokta": "P2", "habitat_tipi": "", "konum": "", "n_kayit": 1, "tur_uzman": 2, "tur_model": 2,
         "sahte_turler": "", "kacirilan_turler": "Erithacus rubecula; Parus major"}], "n_nokta": 2, "n_tekrar": 1}
    tr = TrainingResult(0.6, 0.5, {"A": {"f1-score": 0.5}, "B": {"f1-score": 0.5}}, np.array([[3, 1], [1, 3]]),
                        ["A", "B"], 10, 2, 4, 4, 1, 2, learning_curve=lc, zenginlik_uyumu=zu)
    dlg = mw.ClassifierTrainingDialog()
    assert dlg.chk_curve.isChecked()
    dlg.chk_cv.setChecked(False)
    assert not dlg.chk_curve.isEnabled()
    dlg._show_result(PipelineResult(tr, str(tmp_path), {}), tr.format_summary(), datetime.now(), False, None)
    assert len(dlg.findChildren(mw.FigureCanvas)) >= 3
    assert "TÜR ZENGİNLİĞİ UYUMU" in dlg.details_view.toPlainText()
    from PyQt5.QtWidgets import QTableWidget
    tables = dlg.findChildren(QTableWidget)
    assert tables and tables[-1].rowCount() == 2 and tables[-1].columnCount() == 9
    assert "ÖĞRENME EĞRİSİ" in dlg.details_view.toPlainText()


def test_pipeline_richness_agreement_from_cv(tmp_path):
    pytest.importorskip("tensorflow")
    from unittest.mock import patch
    from src.ml import pipeline as pl, last_result_store as store
    s = _samples((6, 6, 6), seg=3)
    X = np.random.default_rng(5).normal(size=(len(s), 16)).astype("float32")
    for i, x in enumerate(s):
        X[i, int(x.species[-1]) * 4:int(x.species[-1]) * 4 + 3] += 3.0
    rec_ids = sorted({x.recording_id for x in s})
    import csv as _csv
    csv_path = tmp_path / "d.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.writer(f, delimiter=";")
        w.writerow(["Original_Recording_Name", "Habitat_ID"])
        for j, r in enumerate(rec_ids):
            w.writerow([r, f"P{j % 4}"])
    with patch.object(pl, "build_training_samples", return_value=(s, [], {})), \
         patch.object(pl, "_extract_with_cache", return_value=(X, s, 0, None)):
        res = pl.run_full_training_pipeline(save_dir=str(tmp_path / "m"), max_epochs=3, cv_folds=3, cv_repeats=1,
                                            point_csv_path=str(csv_path))
    zu = res.training_result.zenginlik_uyumu
    assert zu and "esikler" in zu and zu["n_nokta"] == 4
    assert [o["guven_esigi"] for o in zu["esikler"]] == [0.0, 0.5, 0.7, 0.9]
    assert zu["nokta"] and {"nokta", "tur_uzman", "tur_model", "sahte_turler", "kacirilan_turler", "habitat_tipi"} <= set(zu["nokta"][0])
    assert "TÜR ZENGİNLİĞİ UYUMU" in res.training_result.format_summary()
    store.save_last_result(str(tmp_path), res, "d", {})
    back = store.load_last_result(str(tmp_path)).result.training_result.zenginlik_uyumu
    assert back["n_nokta"] == 4
    from src.ml import richness_validation as RV
    assert set(RV.write_csvs(str(tmp_path), back)) == {"zenginlik_ozet.csv", "zenginlik_nokta.csv"}
    assert "zenginlik_ozet.csv" in store.clear_saved_results(str(tmp_path))


def test_cv_oof_covers_every_segment_once_per_repeat():
    pytest.importorskip("tensorflow")
    from src.ml import pipeline as pl
    s = _samples((6, 6, 6), seg=3)
    X = np.random.default_rng(6).normal(size=(len(s), 16)).astype("float32")
    cv = pl.run_cross_validation(X, s, n_splits=3, n_repeats=2, class_weight_options=(False,), max_epochs=2)
    assert len(cv.oof[False]) == 2
    for pred, conf in cv.oof[False]:
        assert (pred >= 0).all() and ((conf > 0) & (conf <= 1)).all()


def test_per_point_detail_names_and_info(tmp_path):
    import csv as _csv
    from src.ml import richness_validation as RV
    s = [TrainingSample(f"f{i}.wav", sp, rec) for i, (sp, rec) in enumerate(
        [("A", "r1"), ("B", "r1"), ("A", "r2"), ("B", "r3")])]
    pm = {"r1": "P1", "r2": "P1", "r3": "P2"}
    pred = (np.array([0, 0, 1, 0]), np.array([0.9, 0.9, 0.9, 0.9]))     # r1:B->A, r2:A->B, r3:B->A
    rows = RV.per_point_detail(s, pm, ["A", "B"], pred, 0.0, {"P1": {"tip": "Orman", "konum": "K"}})
    p1 = [r for r in rows if r["nokta"] == "P1"][0]
    assert (p1["tur_uzman"], p1["tur_model"]) == (2, 2) and p1["habitat_tipi"] == "Orman" and p1["n_kayit"] == 2
    p2 = [r for r in rows if r["nokta"] == "P2"][0]
    assert p2["sahte_turler"] == "A" and p2["kacirilan_turler"] == "B" and (p2["tur_uzman"], p2["tur_model"]) == (1, 1)
    f = tmp_path / "d.csv"
    with open(f, "w", newline="", encoding="utf-8-sig") as fh:
        w = _csv.writer(fh, delimiter=";")
        w.writerow(["Original_Recording_Name", "Habitat_ID", "Habitat_Type", "Location"])
        w.writerows([["r1", "P1", "Orman", "K"], ["r2", "P1", "Orman", "K"], ["r3", "P2", "Göl", "L"]])
    info = RV.load_point_info(str(f))
    assert info["P1"] == {"tip": "Orman", "konum": "K", "n_kayit": 2} and info["P2"]["tip"] == "Göl"


def test_scatter_hover_finds_marker_and_builds_html():
    from src.ml import richness_validation as RV
    rows = [{"nokta": "223", "habitat_tipi": "Sulak Alan", "konum": "Isparta", "n_kayit": 4, "n_segment": 49,
             "tur_uzman": 3, "tur_model": 5, "sahte_turler": "A; B", "kacirilan_turler": ""},
            {"nokta": "224", "habitat_tipi": "", "konum": "", "n_kayit": 1, "n_segment": 5,
             "tur_uzman": 2, "tur_model": 2, "sahte_turler": "", "kacirilan_turler": ""},
            {"nokta": "225", "habitat_tipi": "", "konum": "", "n_kayit": 1, "n_segment": 7,
             "tur_uzman": 2, "tur_model": 2, "sahte_turler": "", "kacirilan_turler": ""}]
    fig = RV.make_scatter_figure(rows, "M")
    ax = fig.axes[0]
    fig.canvas.draw()
    px, py = ax.transData.transform((3, 5))
    assert RV.find_marker_at(fig, px, py) == (3, 5)
    assert RV.find_marker_at(fig, px + 200, py + 200) is None
    groups = RV.group_points(rows)
    html = RV.hover_html(groups[(3, 5)])
    assert "223 (Sulak Alan) — Isparta" in html and "Segment: 49" in html and "Sahte türler: A; B" in html
    both = RV.hover_html(groups[(2, 2)])
    assert "224" in both and "225" in both and both.count("<hr>") == 1
