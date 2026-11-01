"""Model kayıt defteri, model başına sonuç saklama, gömme önbelleği ve karşılaştırma penceresi."""
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.ml import model_registry as mr, last_result_store as lrs, pipeline as pl


def test_result_dirs_are_isolated(tmp_path):
    root = str(tmp_path)
    assert mr.result_dir("birdnet", root) == root
    perch = mr.result_dir("perch", root)
    assert perch != root and os.path.isdir(perch) and perch.startswith(root)
    assert mr.model_save_dir("birdnet", root) is None
    assert mr.model_save_dir("perch", root).startswith(perch)
    assert mr.embedding_cache_path("birdnet", root) is None
    assert mr.embedding_cache_path("perch", root).startswith(perch)
    with pytest.raises(ValueError):
        mr.get_model("yok")


def _stored_for(tmp_path, model_id, acc, with_cv=True):
    from tests.test_classifier_dialog import _result
    r = _result(True)
    if with_cv:
        r.training_result.cv_summary = {
            "n_kat": 5, "n_tekrar": 3, "n_deger": 15, "dogruluk": (acc, 0.03), "makro_f1": (acc - 0.02, 0.05),
            "dogruluk_aralik": (acc - 0.05, acc + 0.04), "makro_f1_aralik": (acc - 0.08, acc + 0.04),
            "agirlikli_f1": (acc, 0.03), "makro_pr_auc": (0.97, 0.02),
            "havuzlanmis": {"dogruluk": acc, "makro_f1": acc - 0.01, "agirlikli_f1": acc}}
    else:
        r.training_result.cv_summary = None
    folder = mr.result_dir(model_id, str(tmp_path))
    lrs.save_last_result(folder, r, "ayrıntı", {}, {"hash": "h1", "n_segment": 10})
    return r


def test_stored_results_are_per_model_and_do_not_overwrite(tmp_path):
    _stored_for(tmp_path, "birdnet", 0.945)
    assert [s.id for s, _ in mr.stored_results(str(tmp_path))] == ["birdnet"]
    _stored_for(tmp_path, "perch", 0.90)
    got = {s.id: st for s, st in mr.stored_results(str(tmp_path))}
    assert set(got) == {"birdnet", "perch"}
    assert got["birdnet"].result.training_result.cv_summary["dogruluk"][0] == pytest.approx(0.945)
    assert got["perch"].result.training_result.cv_summary["dogruluk"][0] == pytest.approx(0.90)
    lrs.clear_saved_results(mr.result_dir("perch", str(tmp_path)))
    assert [s.id for s, _ in mr.stored_results(str(tmp_path))] == ["birdnet"]


def _samples(n=4):
    from src.ml.dataset_builder import TrainingSample
    return [TrainingSample(filepath=f"f{i}.wav", species="A" if i % 2 else "B", recording_id=f"r{i}") for i in range(n)]


def test_embedding_cache_reuses_and_respects_backend(tmp_path):
    samples = _samples()
    cache = str(tmp_path / "c.npz")
    calls = []

    def fake_make(backend, report):
        def run(smp, cb):
            calls.append(backend)
            return np.ones((len(smp), 3)), list(smp), 0, None
        return run

    with patch.object(pl, "_make_extractor", side_effect=fake_make):
        X1, k1, nf, _ = pl._extract_with_cache(samples, "perch", cache, lambda *a: None, lambda *a: None)
        assert calls == ["perch"] and os.path.isfile(cache)
        X2, k2, _, _ = pl._extract_with_cache(samples, "perch", cache, lambda *a: None, lambda *a: None)
        assert calls == ["perch"]            # önbellekten okundu
        assert np.array_equal(X1, X2) and [s.filepath for s in k2] == [s.filepath for s in samples]
        pl._extract_with_cache(samples, "birdnet", cache, lambda *a: None, lambda *a: None)
        assert calls == ["perch", "birdnet"]  # farklı kaynak ASLA yeniden kullanılmaz
        pl._extract_with_cache(samples, "birdnet", None, lambda *a: None, lambda *a: None)
        assert calls == ["perch", "birdnet", "birdnet"]   # önbellek yok = her seferinde çıkarım


def test_cache_not_written_when_extraction_has_failures(tmp_path):
    samples = _samples()
    cache = str(tmp_path / "c.npz")
    with patch.object(pl, "_make_extractor",
                      return_value=lambda smp, cb: (np.ones((3, 2)), list(smp[:3]), 1, "hata")):
        pl._extract_with_cache(samples, "perch", cache, lambda *a: None, lambda *a: None)
    assert not os.path.exists(cache)


def test_comparison_rows_and_warnings(tmp_path):
    from src.ui import main_window as mw
    _stored_for(tmp_path, "birdnet", 0.945)
    _stored_for(tmp_path, "perch", 0.90, with_cv=False)
    rows, warns = mw.ModelComparisonDialog.build_rows(mr.stored_results(str(tmp_path)))
    by = {r[0]: r for r in rows}
    assert by["BirdNET 2.4"][1].startswith("%94.5") and "çapraz doğrulama (5×3)" in by["BirdNET 2.4"][9]
    assert by["Perch 2.0"][9].startswith("tek bölme")
    assert any("değerlendirme yöntemi farklı" in w for w in warns)


def test_comparison_warns_on_different_data(tmp_path):
    from src.ui import main_window as mw
    from tests.test_classifier_dialog import _result
    _stored_for(tmp_path, "birdnet", 0.9)
    lrs.save_last_result(mr.result_dir("perch", str(tmp_path)), _result(True), "x", {}, {"hash": "h2"})
    _, warns = mw.ModelComparisonDialog.build_rows(mr.stored_results(str(tmp_path)))
    assert any("farklı veri durumunda" in w for w in warns)


def test_dialogs_smoke(tmp_path, monkeypatch):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from src.ui import main_window as mw
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    _stored_for(tmp_path, "birdnet", 0.945)
    d = mw.ClassifierTrainingDialog(model_id="perch")
    assert "Perch 2.0" in d.windowTitle() and d.result_dir == mr.result_dir("perch", str(tmp_path))
    assert d.last_result is None                       # BirdNET'in kayıtlı sonucu Perch penceresine sızmaz
    d0 = mw.ClassifierTrainingDialog()
    assert d0.windowTitle() == "🧠 Tür Sınıflandırma Eğitimi" and d0.last_result is not None
    cmp = mw.ModelComparisonDialog()
    assert cmp.table.rowCount() == 1 and cmp.table.columnCount() == len(cmp.COLUMNS)


def test_training_card_opens_model_picker_with_all_choices(tmp_path, monkeypatch):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from src.ui import main_window as mw
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    _stored_for(tmp_path, "birdnet", 0.945)
    win = mw.BirdSoundApp()
    text = lambda b: " ".join(l.text() for l in b.findChildren(QtWidgets.QLabel))
    assert "2. Tür Sınıflandırma Eğitimi" in text(win.btn_train_classifier)

    dlg = mw.ModelPickerDialog(win, root=str(tmp_path))
    assert set(dlg.buttons) == {"birdnet", "perch", "birlesik_birdnet_perch", mw.ModelPickerDialog.COMPARE}
    assert "Kayıtlı sonuç" in dlg.buttons["birdnet"].text() and "Henüz eğitilmedi" in dlg.buttons["perch"].text()
    for b in dlg.buttons.values():   # yazı rengi açıkça koyu olmalı (genel stil sayfası beyaz yapıyor)
        assert "color:#1f2a37" in b.styleSheet().replace(" ", "")
    dlg.buttons["perch"].click()
    assert dlg.choice == "perch"
    win._open_model_choice(dlg.choice)
    assert win._model_dialogs["perch"].model_id == "perch" and win._model_dialogs["perch"].isVisible()
    win._open_model_choice(mw.ModelPickerDialog.COMPARE)
    assert win._model_comparison_dialog.isVisible()


def test_download_progress_text_for_perch_and_plain_text_for_birdnet(tmp_path, monkeypatch):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from src.ui import main_window as mw
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(mr, "download_cache_dirs", lambda mid: [str(cache)] if mid == "perch" else [])

    d = mw.ClassifierTrainingDialog(model_id="perch")
    d._update_model_wait_text()
    assert "henüz veri inmedi" in d.status_label.text()
    (cache / "w.bin").write_bytes(b"x" * (3 * 1024 * 1024))
    d._update_model_wait_text()
    t = d.status_label.text()
    assert "3 MB" in t and "MB/sn" in t and "Perch 2.0" in t

    b = mw.ClassifierTrainingDialog()
    b._update_model_wait_text()
    assert "yükleniyor / indiriliyor" in b.status_label.text() and "BirdNET" in b.status_label.text()
    assert "MB" not in b.status_label.text()


def _np_cache(path, samples, backend=None, dim=4):
    keys = pl._sample_keys(samples)
    kw = {} if backend is None else {"backend": np.array(backend)}
    np.savez(path, X=np.arange(len(samples) * dim, dtype=float).reshape(len(samples), dim), keys=keys, **kw)


def test_old_perch_cache_is_reused_without_loading_the_model(tmp_path):
    samples = _samples()
    old, own = str(tmp_path / "embedding_onbellek.npz"), str(tmp_path / "perch_own.npz")
    _np_cache(old, samples, backend=None)            # etiketsiz eski önbellek = Perch
    boom = patch.object(pl, "_make_extractor", side_effect=AssertionError("model yüklenmemeli"))
    with boom:
        X, kept, nf, _ = pl._extract_with_cache(samples, "perch", own, lambda *a: None, lambda *a: None,
                                                read_caches=(old,))
    assert X.shape == (4, 4) and nf == 0 and len(kept) == 4
    assert os.path.isfile(own)                       # kendi önbelleğine kopyalandı, eskisi değişmedi
    assert np.array_equal(np.load(old)["X"], X)


def test_other_backend_old_cache_is_not_reused(tmp_path):
    samples = _samples()
    old = str(tmp_path / "embedding_onbellek.npz")
    _np_cache(old, samples, backend="birdnet")       # BirdNET önbelleği Perch'e ASLA verilmez
    calls = []

    def fake_make(backend, report):
        def run(smp, cb):
            calls.append(backend)
            return np.zeros((len(smp), 2)), list(smp), 0, None
        return run
    with patch.object(pl, "_make_extractor", side_effect=fake_make):
        pl._extract_with_cache(samples, "perch", None, lambda *a: None, lambda *a: None, read_caches=(old,))
    assert calls == ["perch"]
    assert str(np.load(old)["backend"]) == "birdnet"  # eski dosya ezilmedi


def test_training_thread_passes_model_specific_paths(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5.QtWidgets")
    from src.ui import main_window as mw
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        raise mw.ml_pipeline.InsufficientDataError("durdur")
    monkeypatch.setattr(mw.ml_pipeline, "run_full_training_pipeline", fake_run)
    errs = []
    for mid in ("perch", "birdnet"):
        t = mw.ClassifierTrainingThread(30, 2, 0.2, 0.15, 1, False, model_id=mid)
        t.error.connect(errs.append)
        t.run()
        if mid == "perch":
            assert seen["backend"] == "perch"
            assert seen["save_dir"].startswith(str(tmp_path)) and seen["save_dir"].endswith("model")
            assert os.path.join("modeller", "perch") in seen["save_dir"]
            assert seen["embedding_cache_path"].endswith("perch_gomme_onbellegi.npz")
            assert seen["embedding_read_caches"] == (os.path.join(str(tmp_path), "embedding_onbellek.npz"),)
        else:
            assert seen["backend"] == "birdnet" and seen["save_dir"] is None
            assert seen["embedding_cache_path"] is None and seen["embedding_read_caches"] == ()
    assert errs == ["durdur", "durdur"]


def test_full_pipeline_perch_end_to_end_with_fakes(tmp_path):
    from src.ml.dataset_builder import TrainingSample
    from tests.test_cv_extra_metrics import _fake_train
    rng = np.random.default_rng(0)
    samples = [TrainingSample(filepath=f"{sp}{r}_{k}.wav", species=sp, recording_id=f"{sp}{r}")
               for sp in "ABC" for r in range(4) for k in range(4)]
    feats = np.array([rng.standard_normal(5) + "ABC".index(s.species) * 3 for s in samples])
    stages, saved = [], []

    def fake_make(backend, report):
        assert backend == "perch"
        return lambda smp, cb: (feats, list(smp), 0, None)
    own = str(tmp_path / "perch.npz")
    with patch.object(pl, "build_training_samples", return_value=(samples, [], {"A": 16, "B": 16, "C": 16})), \
         patch.object(pl, "_make_extractor", side_effect=fake_make), \
         patch.object(pl, "build_and_train_mlp", side_effect=_fake_train), \
         patch.object(pl, "save_model", side_effect=lambda m, c, d: saved.append(d)):
        res = pl.run_full_training_pipeline(
            backend="perch", save_dir=str(tmp_path / "m"), embedding_cache_path=own, cv_folds=0,
            progress_callback=lambda s, c, t: stages.append(s))
    assert res.save_dir == str(tmp_path / "m") and saved == [str(tmp_path / "m")]
    assert os.path.isfile(own) and res.training_result.test_accuracy > 0.8
    assert "training" in stages and stages[-1] == "done"


def test_comparison_warns_on_different_settings(tmp_path):
    pytest.importorskip("PyQt5.QtWidgets")
    from src.ui import main_window as mw
    a = _stored_for(tmp_path, "birdnet", 0.9)
    b = _stored_for(tmp_path, "perch", 0.9)
    stored = mr.stored_results(str(tmp_path))
    _rows, warns = mw.ModelComparisonDialog.build_rows(stored)
    assert not any("ayarları farklı" in w for w in warns)
    stored[1][1].settings = dict(stored[1][1].settings or {}, max_epochs=50, class_weights=True)
    _rows, warns = mw.ModelComparisonDialog.build_rows(stored)
    assert any("ayarları farklı" in w and "en çok epoch" in w and "sınıf ağırlığı" in w for w in warns)
