"""BirdNET modeli ayrı bir 'model' aşamasında yüklenir; arayüz bunu gösterir (TensorFlow gerekmez)."""
import os
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _samples():
    from src.ml.dataset_builder import TrainingSample
    return [TrainingSample(filepath=f"f{i}.wav", species="A", recording_id=f"r{i}") for i in range(3)]


def test_model_stage_comes_before_embedding_and_loads_once():
    from src.ml import pipeline as pl
    stages = []

    def fake_extract(samples, model=None, progress_callback=None, stage_callback=None, **kw):
        stage_callback("model", 0, 1)
        stage_callback("model", 1, 1)
        for i in range(len(samples)):
            progress_callback(i + 1, len(samples))
        return np.zeros((len(samples), 4)), samples, 0, None

    with patch.object(pl, "build_training_samples", return_value=(_samples(), [], {})), \
            patch.object(pl.bne, "extract_embeddings_for_samples", side_effect=fake_extract):
        pl.load_or_extract_embeddings(progress_callback=lambda st, c, t: stages.append((st, c, t)))
    assert stages[0] == ("model", 0, 1) and stages[1] == ("model", 1, 1) and stages[2][0] == "embedding"


def test_model_load_failure_is_reported_as_clear_error():
    from src.ml import pipeline as pl
    with patch.object(pl, "build_training_samples", return_value=(_samples(), [], {})), \
            patch.object(pl.bne, "extract_embeddings_for_samples",
                         side_effect=pl.bne.BirdNETModelLoadError("model.tflite yok")):
        with pytest.raises(pl.InsufficientDataError, match="model.tflite yok"):
            pl.load_or_extract_embeddings()


def test_cache_from_other_backend_is_not_reused(tmp_path):
    from src.ml import pipeline as pl
    samples = _samples()
    keys = np.array([f"{s.filepath}||{s.species}||{s.recording_id}" for s in samples])
    cache = str(tmp_path / "c.npz")
    np.savez(cache, X=np.ones((3, 1536)), keys=keys)           # etiketsiz (eski) önbellek = Perch
    calls = []

    def fake_extract(samples, model=None, progress_callback=None, stage_callback=None, **kw):
        calls.append(1)
        return np.zeros((len(samples), 4)), samples, 0, None

    with patch.object(pl, "build_training_samples", return_value=(samples, [], {})), \
            patch.object(pl.bne, "extract_embeddings_for_samples", side_effect=fake_extract):
        X, _ = pl.load_or_extract_embeddings(cache_path=cache)
        assert calls and X.shape == (3, 4)                       # Perch önbelleği reddedildi, yeniden çıkarıldı
        calls.clear()
        X2, _ = pl.load_or_extract_embeddings(cache_path=cache)  # artık BirdNET etiketli: önbellekten
        assert not calls and X2.shape == (3, 4)


def test_dialog_shows_model_wait_then_normal_progress(tmp_path, monkeypatch):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from src.ui import main_window as mw
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: QtWidgets.QMessageBox.Ok)   # modal pencere açılmasın
    d = mw.ClassifierTrainingDialog()
    d._on_progress("model", 0, 1)
    assert d.progress_bar.maximum() == 0 and d._model_timer.isActive()      # hareketli çubuk + sayaç
    assert "yükleniyor / indiriliyor" in d.status_label.text() and "BirdNET" in d.status_label.text()
    d._on_progress("model", 1, 1)
    assert not d._model_timer.isActive()
    assert d.progress_bar.maximum() == 0 and "BirdNET modeli yüklendi" in d.status_label.text()   # sonraki aşama için bekleme
    d._on_progress("embedding", 5, 10)
    assert d.progress_bar.maximum() == 100 and d.progress_bar.value() == 50
    assert "5/10" in d.status_label.text() and "Adım 2/" in d.status_label.text()
    d._on_progress("cv", 0, 10)
    assert d.progress_bar.maximum() == 0 and "Çapraz doğrulama başlıyor" in d.status_label.text()
    d._on_progress("cv", 3, 10)
    assert d.progress_bar.value() == 30 and "3/10" in d.status_label.text()
    d._on_progress("training", 0, 100)
    assert d.progress_bar.maximum() == 0 and "MLP" in d.status_label.text()
    d._on_progress("training", 7, 100)
    assert "epoch 7/100" in d.status_label.text()
    d._on_progress("done", 1, 1)
    assert "kaydediliyor" in d.status_label.text()
    d._on_progress("model", 0, 1)
    d._on_error("hata")                                                      # hata gelince sayaç durmalı
    assert not d._model_timer.isActive() and d.progress_bar.maximum() == 100


def test_clear_saved_results_removes_only_result_files(tmp_path):
    from src.ml import last_result_store as lrs
    for name in lrs.CLEARABLE_FILES + ("bird_sound_dataset.csv", "baska.txt"):
        (tmp_path / name).write_text("x")
    removed = lrs.clear_saved_results(str(tmp_path))
    assert sorted(removed) == sorted(lrs.CLEARABLE_FILES)
    assert (tmp_path / "bird_sound_dataset.csv").exists() and (tmp_path / "baska.txt").exists()
    assert lrs.clear_saved_results(str(tmp_path)) == []


def test_dialog_clear_button_resets_view(tmp_path, monkeypatch):
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from src.ui import main_window as mw
    from src.ml import last_result_store as lrs
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    (tmp_path / lrs.FILE_NAME).write_text("{}")
    d = mw.ClassifierTrainingDialog()
    d.details_view.setPlainText("eski")
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.Yes)
    d._on_clear_results()
    assert not (tmp_path / lrs.FILE_NAME).exists()
    assert d.details_view.toPlainText() == "" and "1 dosya silindi" in d.status_label.text()
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.No)
    (tmp_path / lrs.FILE_NAME).write_text("{}")
    d._on_clear_results()
    assert (tmp_path / lrs.FILE_NAME).exists()          # Hayır denince silinmez
