"""Perch modelini internetsiz, yerel klasörden yükleme seçeneği (TensorFlow gerekmez)."""
import pytest

from src.detection import perch_classifier as pc


def test_env_dir_with_saved_model_is_used(tmp_path, monkeypatch):
    (tmp_path / "saved_model.pb").write_bytes(b"x")
    monkeypatch.setenv(pc.LOCAL_MODEL_ENV, str(tmp_path))
    assert pc.find_local_model_dir() == str(tmp_path)


def test_invalid_env_dir_raises_instead_of_silently_downloading(tmp_path, monkeypatch):
    monkeypatch.setenv(pc.LOCAL_MODEL_ENV, str(tmp_path))          # içinde saved_model.pb yok
    with pytest.raises(pc.PerchModelLoadError, match="saved_model.pb"):
        pc.find_local_model_dir()


def test_default_folder_next_to_app_is_detected(tmp_path, monkeypatch):
    monkeypatch.delenv(pc.LOCAL_MODEL_ENV, raising=False)
    monkeypatch.setattr("src.data.config.get_project_root", lambda: str(tmp_path))
    assert pc.find_local_model_dir() is None                        # klasör yok -> internetten indirilecek
    d = tmp_path / pc.LOCAL_MODEL_FOLDER
    d.mkdir()
    assert pc.find_local_model_dir() is None                        # boş klasör sayılmaz
    (d / "saved_model.pb").write_bytes(b"x")
    assert pc.find_local_model_dir() == str(d)


def test_resolve_downloads_with_kagglehub_and_finds_saved_model(tmp_path, monkeypatch):
    import sys
    import types
    from src.detection import perch_classifier as pc
    monkeypatch.delenv("PERCH_MODEL_DIR", raising=False)
    monkeypatch.delenv("PERCH_MODEL_HANDLE", raising=False)
    monkeypatch.setattr(pc, "find_local_model_dir", lambda: None)
    inner = tmp_path / "dl" / "sub"
    inner.mkdir(parents=True)
    (inner / "saved_model.pb").write_bytes(b"x")
    calls = []
    fake = types.SimpleNamespace(model_download=lambda h: (calls.append(h), str(tmp_path / "dl"))[1])
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    assert pc.resolve_model_location(True) == str(inner)
    assert calls == [pc.KAGGLE_HANDLE_CPU] and "perch_v2_cpu" in calls[0]
    monkeypatch.setenv("PERCH_MODEL_HANDLE", "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/2")
    pc.resolve_model_location(True)
    assert calls[-1].endswith("perch_v2_cpu/2")


def test_resolve_prefers_local_folder_and_fails_fast_without_kagglehub(tmp_path, monkeypatch):
    import sys
    from src.detection import perch_classifier as pc
    monkeypatch.setattr(pc, "find_local_model_dir", lambda: str(tmp_path))
    assert pc.resolve_model_location() == str(tmp_path)          # elle konan klasör varsa indirme yok
    monkeypatch.setattr(pc, "find_local_model_dir", lambda: None)
    monkeypatch.setitem(sys.modules, "kagglehub", None)          # import -> ModuleNotFoundError
    with pytest.raises(pc.PerchModelLoadError, match="pip install kagglehub"):
        pc.resolve_model_location()


def test_resolve_reports_download_failure_clearly(monkeypatch):
    import sys
    import types
    from src.detection import perch_classifier as pc
    monkeypatch.setattr(pc, "find_local_model_dir", lambda: None)
    monkeypatch.delenv("PERCH_MODEL_HANDLE", raising=False)

    def boom(h):
        raise OSError("bağlantı zaman aşımı")
    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(model_download=boom))
    with pytest.raises(pc.PerchModelLoadError, match="indirilemedi.*zaman aşımı|zaman aşımı"):
        pc.resolve_model_location()
