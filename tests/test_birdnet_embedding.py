"""src.ml.birdnet_embedding: sahte BirdNET modeliyle gömme çıkarma ve kayıt tarama."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml import birdnet_embedding as bne
from src.ml import richness_inference as RI
from src.ml.dataset_builder import TrainingSample

SR = 48000


class FakeModel:
    """Her girdi için tek pencere: gömme = [girdinin ortalama mutlak genliği, uzunluk(sn), 0, 0]."""
    def __init__(self):
        self.calls = []

    def encode_arrays(self, batch):
        self.calls.append(len(batch))
        emb = np.zeros((len(batch), 1, 4), dtype=np.float32)
        for i, (y, sr) in enumerate(batch):
            assert sr == SR and y.dtype == np.float32
            emb[i, 0, 0] = float(np.abs(y).mean())
            emb[i, 0, 1] = len(y) / SR
        return SimpleNamespace(embeddings=emb, embeddings_masked=np.zeros((len(batch), 1, 4), dtype=bool))


def _wav(path, seconds, amp=0.1):
    t = np.arange(int(SR * seconds)) / SR
    sf.write(str(path), (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SR)
    return str(path)


def test_extract_pads_short_segments_and_batches(tmp_path):
    samples = [TrainingSample(_wav(tmp_path / f"s{i}.wav", 2.0 if i == 0 else 3.0), "A", f"r{i}") for i in range(5)]
    m = FakeModel()
    seen = []
    X, kept, n_fail, err = bne.extract_embeddings_for_samples(
        samples, model=m, chunk_size=2, progress_callback=lambda c, t: seen.append((c, t)))
    assert X.shape == (5, 4) and len(kept) == 5 and n_fail == 0 and err is None
    assert np.allclose(X[:, 1], 3.0)                    # 2 sn'lik segment 3 sn'ye tamamlandı
    assert m.calls == [2, 2, 1] and seen[-1] == (5, 5)


def test_unreadable_file_is_skipped_and_reported(tmp_path):
    good = _wav(tmp_path / "ok.wav", 3.0)
    bad = str(tmp_path / "bozuk.wav")
    open(bad, "wb").write(b"FAKE")
    samples = [TrainingSample(good, "A", "r0"), TrainingSample(bad, "A", "r1")]
    X, kept, n_fail, err = bne.extract_embeddings_for_samples(samples, model=FakeModel())
    assert len(kept) == 1 and kept[0].filepath == good and X.shape[0] == 1 and n_fail == 1 and err


def test_model_failure_counts_all_files(tmp_path):
    class Boom:
        def encode_arrays(self, batch):
            raise RuntimeError("tflite hata")
    samples = [TrainingSample(_wav(tmp_path / f"s{i}.wav", 3.0), "A", f"r{i}") for i in range(3)]
    X, kept, n_fail, err = bne.extract_embeddings_for_samples(samples, model=Boom())
    assert len(kept) == 0 and X.shape == (0, bne.EMBEDDING_DIM) and n_fail == 3 and "tflite" in err


def test_embedder_splits_long_recording_into_3s_windows(tmp_path):
    p = _wav(tmp_path / "uzun.wav", 10.0)
    starts, emb = bne.BirdNETEmbedder(model=FakeModel()).embed_recording(p, hop=3.0)
    assert list(starts) == [0.0, 3.0, 6.0]              # 10 sn -> 3 tam pencere (kalan 1 sn atlanır)
    assert emb.shape == (3, 4) and np.allclose(emb[:, 1], 3.0)


def test_embedder_short_recording_gives_one_padded_window(tmp_path):
    p = _wav(tmp_path / "kisa.wav", 1.5)
    starts, emb = bne.BirdNETEmbedder(model=FakeModel()).embed_recording(p)
    assert list(starts) == [0.0] and emb.shape == (1, 4)


def test_scan_recording_uses_embedder_and_separate_cache(tmp_path):
    p = _wav(tmp_path / "k.wav", 9.0)
    m = FakeModel()
    emb = bne.BirdNETEmbedder(model=m)
    cache = str(tmp_path / "c")
    s1, e1 = RI.scan_recording(emb, p, hop=3.0, cache_dir=cache)
    n_calls = len(m.calls)
    s2, e2 = RI.scan_recording(emb, p, hop=3.0, cache_dir=cache)
    assert len(m.calls) == n_calls                      # ikinci çağrı önbellekten
    assert np.array_equal(e1, e2) and len(s1) == 3
    perch_file = RI._cache_file(cache, p, 3.0)           # Perch önbellek adıyla çakışmaz
    assert perch_file != RI._cache_file(cache, p, 3.0, "birdnet")


def test_missing_birdnet_package_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "birdnet", None)
    monkeypatch.setattr(bne, "_models", {})
    with pytest.raises(bne.BirdNETModelLoadError, match="pip install birdnet"):
        bne.get_birdnet_model()


def _fake_birdnet_package(root):
    pkg = root / "birdnet"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "import numpy as np\nfrom types import SimpleNamespace\n"
        "class _M:\n"
        "    def encode_arrays(self, batch):\n"
        "        emb = np.zeros((len(batch), 1, 4), dtype=np.float32)\n"
        "        for i, (y, sr) in enumerate(batch):\n"
        "            emb[i, 0, 0] = float(np.abs(y).mean()); emb[i, 0, 1] = len(y) / sr\n"
        "        return SimpleNamespace(embeddings=emb, embeddings_masked=np.zeros((len(batch), 1, 4), dtype=bool))\n"
        "def load(*a, **k):\n    return _M()\n", encoding="utf-8")


def test_subprocess_extraction_end_to_end(tmp_path, monkeypatch):
    fake_root = tmp_path / "fake"
    fake_root.mkdir()
    _fake_birdnet_package(fake_root)
    monkeypatch.setenv("PYTHONPATH", str(fake_root))
    good = [_wav(tmp_path / f"s{i}.wav", 3.0) for i in range(3)]
    bad = str(tmp_path / "bozuk.wav")
    open(bad, "wb").write(b"FAKE")
    samples = [TrainingSample(p, "A", f"r{i}") for i, p in enumerate(good[:2] + [bad] + good[2:])]
    stages, prog = [], []
    X, kept, n_fail, err = bne.extract_embeddings_for_samples(
        samples, chunk_size=2, progress_callback=lambda c, t: prog.append((c, t)),
        stage_callback=lambda st, c, t: stages.append((st, c, t)))
    assert [s.filepath for s in kept] == good and X.shape == (3, 4) and n_fail == 1 and err
    assert stages == [("model", 0, 1), ("model", 1, 1)] and prog[-1] == (4, 4)
    assert np.allclose(X[:, 1], 3.0)


def test_subprocess_reports_model_load_error(tmp_path, monkeypatch):
    fake_root = tmp_path / "fake"
    (fake_root / "birdnet").mkdir(parents=True)
    (fake_root / "birdnet" / "__init__.py").write_text(
        "def load(*a, **k):\n    raise RuntimeError('model dosyasi yok')\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(fake_root))
    samples = [TrainingSample(_wav(tmp_path / "s.wav", 3.0), "A", "r0")]
    with pytest.raises(bne.BirdNETModelLoadError, match="model dosyasi yok"):
        bne.extract_embeddings_for_samples(samples)


def test_log_tail_is_appended_to_cancel_message(tmp_path):
    log = tmp_path / "birdnet_session_x.log"
    log.write_text("a\nb\nGERCEK NEDEN: ImportError xyz\n", encoding="utf-8")
    msg = bne._with_log_tail(f"Analysis was cancelled. Please check the logs: {log}")
    assert "GERCEK NEDEN" in msg and msg.startswith("Analysis was cancelled")
    assert bne._with_log_tail("baska hata") == "baska hata"


def test_model_dir_redirected_when_default_path_is_non_ascii(tmp_path, monkeypatch):
    monkeypatch.delenv("BIRDNET_APP_DATA", raising=False)
    monkeypatch.setattr(bne, "_default_birdnet_dir", lambda: r"C:\Users\GökhanTURAN\AppData\Roaming\birdnet")
    monkeypatch.setattr(bne, "_project_root", lambda: str(tmp_path))
    d = bne.configure_model_dir()
    assert d == str(tmp_path / "birdnet_models") and os.path.isdir(d)
    assert os.environ["BIRDNET_APP_DATA"] == d
    monkeypatch.delenv("BIRDNET_APP_DATA")


def test_model_dir_untouched_for_ascii_path_or_user_setting(tmp_path, monkeypatch):
    monkeypatch.delenv("BIRDNET_APP_DATA", raising=False)
    monkeypatch.setattr(bne, "_default_birdnet_dir", lambda: r"C:\Users\Gokhan\AppData\Roaming\birdnet")
    assert bne.configure_model_dir() is None and "BIRDNET_APP_DATA" not in os.environ
    monkeypatch.setattr(bne, "_default_birdnet_dir", lambda: r"C:\Users\GökhanT\birdnet")
    monkeypatch.setenv("BIRDNET_APP_DATA", str(tmp_path))
    assert bne.configure_model_dir() is None
