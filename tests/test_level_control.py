import numpy as np
import pytest

from src.ml import level_control as lc
from src.ml.dataset_builder import TrainingSample

sf = pytest.importorskip("soundfile")


def _sine(amp, n=22050, f=440.0, sr=22050):
    return amp * np.sin(2 * np.pi * f * np.arange(n) / sr)


def test_measure_level_and_silence():
    rms, peak = lc.measure_level(_sine(0.1))
    assert abs(peak - (-20.0)) < 0.01 and abs(rms - (-23.0)) < 0.05
    assert lc.measure_level(np.zeros(100)) == (lc.SESSIZ_DBFS, lc.SESSIZ_DBFS)


def test_normalize_rms_hits_target_and_peak_guard():
    out = lc.normalize_waveform(_sine(0.01), "rms", target_rms_dbfs=-30.0)
    assert abs(lc.measure_level(out)[0] - (-30.0)) < 0.05
    spiky = np.zeros(1000)
    spiky[10] = 0.5
    out2 = lc.normalize_waveform(spiky, "rms", target_rms_dbfs=-10.0)
    assert abs(np.abs(out2).max() - lc.TAVAN) < 1e-9                    # tepe sınırı aşılmaz
    with pytest.raises(ValueError):
        lc.normalize_waveform(_sine(0.1), "rms")


def test_normalize_peak_and_silent_unchanged():
    out = lc.normalize_waveform(_sine(0.02), "tepe")
    assert abs(lc.measure_level(out)[1] - lc.TEPE_HEDEF_DBFS) < 0.01
    z = np.zeros(50)
    assert np.array_equal(lc.normalize_waveform(z, "tepe"), z)
    with pytest.raises(ValueError):
        lc.normalize_waveform(z, "yok")


def _samples(tmp_path):
    out = []
    for i, amp in enumerate((0.01, 0.1, 0.3)):
        p = tmp_path / f"s{i}.wav"
        sf.write(str(p), _sine(amp), 22050, subtype="PCM_16")
        out.append(TrainingSample(filepath=str(p), species=f"sp{i % 2}", recording_id=f"r{i}"))
    return out


def test_level_features_and_median(tmp_path):
    s = _samples(tmp_path)
    F = lc.level_features(s)
    assert F.shape == (3, 2) and F[0, 0] < F[1, 0] < F[2, 0]
    assert lc.median_rms_dbfs(s) == pytest.approx(F[1, 0])


def test_write_copies_preserves_labels_and_equalizes_level(tmp_path):
    s = _samples(tmp_path)
    copies = lc.write_normalized_copies(s, "tepe", str(tmp_path / "out"))
    assert [c.species for c in copies] == [x.species for x in s]
    assert [c.recording_id for c in copies] == [x.recording_id for x in s]
    peaks = lc.level_features(copies)[:, 1]
    assert np.allclose(peaks, lc.TEPE_HEDEF_DBFS, atol=0.01)


def test_embeddings_for_mode_uses_extractor_and_cache(tmp_path, monkeypatch):
    from src.ml import birdnet_embedding as bne
    s = _samples(tmp_path)
    calls = []

    def fake(samples, **kw):
        calls.append([x.filepath for x in samples])
        return np.ones((len(samples), 4), dtype=np.float32), list(samples), 0, None

    monkeypatch.setattr(bne, "extract_embeddings_for_samples", fake)
    cache = str(tmp_path / "c.npz")
    X = lc.embeddings_for_mode(s, "rms", cache_path=cache, tmp_dir=str(tmp_path / "tmp"))
    assert X.shape == (3, 4) and len(calls) == 1 and "rms_" in calls[0][0]
    X2 = lc.embeddings_for_mode(s, "rms", cache_path=cache, tmp_dir=str(tmp_path / "tmp"))
    assert len(calls) == 1 and np.array_equal(X, X2)                      # önbellekten geldi

    def partial(samples, **kw):
        return np.ones((2, 4), dtype=np.float32), list(samples)[:2], 1, "x"
    monkeypatch.setattr(bne, "extract_embeddings_for_samples", partial)
    with pytest.raises(RuntimeError):
        lc.embeddings_for_mode(s, "tepe", tmp_dir=str(tmp_path / "tmp2"))
