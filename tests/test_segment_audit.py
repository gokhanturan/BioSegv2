import numpy as np
import pytest

from src.ml import segment_audit as sa

sf = pytest.importorskip("soundfile")


def _wav(path, sr, seconds, amp, freq=1000.0, channels=1):
    t = np.arange(int(sr * seconds)) / sr
    y = amp * np.sin(2 * np.pi * freq * t)
    if channels == 2:
        y = np.stack([y, y], axis=1)
    sf.write(str(path), y, sr, subtype="PCM_16")


def test_audit_file_measures_header_and_level(tmp_path):
    p = tmp_path / "a.wav"
    _wav(p, 22050, 3.0, 0.1)
    r = sa.audit_file(str(p))
    assert r["ornekleme_hizi_hz"] == 22050 and r["kanal"] == 1 and r["sure_sn"] == 3.0
    assert abs(r["tepe_dbfs"] - (-20.0)) < 0.1 and abs(r["rms_dbfs"] - (-23.0)) < 0.2
    assert r["kirpilma_orani"] == 0 and 900 <= r["enerji_%99_hz"] <= 1100 and r["nyquist_hz"] == 11025


def test_audit_file_unreadable_and_silent(tmp_path):
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"not a wav")
    assert "hata" in sa.audit_file(str(bad))
    z = tmp_path / "z.wav"
    sf.write(str(z), np.zeros(1000), 8000, subtype="PCM_16")
    r = sa.audit_file(str(z))
    assert r["rms_dbfs"] is None and "enerji_%99_hz" not in r


def test_summarize_groups_by_species(tmp_path):
    files, sp = [], {}
    for i, (name, sr, amp) in enumerate([("A", 22050, 0.1), ("A", 22050, 0.05), ("B", 32000, 0.2)]):
        p = tmp_path / f"{i}.wav"
        _wav(p, sr, 3.0, amp)
        files.append(sa.audit_file(str(p)))
        sp[str(p)] = name
    out = {r["tur"]: r for r in sa.summarize(files + [{"dosya": "yok.wav", "hata": "x"}], sp)}
    assert out["TÜMÜ"]["n_dosya"] == 3 and out["TÜMÜ"]["okunamayan_dosya"] == 1
    assert out["TÜMÜ"]["ornekleme_hizlari_hz"] == "22050/32000"
    assert out["A"]["n_dosya"] == 2 and out["B"]["ornekleme_hizlari_hz"] == "32000"
