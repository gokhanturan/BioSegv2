"""cv_runner ve arayüzden çalıştırma: TensorFlow gerektirmeyen sahte bileşenlerle."""
import os
import types

import numpy as np
import pytest

from src.ml import cv_runner, pipeline


class S:
    def __init__(self, sp, rec): self.species, self.recording_id = sp, rec


def _fake_pipeline(monkeypatch, n_species=3):
    kept = [S(f"T{i % n_species}", f"r{i}") for i in range(30)]
    monkeypatch.setattr(pipeline, "load_or_extract_embeddings",
                        lambda **kw: (np.zeros((30, 4)), kept))
    rows = [{"tekrar": 1, "kat": 1, "sinif_agirligi": True, "dogruluk": .8, "makro_f1": .7,
             "n_train": 20, "n_val": 5, "n_test": 5, "n_test_kayit": 2}]
    pc = [{"sinif_agirligi": True, "tur": f"T{i}", "precision": .8, "recall": .8, "f1": .8, "destek_segment": 10}
          for i in range(n_species)]
    res = types.SimpleNamespace(fold_rows=rows, per_class_rows=pc, summary={}, class_names=[f"T{i}" for i in range(n_species)])
    monkeypatch.setattr(pipeline, "run_cross_validation", lambda *a, **kw: res)


def test_runner_writes_files(monkeypatch, tmp_path):
    _fake_pipeline(monkeypatch)
    seen = []
    r = cv_runner.run_evaluation(str(tmp_path), temel_cizgi=False, progress=lambda *a: seen.append(a))
    assert os.path.exists(tmp_path / "cv_tur_bazli.csv") and r["n_tur"] == 3
    assert r["dosyalar"] == ["cv_sonuclari.csv", "cv_tur_bazli.csv"]


def test_runner_needs_two_species(monkeypatch, tmp_path):
    _fake_pipeline(monkeypatch, n_species=1)
    with pytest.raises(ValueError, match="en az 2"):
        cv_runner.run_evaluation(str(tmp_path), temel_cizgi=False)


def test_runner_cancel(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "load_or_extract_embeddings",
                        lambda progress_callback=None, **kw: progress_callback("x", 1, 2))
    with pytest.raises(cv_runner.Cancelled):
        cv_runner.run_evaluation(str(tmp_path), cancelled=lambda: True)


def test_runner_with_improvements(monkeypatch, tmp_path):
    from src.ml import baselines
    from tests.test_baselines import _synthetic
    X, kept = _synthetic(n_species=4, rec_per_species=6)
    monkeypatch.setattr(pipeline, "load_or_extract_embeddings", lambda **kw: (X, kept))
    monkeypatch.setattr(baselines, "load_or_extract_mfcc", lambda k, **kw: X.copy())
    rows = [{"tekrar": r, "kat": k, "sinif_agirligi": True, "dogruluk": .9, "makro_f1": .85,
             "n_train": 40, "n_val": 10, "n_test": 20, "n_test_kayit": 4} for r in (1, 2, 3) for k in (1, 2, 3, 4, 5)]
    classes = sorted({s.species for s in kept})
    pc = [{"sinif_agirligi": True, "tur": c, "precision": .9, "recall": .9, "f1": .9, "destek_segment": 18} for c in classes]
    res = types.SimpleNamespace(fold_rows=rows, per_class_rows=pc, class_names=classes,
                                summary={True: {"dogruluk": (.9, .02), "makro_f1": (.85, .02)}})
    monkeypatch.setattr(pipeline, "run_cross_validation", lambda *a, **kw: res)
    r = cv_runner.run_evaluation(str(tmp_path), sifir_atis=False, mlp_topluluk=0)
    assert "cv_secici_tahmin.csv" in r["dosyalar"] and "cv_yontem_ozeti.csv" in r["dosyalar"]
    from src.ml import classification_diagnosis as C
    ozet = C.read_semicolon_csv(str(tmp_path / "cv_yontem_ozeti.csv"))
    assert {"Prototip_BirdNET", "Topluluk_BirdNET"} <= {o["yontem"] for o in ozet}
    d = C.diagnose(_df_for(classes), ozet, C.read_semicolon_csv(str(tmp_path / "cv_tur_bazli.csv")),
                   C.read_semicolon_csv(str(tmp_path / "cv_eslestirilmis_fark.csv")),
                   selective_rows=C.load_selective(str(tmp_path)))
    assert "Seçici tahmin" in d.format() and "Prototip" in d.format()


def _df_for(classes):
    import pandas as pd
    return pd.DataFrame([{"Scientific_Name": c, "Original_Recording_Name": f"{c}_{i % 3}"}
                         for c in classes for i in range(18)])


def test_runner_writes_and_clears_fingerprint(monkeypatch, tmp_path):
    from src.ml import classification_diagnosis as C
    import pandas as pd
    pd.DataFrame([{"Original_Recording_Name": "r1", "Start_Time": "0", "End_Time": "3", "Scientific_Name": "T0"}]) \
        .to_csv(tmp_path / "bird_sound_dataset.csv", sep=";", index=False, encoding="utf-8-sig")
    C.write_fingerprint(str(tmp_path), {"n_segment": 99, "n_tur": 9, "n_kayit": 9, "hash": "eski"})
    _fake_pipeline(monkeypatch)
    r = cv_runner.run_evaluation(str(tmp_path), temel_cizgi=False)
    fp = C.load_fingerprint(str(tmp_path))
    assert fp["hash"] != "eski" and fp["n_segment"] == 1 and C.FINGERPRINT_FILE in r["dosyalar"]
    # yarıda kesilen çalıştırma: eski iz silinmiş olmalı
    monkeypatch.setattr(pipeline, "load_or_extract_embeddings", lambda **kw: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        cv_runner.run_evaluation(str(tmp_path), temel_cizgi=False)
    assert C.load_fingerprint(str(tmp_path)) is None
