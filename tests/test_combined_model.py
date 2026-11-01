"""Birleşik BirdNET + Perch modeli: bileşenlerin hizalanması, önbellekler, kayıt defteri ve uçtan uca hat."""
import os
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.ml import model_registry as mr, pipeline as pl
from src.ml.dataset_builder import TrainingSample
from tests.test_cv_extra_metrics import _fake_train

DB, DP = 4, 6        # sahte BirdNET / Perch gömme boyutları


def _samples(n_per=4, recs=4):
    return [TrainingSample(filepath=f"{sp}{r}_{k}.wav", species=sp, recording_id=f"{sp}{r}")
            for sp in "ABC" for r in range(recs) for k in range(n_per)]


def _fake_make(feats, calls, drop=None):
    """backend -> çıkarıcı; `drop[backend]` = o backend'in atlayacağı dosya adları."""
    drop = drop or {}

    def make(backend, report):
        def run(smp, cb):
            calls.append(backend)
            keep = [s for s in smp if s.filepath not in drop.get(backend, ())]
            idx = {s.filepath: i for i, s in enumerate(smp)}
            return feats[backend][[idx[s.filepath] for s in keep]], keep, len(smp) - len(keep), "hata"
        return run
    return make


def _feats(samples):
    rng = np.random.default_rng(0)
    return {"birdnet": rng.normal(size=(len(samples), DB)).astype(np.float32),
            "perch": rng.normal(size=(len(samples), DP)).astype(np.float32)}


def test_combined_concatenates_in_fixed_order_and_aligns_rows(tmp_path):
    samples = _samples()
    feats, calls = _feats(samples), []
    with patch.object(pl, "_make_extractor", side_effect=_fake_make(feats, calls)):
        X, kept, n_fail, _ = pl._extract_with_cache(samples, "combined", None, lambda *a: None, lambda *a: None)
    assert X.shape == (len(samples), DB + DP) and n_fail == 0 and X.dtype == np.float32
    assert np.array_equal(X[:, :DB], feats["birdnet"]) and np.array_equal(X[:, DB:], feats["perch"])
    assert [s.filepath for s in kept] == [s.filepath for s in samples] and calls == ["birdnet", "perch"]


def test_combined_keeps_only_segments_present_in_both_and_counts_failures():
    samples = _samples()
    feats, calls = _feats(samples), []
    gone_b, gone_p = samples[1].filepath, samples[7].filepath
    with patch.object(pl, "_make_extractor",
                      side_effect=_fake_make(feats, calls, drop={"birdnet": {gone_b}, "perch": {gone_p}})):
        X, kept, n_fail, err = pl._extract_with_cache(samples, "combined", None, lambda *a: None, lambda *a: None)
    assert n_fail == 2 and err == "hata" and len(kept) == len(samples) - 2
    names = [s.filepath for s in kept]
    assert gone_b not in names and gone_p not in names
    for row, s in zip(X, kept):                                    # her satır doğru segmentin iki gömmesi
        i = [x.filepath for x in samples].index(s.filepath)
        assert np.array_equal(row[:DB], feats["birdnet"][i]) and np.array_equal(row[DB:], feats["perch"][i])


def test_combined_total_failure_returns_empty():
    samples = _samples(n_per=1, recs=2)
    feats, calls = _feats(samples), []
    allp = {s.filepath for s in samples}
    with patch.object(pl, "_make_extractor", side_effect=_fake_make(feats, calls, drop={"perch": allp})):
        X, kept, n_fail, _ = pl._extract_with_cache(samples, "combined", None, lambda *a: None, lambda *a: None)
    assert X.shape[0] == 0 and kept == [] and n_fail == len(samples)


def test_combined_uses_component_caches_and_loads_no_model(tmp_path):
    samples = _samples()
    feats, calls = _feats(samples), []
    caches = {"birdnet": (str(tmp_path / "b.npz"), ()), "perch": (str(tmp_path / "p.npz"), ())}
    with patch.object(pl, "_make_extractor", side_effect=_fake_make(feats, calls)):
        X1, *_ = pl._extract_with_cache(samples, "combined", None, lambda *a: None, lambda *a: None,
                                        component_caches=caches)
    assert calls == ["birdnet", "perch"] and os.path.isfile(caches["birdnet"][0]) and os.path.isfile(caches["perch"][0])
    with patch.object(pl, "_make_extractor", side_effect=AssertionError("model yüklenmemeli")):
        X2, *_ = pl._extract_with_cache(samples, "combined", None, lambda *a: None, lambda *a: None,
                                        component_caches=caches)
    assert np.array_equal(X1, X2)


def test_combined_perch_component_reuses_old_perch_cache(tmp_path):
    samples = _samples()
    feats, calls = _feats(samples), []
    old = str(tmp_path / "embedding_onbellek.npz")
    np.savez(old, X=feats["perch"], keys=pl._sample_keys(samples))            # etiketsiz eski önbellek = Perch
    caches = {"birdnet": (str(tmp_path / "b.npz"), ()), "perch": (str(tmp_path / "p.npz"), (old,))}
    with patch.object(pl, "_make_extractor", side_effect=_fake_make(feats, calls)):
        X, *_ = pl._extract_with_cache(samples, "combined", None, lambda *a: None, lambda *a: None,
                                       component_caches=caches)
    assert calls == ["birdnet"]                                                 # Perch önbellekten, BirdNET çıkarıldı
    assert np.array_equal(X[:, DB:], feats["perch"])


def test_combined_progress_is_monotonic_over_both_parts():
    samples = _samples(n_per=2, recs=2)
    seen = []

    def make(backend, report):
        def run(smp, cb):
            for i in range(1, len(smp) + 1):
                cb(i, len(smp))
            f = np.zeros((len(smp), 2), np.float32)
            return f, list(smp), 0, None
        return run
    with patch.object(pl, "_make_extractor", side_effect=make):
        pl._extract_with_cache(samples, "combined", None, lambda *a: None, lambda d, t: seen.append((d, t)))
    assert all(t == 2 * len(samples) for _d, t in seen)
    assert [d for d, _t in seen] == list(range(1, 2 * len(samples) + 1))


def test_registry_entry_and_component_cache_paths(tmp_path):
    root = str(tmp_path)
    spec = mr.get_model("birlesik_birdnet_perch")
    assert spec.backend == "combined" and spec.backend in pl.EMBEDDING_BACKENDS
    assert spec.gomme_boyutu == 1024 + 1536 and spec.egitici == "mlp" and spec.gomme_onbellegi is None
    assert "Hipotez" in spec.girdi_notu and "tek fark birleştirmedir" in spec.girdi_notu
    comp = mr.component_cache_paths("birlesik_birdnet_perch", root)
    assert set(comp) == set(pl.COMBINED_PARTS)
    assert comp["perch"] == (mr.embedding_cache_path("perch", root), mr.embedding_read_caches("perch", root))
    assert comp["birdnet"][0].startswith(mr.result_dir("birlesik_birdnet_perch", root))
    assert mr.component_cache_paths("birdnet", root) == {} == mr.component_cache_paths("perch", root)
    with pytest.raises(ValueError):
        pl._make_extractor("combined", lambda *a: None)


def test_training_thread_passes_component_caches(tmp_path, monkeypatch):
    pytest.importorskip("PyQt5.QtWidgets")
    from src.ui import main_window as mw
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        raise mw.ml_pipeline.InsufficientDataError("durdur")
    monkeypatch.setattr(mw.ml_pipeline, "run_full_training_pipeline", fake_run)
    t = mw.ClassifierTrainingThread(30, 2, 0.2, 0.15, 1, False, model_id="birlesik_birdnet_perch")
    t.error.connect(lambda _m: None)
    t.run()
    assert seen["backend"] == "combined" and seen["trainer"] == "mlp"
    assert set(seen["component_caches"]) == {"birdnet", "perch"}
    assert seen["embedding_cache_path"] is None


def test_full_pipeline_combined_end_to_end_with_cv_and_fold_rows(tmp_path):
    samples = _samples()
    rng = np.random.default_rng(1)
    cls = np.array(["ABC".index(s.species) for s in samples])
    feats = {"birdnet": (rng.normal(size=(len(samples), DB)) + cls[:, None] * 3).astype(np.float32),
             "perch": (rng.normal(size=(len(samples), DP)) + cls[:, None] * 3).astype(np.float32)}
    calls = []
    caches = {"birdnet": (str(tmp_path / "b.npz"), ()), "perch": (str(tmp_path / "p.npz"), ())}
    with patch.object(pl, "build_training_samples", return_value=(samples, [], {"A": 16, "B": 16, "C": 16})), \
         patch.object(pl, "_make_extractor", side_effect=_fake_make(feats, calls)), \
         patch.object(pl, "build_and_train_mlp", side_effect=_fake_train), \
         patch.object(pl, "save_model", lambda *a, **k: None):
        res = pl.run_full_training_pipeline(
            backend="combined", save_dir=str(tmp_path / "m"), component_caches=caches,
            cv_folds=3, cv_repeats=2, progress_callback=lambda *a: None)
    tr = res.training_result
    assert tr.cv_summary["makro_f1"][0] > 0.8 and len(tr.cv_summary["kat_satirlari"]) == 6
    assert tr.cv_summary["bolme_izi"] and calls == ["birdnet", "perch"]
