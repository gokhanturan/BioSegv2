"""embedding_comparison: sentetik gömmelerle bölme eşliği, birleştirme, istatistik ve BirdNET havuzlama."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.ml import embedding_comparison as ec
from src.ml.dataset_builder import TrainingSample


def _data(n_species=4, recs_per_species=6, seg_per_rec=5, d=16, seed=0, signal=3.0):
    rng = np.random.default_rng(seed)
    samples, X = [], []
    for sp in range(n_species):
        centre = rng.normal(0, 1, d) * signal
        for r in range(recs_per_species):
            rec_shift = rng.normal(0, 0.5, d)
            for s in range(seg_per_rec):
                samples.append(TrainingSample(filepath=f"f_{sp}_{r}_{s}.wav", species=f"sp{sp}",
                                              recording_id=f"rec_{sp}_{r}"))
                X.append(centre + rec_shift + rng.normal(0, 1, d))
    return np.array(X), samples


def test_comparison_rows_and_concat():
    X, kept = _data()
    noise = np.random.default_rng(1).normal(size=(len(kept), 8))
    rows, oof, n_classes = ec.run_embedding_comparison({"A": X, "Gurultu": noise}, kept, n_splits=3, n_repeats=2)
    names = {r["yontem"] for r in rows}
    assert names == {"LR_A", "LR_Gurultu", "LR_Birlesik(A+Gurultu)"}
    assert len(rows) == 3 * 2 * 3
    assert n_classes == 4
    # sinyalli gömme gürültüden belirgin iyi olmalı
    ozet = {o["yontem"]: o for o in ec.summarize_comparison(rows, oof, n_classes, n_boot=50)}
    assert ozet["LR_A"]["makro_f1_ort"] > ozet["LR_Gurultu"]["makro_f1_ort"] + 0.3
    assert ozet["LR_A"]["makro_f1_ga_alt"] <= ozet["LR_A"]["makro_f1_havuz"] <= ozet["LR_A"]["makro_f1_ga_ust"]


def test_splits_are_shared_across_methods():
    X, kept = _data()
    rows, _, _ = ec.run_embedding_comparison({"A": X, "B": X + 1}, kept, n_splits=3, n_repeats=2, add_concat=False)
    a = {(r["tekrar"], r["kat"]): (r["n_train"], r["n_test"]) for r in rows if r["yontem"] == "LR_A"}
    b = {(r["tekrar"], r["kat"]): (r["n_train"], r["n_test"]) for r in rows if r["yontem"] == "LR_B"}
    assert a == b and len(a) == 6


def test_oof_covers_each_sample_once_per_repeat():
    X, kept = _data()
    _, oof, _ = ec.run_embedding_comparison({"A": X}, kept, n_splits=3, n_repeats=2, add_concat=False)
    o = oof["LR_A"]
    assert len(o["y_true"]) == 2 * len(kept)
    assert o["tekrar"].count(1) == len(kept)


def test_paired_difference_identical_is_zero_and_better_is_positive():
    X, kept = _data()
    noise = np.random.default_rng(2).normal(size=(len(kept), 16))
    rows, _, _ = ec.run_embedding_comparison({"A": X, "N": noise}, kept, n_splits=3, n_repeats=3, add_concat=False)
    same = ec.paired_difference(rows, "LR_A", "LR_A")
    assert same["fark_ort"] == 0 and same["p_duzeltilmis"] == 1.0
    better = ec.paired_difference(rows, "LR_A", "LR_N")
    assert better["fark_ort"] > 0.3 and better["a_kazanilan_kat"] == better["n_kat"]
    assert better["p_duzeltilmis"] >= better["p_naif"]     # düzeltme p'yi küçültmez


def test_pairwise_table_excludes_reference():
    X, kept = _data()
    rows, _, _ = ec.run_embedding_comparison({"Perch": X, "B": X * 2}, kept, n_splits=3, n_repeats=2, add_concat=False)
    tab = ec.pairwise_table(rows, "LR_Perch")
    assert [t["karsilastirma"] for t in tab] == ["LR_B - LR_Perch"]


def test_bootstrap_ci_perfect_and_bounds():
    y = np.repeat([0, 1], 20)
    rec = np.array([f"r{i // 4}" for i in range(40)])
    p, lo, hi = ec.bootstrap_macro_f1_ci(y, y, rec, 2, n_boot=100)
    assert p == lo == hi == 1.0
    rng = np.random.default_rng(0)
    pred = np.where(rng.random(40) < 0.7, y, 1 - y)
    p, lo, hi = ec.bootstrap_macro_f1_ci(y, pred, rec, 2, n_boot=200)
    assert 0 <= lo <= p <= hi <= 1 and hi > lo


def test_concatenate_validation():
    with pytest.raises(ValueError):
        ec.concatenate_embeddings({"A": np.zeros((3, 2))})
    with pytest.raises(ValueError):
        ec.concatenate_embeddings({"A": np.zeros((3, 2)), "B": np.zeros((4, 2))})
    assert ec.concatenate_embeddings({"A": np.zeros((3, 2)), "B": np.ones((3, 5))}).shape == (3, 7)


def test_external_embeddings_aligned_by_key(tmp_path):
    X, kept = _data(n_species=2, recs_per_species=2, seg_per_rec=2, d=4)
    keys = ec.sample_keys(kept)
    perm = np.random.default_rng(0).permutation(len(kept))
    np.savez(tmp_path / "e.npz", X=X[perm], keys=keys[perm])
    out = ec.load_external_embeddings(str(tmp_path / "e.npz"), kept)
    assert np.allclose(out, X)
    np.savez(tmp_path / "eksik.npz", X=X[:3], keys=keys[:3])
    with pytest.raises(ValueError):
        ec.load_external_embeddings(str(tmp_path / "eksik.npz"), kept)


def test_pool_birdnet_result_ignores_masked_windows():
    emb = np.zeros((1, 4, 3), dtype=np.float32)
    emb[0, 0] = 1.0
    emb[0, 1] = 3.0
    emb[0, 2:] = 99.0
    masked = np.zeros((1, 4, 3), dtype=bool)
    masked[0, 2:] = True
    v = ec.pool_birdnet_result(SimpleNamespace(embeddings=emb, embeddings_masked=masked))
    assert np.allclose(v, 2.0)
    with pytest.raises(ValueError):
        ec.pool_birdnet_result(SimpleNamespace(embeddings=emb, embeddings_masked=np.ones_like(masked)))


def test_extract_birdnet_with_fake_model_pads_short_audio_and_caches(tmp_path, monkeypatch):
    librosa = pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    kept = []
    for i in range(3):
        p = tmp_path / f"s{i}.wav"
        sf.write(str(p), (np.random.default_rng(i).normal(0, 0.1, 48000 // 2)).astype(np.float32), 48000)  # 0,5 sn
        kept.append(TrainingSample(filepath=str(p), species="a", recording_id=f"r{i}"))
    seen = []

    class FakeModel:
        def encode_arrays(self, inp):
            items = [inp] if isinstance(inp, tuple) else list(inp)       # tek girdi ya da liste
            emb = np.zeros((len(items), 2, 5), dtype=np.float32)
            for i, (y, sr) in enumerate(items):
                seen.append(len(y))
                emb[i] = float(len(y))
            return SimpleNamespace(embeddings=emb, embeddings_masked=np.zeros_like(emb, dtype=bool))

    cache = str(tmp_path / "bn.npz")
    X = ec.extract_birdnet_embeddings(kept, model=FakeModel(), cache_path=cache, chunk_size=2)
    assert X.shape == (3, 5) and all(n >= 144000 for n in seen)   # >= 3 sn @ 48 kHz
    X2 = ec.extract_birdnet_embeddings(kept, model=None, cache_path=cache)   # önbellekten; model gerekmez
    assert np.array_equal(X, X2) and len(seen) == 3


def test_birdnet_extraction_batches_calls_and_keeps_order(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    kept = []
    for i in range(5):
        p = tmp_path / f"s{i}.wav"
        sf.write(str(p), np.full(48000 * (i + 1), 0.1, dtype=np.float32), 48000)     # 1..5 sn: uzunluk = i+1
        kept.append(TrainingSample(filepath=str(p), species="a", recording_id=f"r{i}"))
    calls = []

    class Fake:
        def encode_arrays(self, inp):
            items = list(inp)
            calls.append(len(items))
            emb = np.zeros((len(items), 1, 3), dtype=np.float32)
            for i, (y, sr) in enumerate(items):
                emb[i] = len(y) / sr
            return SimpleNamespace(embeddings=emb, embeddings_masked=np.zeros_like(emb, dtype=bool))

    X = ec.extract_birdnet_embeddings(kept, model=Fake(), chunk_size=2)
    assert calls == [2, 2, 1]                                  # 5 örnek -> 3 çağrı (5 değil)
    assert np.allclose(X[:, 0], [3, 3, 3, 4, 5])               # 1-3 sn doldurulur; sıra korunur
