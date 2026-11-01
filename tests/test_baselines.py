"""baselines.py testleri (TensorFlow gerektirmez; sentetik veri)."""
import numpy as np
import pytest

from src.ml import baselines
from src.ml.dataset_builder import TrainingSample


def _synthetic(n_species=4, rec_per_species=6, seg_per_rec=3, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 4, size=(n_species, dim))
    samples, X = [], []
    for s in range(n_species):
        for r in range(rec_per_species):
            for k in range(seg_per_rec):
                samples.append(TrainingSample(filepath=f"f{s}_{r}_{k}.wav", species=f"sp{s}",
                                              recording_id=f"rec{s}_{r}"))
                X.append(centers[s] + rng.normal(0, 1, dim))
    return np.array(X), samples


def test_logreg_baseline_beats_chance_and_rows_complete():
    X, kept = _synthetic()
    rows = baselines.run_baselines(X, kept, mfcc=X.copy(), n_splits=3, n_repeats=2)
    assert len(rows) == 2 * 3 * 2 * 2          # tekrar x kat x (2 öznitelik) x (2 ağırlık)
    assert {r["yontem"] for r in rows} == {"LR_BirdNET", "LR_MFCC"}
    assert all(r["dogruluk"] > 0.8 for r in rows)
    assert all(0 <= r["makro_f1"] <= 1 for r in rows)


def test_zero_shot_scores_used_and_shape_checked():
    X, kept = _synthetic()
    classes = sorted({s.species for s in kept})
    y = np.array([classes.index(s.species) for s in kept])
    perfect = np.full((len(y), len(classes)), -np.inf)
    perfect[np.arange(len(y)), y] = 1.0
    rows = baselines.run_baselines(X, kept, zero_shot_scores=perfect, n_splits=3, n_repeats=1,
                                   class_weight_options=(False,))
    zs = [r for r in rows if r["yontem"] == "SifirAtis_Perch"]
    assert len(zs) == 3 and all(r["dogruluk"] == pytest.approx(1.0) for r in zs)
    with pytest.raises(ValueError):
        baselines.run_baselines(X, kept, zero_shot_scores=perfect[:, :2], n_splits=3, n_repeats=1)


def test_same_splits_as_cross_validation():
    """Temel çizgi satırları, run_cross_validation ile aynı n_test dizisini kullanmalı."""
    from src.ml.train_classifier import group_kfold_splits
    X, kept = _synthetic()
    rows = baselines.run_baselines(X, kept, n_splits=3, n_repeats=1, class_weight_options=(False,))
    sizes = [len(sp.test_idx) for sp in group_kfold_splits(kept, n_splits=3, random_state=42)]
    assert [r["n_test"] for r in rows] == sizes


def test_summarize():
    rows = [{"yontem": "A", "sinif_agirligi": False, "dogruluk": v, "makro_f1": v / 2} for v in (0.5, 0.7)]
    s = baselines.summarize(rows)[0]
    assert s["dogruluk_ort"] == pytest.approx(0.6) and s["n_kat"] == 2
    assert s["dogruluk_ss"] == pytest.approx(np.std([0.5, 0.7], ddof=1))


def test_coverage():
    assert baselines.zero_shot_coverage(["A b", "C d"], ["a b", "x y"]) == 0.5


def test_mfcc_features_and_cache(tmp_path):
    pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    sr = 32000
    t = np.arange(sr) / sr
    kept = []
    for i, f in enumerate((1000, 3000)):
        p = tmp_path / f"w{i}.wav"
        sf.write(p, 0.3 * np.sin(2 * np.pi * f * t), sr)
        kept.append(TrainingSample(filepath=str(p), species=f"s{i}", recording_id=f"r{i}"))
    cache = str(tmp_path / "mfcc.npz")
    A = baselines.load_or_extract_mfcc(kept, cache)
    assert A.shape == (2, 4 * baselines.N_MFCC) and np.isfinite(A).all()
    assert not np.allclose(A[0], A[1])
    B = baselines.load_or_extract_mfcc(kept, cache)   # önbellekten
    assert np.array_equal(A, B)


def test_paired_comparison_nadeau_bengio():
    rng = np.random.default_rng(1)
    mlp, base = [], []
    for rep in (1, 2, 3):
        for kat in range(1, 6):
            b = 0.6 + rng.normal(0, 0.02)
            mlp.append({"tekrar": rep, "kat": kat, "sinif_agirligi": True, "makro_f1": b + 0.08 + rng.normal(0, 0.01),
                        "dogruluk": 0.8, "n_train": 500, "n_val": 150, "n_test": 160})
            base.append({"tekrar": rep, "kat": kat, "yontem": "LR_BirdNET", "sinif_agirligi": True,
                         "makro_f1": b, "dogruluk": 0.8, "n_test": 160})
    r = baselines.paired_comparison(mlp, base, "LR_BirdNET", True)
    assert r["fark_ort"] == pytest.approx(0.08, abs=0.02) and r["mlp_kazanilan_kat"] == 15
    assert r["p_duzeltilmis"] > r["p_naif"]                   # düzeltme p'yi büyütür
    eq = baselines.paired_comparison(mlp, base, "LR_BirdNET", True, metric="dogruluk")
    assert eq["fark_ort"] == 0 and eq["p_duzeltilmis"] == 1.0
    with pytest.raises(ValueError):
        baselines.paired_comparison(mlp[:2], base[:2], "LR_BirdNET", True)
