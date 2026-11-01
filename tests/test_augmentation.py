"""augmentation: dönüşümler, önbellek, eğitim-yalnız artırma ve sızıntı denetimi (TensorFlow gerekmez)."""
import numpy as np
import pytest

from src.ml import augmentation as aug
from src.ml.dataset_builder import TrainingSample


def _wave(n=96000, seed=0):
    return (np.random.default_rng(seed).normal(0, 0.1, n)).astype(np.float32)


def test_transforms_keep_length_dtype_and_are_deterministic():
    y = _wave()
    for k in aug.DEFAULT_KINDS:
        a = aug.augment_waveform(y, k, "anahtar", seed=1)
        b = aug.augment_waveform(y, k, "anahtar", seed=1)
        c = aug.augment_waveform(y, k, "baska", seed=1)
        assert a.shape == y.shape and a.dtype == np.float32
        assert np.array_equal(a, b) and not np.array_equal(a, c)
    with pytest.raises(ValueError):
        aug.augment_waveform(y, "pitch", "a")


def test_gain_within_range_and_shift_zero_fills_and_noise_snr():
    y = _wave()
    g = aug.augment_waveform(y, "gain", "k")
    ratio_db = 20 * np.log10(np.sqrt(np.mean(g ** 2)) / np.sqrt(np.mean(y ** 2)))
    assert aug.GAIN_DB_RANGE[0] - 0.1 <= ratio_db <= aug.GAIN_DB_RANGE[1] + 0.1

    s = aug.augment_waveform(y, "shift", "k")
    assert len(s) == len(y) and np.any(s == 0.0)
    k = int(np.argmax(np.abs(s) > 0))                 # kaydırma sonrası ilk dolu örnek
    last = len(y) - int(np.argmax(np.abs(s[::-1]) > 0))
    assert (k > 0 or last < len(y))                   # bir uçta sıfır dolgu var

    n = aug.augment_waveform(y, "noise", "k")
    resid = n - y
    snr = 20 * np.log10(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(resid ** 2)))
    assert aug.SNR_DB_RANGE[0] - 1.0 <= snr <= aug.SNR_DB_RANGE[1] + 1.0

    silent = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(aug.augment_waveform(silent, "noise", "k"), silent)


class _FakeClassifier:
    def __init__(self):
        self.calls = 0

    def predict_window_embedding(self, w):
        self.calls += 1
        return np.array([w.mean(), w.std(), np.abs(w).max()] * 4, dtype=np.float32)


def test_extract_augmented_embeddings_shape_and_cache(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    kept = []
    for i in range(3):
        p = tmp_path / f"s{i}.wav"
        sf.write(str(p), _wave(96000, i), 32000)
        kept.append(TrainingSample(filepath=str(p), species="a", recording_id=f"r{i}"))
    clf = _FakeClassifier()
    cache = str(tmp_path / "c.npz")
    X = aug.extract_augmented_embeddings(kept, kinds=("gain", "noise"), seed=3, classifier=clf, cache_path=cache)
    assert X.shape == (3, 2, 12)
    n_calls = clf.calls
    X2 = aug.extract_augmented_embeddings(kept, kinds=("gain", "noise"), seed=3, classifier=None, cache_path=cache)
    assert np.array_equal(X, X2) and clf.calls == n_calls          # önbellekten, model gerekmedi
    X3 = aug.extract_augmented_embeddings(kept, kinds=("gain",), seed=3, classifier=clf, cache_path=cache)
    assert X3.shape == (3, 1, 12) and clf.calls > n_calls          # dönüşüm listesi değişti -> yeniden


def _data(n_species=4, recs=6, seg=5, d=8, seed=0):
    rng = np.random.default_rng(seed)
    samples, X = [], []
    for sp in range(n_species):
        c = rng.normal(0, 3, d)
        for r in range(recs):
            shift = rng.normal(0, 0.5, d)
            for s in range(seg):
                samples.append(TrainingSample(filepath=f"f{sp}_{r}_{s}.wav", species=f"sp{sp}",
                                              recording_id=f"rec{sp}_{r}"))
                X.append(c + shift + rng.normal(0, 1, d))
    return np.array(X), samples


def test_comparison_uses_aug_only_in_train_and_never_in_test(monkeypatch):
    X, kept = _data()
    K = 2
    X_aug = np.stack([X + 1000.0 * (j + 1) for j in range(K)], axis=1)   # kopyalar açıkça ayırt edilebilir
    seen = []
    import src.ml.baselines as bl
    real = bl.fit_predict_logreg

    def spy(Xtr, ytr, Xte, cw, C=1.0, seed=0):
        seen.append((Xtr.copy(), Xte.copy(), len(ytr)))
        return real(Xtr, ytr, Xte, cw, C=C, seed=seed)

    monkeypatch.setattr(bl, "fit_predict_logreg", spy)
    rows, oof, n_classes = aug.run_augmentation_comparison(X, X_aug, ["gain", "noise"], kept,
                                                           n_splits=3, n_repeats=2)
    assert n_classes == 4
    assert {r["yontem"] for r in rows} == {"LR_Artirmasiz", "LR_Artirmali(tum)",
                                          "LR_Artirmali(gain)", "LR_Artirmali(noise)"}
    assert len(rows) == 2 * 3 * 4
    for Xtr, Xte, n in seen:
        assert Xte.max() < 500                                 # test: yalnızca özgün satırlar
        n_aug = int((Xtr.max(axis=1) >= 500).sum())
        n_orig = int((Xtr.max(axis=1) < 500).sum())
        assert n == n_orig + n_aug and n_aug in (0, n_orig, 2 * n_orig)
    # aynı bölmelerde tüm yöntemlerin test kümesi aynı
    a = {(r["tekrar"], r["kat"]): r["n_test"] for r in rows if r["yontem"] == "LR_Artirmasiz"}
    b = {(r["tekrar"], r["kat"]): r["n_test"] for r in rows if r["yontem"] == "LR_Artirmali(tum)"}
    assert a == b
    # artırmalı eğitim kümesi artırmasızdan (1+K) kat büyük
    r0 = next(r for r in rows if r["yontem"] == "LR_Artirmali(tum)")
    assert r0["n_train_artirmali"] == r0["n_train"] * (1 + K)


def test_comparison_shape_validation():
    X, kept = _data()
    with pytest.raises(ValueError):
        aug.run_augmentation_comparison(X, np.zeros((len(kept), 3, 8)), ["gain", "noise"], kept)
    with pytest.raises(ValueError):
        aug.run_augmentation_comparison(X, np.zeros((len(kept) - 1, 2, 8)), ["gain", "noise"], kept)


def test_summary_pipeline_is_compatible():
    from src.ml import embedding_comparison as ec
    X, kept = _data()
    X_aug = np.stack([X + 0.1, X - 0.1], axis=1)
    rows, oof, n_classes = aug.run_augmentation_comparison(X, X_aug, ["gain", "noise"], kept,
                                                           n_splits=3, n_repeats=3, include_single=False)
    ozet = ec.summarize_comparison(rows, oof, n_classes, n_boot=30)
    assert {o["yontem"] for o in ozet} == {"LR_Artirmasiz", "LR_Artirmali(tum)"}
    tab = ec.pairwise_table(rows, "LR_Artirmasiz")
    assert [t["karsilastirma"] for t in tab] == ["LR_Artirmali(tum) - LR_Artirmasiz"]
