"""stratified_group_folds / group_kfold_splits: tohuma duyarlılık, sızıntısızlık, tür dengesi."""
import numpy as np
import pytest

from src.ml.dataset_builder import TrainingSample
from src.ml.train_classifier import group_kfold_splits, stratified_group_folds


def _samples(n_rec=60, n_species=11, seed=0):
    rng = np.random.default_rng(seed)
    sp = [f"S{i}" for i in range(n_species)]
    out = []
    for r in range(n_rec):
        for s in rng.choice(n_species, size=int(rng.integers(1, min(5, n_species + 1))), replace=False):
            for j in range(int(rng.integers(1, 15))):
                out.append(TrainingSample(f"f{r}_{s}_{j}.wav", sp[s], f"rec{r}"))
    return out


def _test_sets(splits, samples):
    return [tuple(sorted({samples[i].recording_id for i in s.test_idx})) for s in splits]


def test_different_seeds_give_different_splits_with_multispecies_recordings():
    samples = _samples()
    a = _test_sets(group_kfold_splits(samples, 5, 42), samples)
    b = _test_sets(group_kfold_splits(samples, 5, 43), samples)
    c = _test_sets(group_kfold_splits(samples, 5, 44), samples)
    assert a != b and a != c and b != c
    assert sum(x == y for x, y in zip(a, b)) == 0


def test_same_seed_is_deterministic():
    samples = _samples()
    assert _test_sets(group_kfold_splits(samples, 5, 7), samples) == _test_sets(group_kfold_splits(samples, 5, 7), samples)


def test_test_folds_partition_samples_and_recordings_never_shared():
    samples = _samples()
    splits = group_kfold_splits(samples, 5, 3)
    all_test = np.concatenate([s.test_idx for s in splits])
    assert sorted(all_test.tolist()) == list(range(len(samples)))
    for s in splits:
        r = [{samples[i].recording_id for i in idx} for idx in (s.train_idx, s.val_idx, s.test_idx)]
        assert not (r[0] & r[1]) and not (r[0] & r[2]) and not (r[1] & r[2])
        assert len(s.val_idx) > 0 and len(s.test_idx) > 0


def test_species_are_reasonably_balanced_across_folds():
    samples = _samples()
    y = np.array([s.species for s in samples])
    tot = {c: (y == c).sum() for c in set(y)}
    for seed in (1, 2, 3):
        dev = [abs((y[s.test_idx] == c).sum() / tot[c] - 0.2) for s in group_kfold_splits(samples, 5, seed) for c in tot]
        assert np.mean(dev) < 0.05 and max(dev) < 0.2


def test_no_empty_folds_and_errors():
    samples = _samples(n_rec=6, n_species=3)
    assert all(len(s.test_idx) > 0 for s in group_kfold_splits(samples, 6, 0))
    with pytest.raises(ValueError):
        group_kfold_splits(samples, 10, 0)
    with pytest.raises(ValueError):
        stratified_group_folds(["a", "b"], ["g1", "g2"], 3, 0)
