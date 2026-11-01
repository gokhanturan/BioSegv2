import numpy as np
import pytest

from src.ml import birdnet_zero_shot as bzs
from src.ml.dataset_builder import TrainingSample

sf = pytest.importorskip("soundfile")

SPECIES = ["Parus major_Great Tit", "Cyanistes caeruleus_Eurasian Blue Tit", "Corvus corax_Common Raven"]


class FakeResult:
    def __init__(self, names, ids, probs, masked):
        self.species_list, self.species_ids = np.array(names), ids
        self.species_probs, self.species_masked = probs, masked


class FakeModel:
    """Girdinin ortalama genliğine göre 'tür' belirler: genlik ~ k*0.1 -> species k en olası."""
    species_list = SPECIES

    def predict_arrays(self, batch, top_k=5, custom_species_list=None, **kw):
        names = list(custom_species_list) if custom_species_list else list(SPECIES)
        full = SPECIES
        n = len(batch)
        ids = np.zeros((n, 1, top_k), dtype=int)
        probs = np.zeros((n, 1, top_k), dtype=np.float32)
        masked = np.ones((n, 1, top_k), dtype=bool)
        for i, (y, _) in enumerate(batch):
            k = int(round(float(np.abs(y).max()) / 0.1)) - 1
            scores = {nm: (0.9 if full.index(nm) == k else 0.1) for nm in names}
            order = sorted(names, key=lambda nm: -scores[nm])[:top_k]
            for j, nm in enumerate(order):
                ids[i, 0, j], probs[i, 0, j], masked[i, 0, j] = full.index(nm), scores[nm], False
        return FakeResult(full, ids, probs, masked)


def _samples(tmp_path, species_seq):
    out = []
    for i, (sp, amp) in enumerate(species_seq):
        p = tmp_path / f"{i}.wav"
        y = amp * np.sin(2 * np.pi * 440 * np.arange(48000 * 3) / 48000)
        sf.write(str(p), y, 48000)
        out.append(TrainingSample(str(p), sp, f"r{i // 2}"))
    return out


def test_label_scientific_and_match():
    assert bzs.label_scientific("Parus major_Great Tit") == "parus major"
    m = bzs.match_classes(["Parus major", "Bilinmeyen tur"], SPECIES)
    assert m["Parus major"] == "Parus major_Great Tit" and m["Bilinmeyen tur"] is None


def test_zero_shot_predictions_restricted_and_open(tmp_path):
    # sınıf sırası: Cyanistes (0), Parus (1), Yok (2, BirdNET'te karşılığı yok)
    classes = ["Cyanistes caeruleus", "Parus major", "Yok yok"]
    kept = _samples(tmp_path, [("Parus major", 0.1), ("Cyanistes caeruleus", 0.2), ("Yok yok", 0.3)])
    kis, aci, match = bzs.zero_shot_predictions(kept, classes, model=FakeModel())
    assert match["Yok yok"] is None
    assert kis.tolist() == [1, 0, 0]       # 3. segment için en olası tür (Corvus) kısıtlı listede yok -> eldeki en iyi sınıf
    assert aci.tolist() == [1, 0, -1]      # açık kipte Corvus veri setinde yok -> -1


def test_fold_rows_use_group_splits_and_count_missing_as_wrong(tmp_path):
    seq = [(sp, 0.1) for sp in ["A"] * 6 + ["B"] * 6]
    kept = []
    for i, (sp, _) in enumerate(seq):
        kept.append(TrainingSample(f"f{i}", sp, f"rec_{sp}_{i // 2}"))
    pred = np.array([0] * 6 + [1] * 6)
    rows = bzs.fold_rows(pred, kept, "x", n_splits=3, n_repeats=2)
    assert len(rows) == 6 and all(r["dogruluk"] == 1.0 for r in rows)
    rows2 = bzs.fold_rows(np.full(12, -1), kept, "y", n_splits=3, n_repeats=1)
    assert all(r["dogruluk"] == 0.0 and r["makro_f1"] == 0.0 for r in rows2)
