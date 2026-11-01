"""richness_validation testleri (TensorFlow gerektirmez; 'lr' yöntemi ve elle hesaplı örnekler)."""
import numpy as np
import pytest

from src.ml import richness_validation as RV
from src.ml.dataset_builder import TrainingSample


def test_point_incidence_and_compare_hand_computed():
    # 3 nokta, 3 tür. uzman: p1:{0,1} p2:{1} p3:{2}
    pts = np.array(["p1", "p1", "p2", "p3"], dtype=object)
    e = RV.point_incidence(np.array([0, 1, 1, 2]), pts, 3)
    assert e.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]] or e.astype(int).tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    # model: segment 2 yanlış (1 -> 2): p2:{2}; segment 4 tespit yok
    m = RV.point_incidence(np.array([0, 1, 2, -1]), pts, 3)
    r = RV.compare_incidence(e, m)
    # TP: (p1,0),(p1,1) = 2; FP: (p2,2) = 1; FN: (p2,1),(p3,2) = 2
    assert (r["sahte_tur_nokta_cifti"], r["kacirilan_tur_nokta_cifti"]) == (1, 2)
    assert r["var_yok_kesinlik"] == pytest.approx(2 / 3)
    assert r["var_yok_duyarlilik"] == pytest.approx(2 / 4)
    assert r["var_yok_jaccard"] == pytest.approx(2 / 5)
    assert r["tur_uzman"] == 3 and r["tur_model"] == 3


def test_perfect_prediction_gives_identical_richness():
    e = np.random.default_rng(0).random((8, 6)) < 0.4
    r = RV.compare_incidence(e, e.copy())
    assert r["var_yok_jaccard"] == 1.0 and r["nokta_basina_tur_mutlak_fark_ort"] == 0.0
    assert r["tur_uzman"] == r["tur_model"]


def _synthetic(n_species=4, rec_per_species=6, seg_per_rec=3, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 4, size=(n_species, dim))
    samples, X, point = [], [], {}
    for s in range(n_species):
        for r in range(rec_per_species):
            rid = f"rec{s}_{r}"
            point[rid] = f"P{(s + r) % 5}"
            for k in range(seg_per_rec):
                samples.append(TrainingSample(filepath=f"f{s}_{r}_{k}.wav", species=f"sp{s}", recording_id=rid))
                X.append(centers[s] + rng.normal(0, 1, dim))
    return np.array(X), samples, point


def test_out_of_fold_lr_and_evaluation_pipeline():
    X, kept, point = _synthetic()
    classes, preds = RV.out_of_fold_predictions(X, kept, method="lr", n_splits=3, n_repeats=2)
    assert len(preds) == 2 and all((p >= 0).all() for p, _ in preds)   # her segment bir test katında
    rows = RV.evaluate_richness_agreement(kept, point, classes, preds, thresholds=(0.0, 0.99))
    assert len(rows) == 4
    hi = [r for r in rows if r["guven_esigi"] == 0.99]
    lo = [r for r in rows if r["guven_esigi"] == 0.0]
    assert all(r["tespit_orani"] <= l["tespit_orani"] for r, l in zip(hi, lo))
    assert all(r["var_yok_jaccard"] > 0.7 for r in lo)           # ayrışabilir sentetik veri
    s = RV.summarize_by_threshold(rows)
    assert [x["guven_esigi"] for x in s] == [0.0, 0.99] and s[0]["n_tekrar"] == 2
    t = RV.per_point_table(kept, point, classes, preds[0])
    assert {"nokta", "tur_uzman", "tur_model"} <= set(t[0])


def test_zeroshot_requires_scores_and_unknown_method():
    X, kept, _ = _synthetic()
    with pytest.raises(ValueError):
        RV.out_of_fold_predictions(X, kept, method="zeroshot")
    with pytest.raises(ValueError):
        RV.out_of_fold_predictions(X, kept, method="x")
    classes = sorted({s.species for s in kept})
    y = np.array([classes.index(s.species) for s in kept])
    sc = np.full((len(y), len(classes)), -np.inf); sc[np.arange(len(y)), y] = 0.9
    _, preds = RV.out_of_fold_predictions(X, kept, method="zeroshot", n_repeats=1, zero_shot_scores=sc)
    assert (preds[0][0] == y).all()
