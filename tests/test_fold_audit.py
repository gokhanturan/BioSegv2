"""Kat tür desteği denetimi."""
import numpy as np

from src.ml.dataset_builder import TrainingSample
from src.ml.fold_audit import audit_folds, summarize
from src.ml.train_classifier import group_kfold_splits


def _samples(rec_per=(2, 3, 6, 8), seg=6):
    out, rid = [], 0
    for si, k in enumerate(rec_per):
        for _ in range(k):
            rid += 1
            out += [TrainingSample(f"s{si}_{rid}_{j}.wav", f"Sp{si}", f"rec{rid}") for j in range(seg)]
    return out


def test_audit_matches_real_splits_and_flags_missing_species():
    samples = _samples()
    res = audit_folds(samples, n_splits=5, n_repeats=2)
    assert res["n_kat"] == 10 and len(res["satirlar"]) == 10 * 4
    for rep in range(2):                                    # satırlar run_cross_validation ile aynı bölmelerden gelir
        splits = group_kfold_splits(samples, n_splits=5, random_state=42 + rep)
        for fold_i, sp in enumerate(splits):
            for r in res["satirlar"]:
                if r["tekrar"] == rep + 1 and r["kat"] == fold_i + 1:
                    n = sum(1 for i in sp.test_idx if samples[i].species == r["tur"])
                    assert r["test_segment"] == n
                    assert (r["durum"] == "EĞİTİMDE YOK") == (r["test_segment"] > 0 and r["egitim_segment"] == 0)
    lost = sum(f["kaybolan_test_segment"] for f in res["katlar"])
    assert lost == sum(r["test_segment"] for r in res["satirlar"] if r["durum"] == "EĞİTİMDE YOK")
    assert "Kat sayısı: 10" in summarize(res)


def test_audit_detects_constructed_problem():
    # iki kayıtlı tür: 2 katlı bölmede bir kayıt test, diğeri eğitimde kalır -> sorun yok; 1 kayıtlı eğitim yok
    samples = _samples((2, 2))
    res = audit_folds(samples, n_splits=2, n_repeats=1)
    assert all(f["kaybolan_oran"] <= 1.0 for f in res["katlar"])
    assert set(res["turler"]) == {"Sp0", "Sp1"} and res["turler"]["Sp0"]["kayit"] == 2
