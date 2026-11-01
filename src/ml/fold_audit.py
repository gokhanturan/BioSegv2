"""
src/ml/fold_audit.py
====================
Çapraz doğrulama katlarının tür desteği denetimi. Her (tekrar, kat) için her türün eğitim/doğrulama/test
segment ve kayıt sayısını çıkarır; testte bulunup eğitimde HİÇ bulunmayan türleri işaretler (model o türü
öğrenemez, o test segmentleri kaçınılmaz olarak yanlış sınıflanır).

Kullanım (proje klasöründe):  python -m src.ml.fold_audit [--kat 5] [--tekrar 2]
Çıktı: kat_denetimi.csv (tür × kat ayrıntısı). Eğitim yapılmaz, ses/embedding okunmaz.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .train_classifier import group_kfold_splits


def audit_folds(samples: list, n_splits: int = 5, n_repeats: int = 2, base_seed: int = 42) -> dict:
    """{'satirlar': [...], 'katlar': [...], 'turler': {...}}. Bölmeler run_cross_validation ile aynı tohumları kullanır."""
    species = sorted({s.species for s in samples})
    rows: List[dict] = []
    folds: List[dict] = []
    for rep in range(n_repeats):
        splits = group_kfold_splits(samples, n_splits=n_splits, random_state=base_seed + rep)
        for fold_i, sp in enumerate(splits):
            n_test = len(sp.test_idx)
            lost = 0
            lost_species = []
            for name in species:
                cell = {}
                for part, idx in (("egitim", sp.train_idx), ("dogrulama", sp.val_idx), ("test", sp.test_idx)):
                    sel = [samples[i] for i in idx if samples[i].species == name]
                    cell[part + "_segment"] = len(sel)
                    cell[part + "_kayit"] = len({s.recording_id for s in sel})
                durum = ""
                if cell["test_segment"] > 0 and cell["egitim_segment"] == 0:
                    durum = "EĞİTİMDE YOK"
                    lost += cell["test_segment"]
                    lost_species.append(name)
                elif cell["test_segment"] > 0 and cell["egitim_kayit"] == 1:
                    durum = "eğitimde tek kayıt"
                rows.append({"tekrar": rep + 1, "kat": fold_i + 1, "tur": name, **cell, "durum": durum})
            folds.append({"tekrar": rep + 1, "kat": fold_i + 1, "n_test": n_test, "kaybolan_test_segment": lost,
                          "kaybolan_oran": (lost / n_test) if n_test else 0.0, "kaybolan_turler": lost_species,
                          "dogrulama_yok": len(sp.val_idx) == 0})
    per_species: Dict[str, dict] = defaultdict(lambda: {"kayit": 0, "etkilenen_kat": 0})
    for name in species:
        per_species[name]["kayit"] = len({s.recording_id for s in samples if s.species == name})
    for r in rows:
        if r["durum"] == "EĞİTİMDE YOK":
            per_species[r["tur"]]["etkilenen_kat"] += 1
    return {"satirlar": rows, "katlar": folds, "turler": dict(per_species), "n_kat": len(folds)}


def summarize(res: dict) -> str:
    folds = res["katlar"]
    affected = [f for f in folds if f["kaybolan_test_segment"] > 0]
    mean_lost = float(np.mean([f["kaybolan_oran"] for f in folds])) if folds else 0.0
    lines = [f"Kat sayısı: {len(folds)} · eğitimde olmayan tür içeren kat: {len(affected)}",
             f"Test segmentlerinin ortalama %{mean_lost * 100:.1f}'i, eğitimde hiç görülmeyen türlerden geliyor "
             f"(bu oran doğruluğun kaçınılmaz üst sınırını %{(1 - mean_lost) * 100:.1f}'e düşürür).",
             "Tür başına kayıt sayısı ve etkilenen kat:"]
    for name, v in sorted(res["turler"].items(), key=lambda kv: kv[1]["kayit"]):
        lines.append(f"  {name:<32} kayıt={v['kayit']:<3} eğitimde yok olan kat={v['etkilenen_kat']}/{len(folds)}")
    return "\n".join(lines)


def main():
    from .dataset_builder import build_training_samples
    ap = argparse.ArgumentParser()
    ap.add_argument("--kat", type=int, default=5)
    ap.add_argument("--tekrar", type=int, default=2)
    ap.add_argument("--min-segment", type=int, default=30)
    ap.add_argument("--min-kayit", type=int, default=2)
    a = ap.parse_args()
    samples, _excl, _raw = build_training_samples(min_samples_per_class=a.min_segment,
                                                  min_recordings_per_class=a.min_kayit)
    res = audit_folds(samples, n_splits=a.kat, n_repeats=a.tekrar)
    print(f"Örnek: {len(samples)} segment, {len({s.recording_id for s in samples})} kayıt, "
          f"{len({s.species for s in samples})} tür")
    print(summarize(res))
    rows = res["satirlar"]
    with open("kat_denetimi.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)
    print("Yazıldı: kat_denetimi.csv")


if __name__ == "__main__":
    main()
