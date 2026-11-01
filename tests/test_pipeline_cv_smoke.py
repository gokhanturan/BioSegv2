"""Boru hattı + çapraz doğrulama bloğu: TensorFlow GEREKTİRMEYEN uçtan uca duman testi (sahte MLP)."""
import csv
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.birdnet_fakes import fake_birdnet

from src.ml.pipeline import run_full_training_pipeline


class _FakeModel:
    """Keras benzeri: predict(X, verbose=0) -> (n, sınıf) olasılık; LR ile."""
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression(max_iter=500).fit(X, y)
        return self

    def predict(self, X, verbose=0):
        out = np.zeros((len(X), self.n_classes))
        out[:, self.clf.classes_] = self.clf.predict_proba(X)
        return out


def _fake_build_and_train(X_train, y_train, X_val, y_val, n_classes, **kw):
    cb = kw.get("progress_callback")
    if cb:
        cb(1, 1, {})
    return _FakeModel(n_classes).fit(X_train, y_train), {"loss": [0.0]}


class TestPipelineWithCV(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cdir = os.path.join(self.tmp, "segments", "classification")
        self.csv = os.path.join(self.tmp, "d.csv")
        self.species = ["Turdus merula", "Lullula arborea", "Parus major"]

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _dataset(self, n_rec=8, n_seg=3):
        rows = []
        for sp in self.species:
            os.makedirs(os.path.join(self.cdir, sp), exist_ok=True)
            for r in range(n_rec):
                for s in range(n_seg):
                    fn = f"{sp}_{r}_{s}.wav"
                    open(os.path.join(self.cdir, sp, fn), "wb").write(b"x")
                    rows.append([fn, sp, f"REC_{sp}_{r}"])
        with open(self.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["Filename", "Scientific_Name", "Original_Recording_Name"])
            w.writerows(rows)

    def _run(self, cv_folds):
        self._dataset()
        rng = np.random.default_rng(0)
        centers = {sp: rng.standard_normal(16) * 3 for sp in self.species}

        def fake_extract(classifier, filepath, hop_seconds=2.5):
            sp = next(s for s in self.species if s in filepath)
            return centers[sp] + rng.standard_normal(16) * 0.3

        stages = []
        with fake_birdnet(fake_extract), \
             patch("src.ml.pipeline.build_and_train_mlp", side_effect=_fake_build_and_train), \
             patch("src.ml.pipeline.save_model"):
            res = run_full_training_pipeline(
                min_samples_per_class=5, test_size=0.25, val_size=0.2, max_epochs=3,
                progress_callback=lambda st, c, t: stages.append(st), csv_path=self.csv,
                classification_dir=self.cdir, save_dir=os.path.join(self.tmp, "out"),
                cv_folds=cv_folds, cv_repeats=2)
        return res, stages

    def test_cv_runs_and_becomes_headline(self):
        res, stages = self._run(cv_folds=3)
        tr = res.training_result
        cv = tr.cv_summary
        self.assertIsNotNone(cv)
        self.assertNotIn("hata", cv)
        self.assertEqual((cv["n_kat"], cv["n_tekrar"], cv["n_deger"]), (3, 2, 6))
        self.assertGreater(cv["dogruluk"][0], 0.9)           # ayrılabilir sentetik veri
        self.assertEqual(set(cv["tur_f1"]), set(self.species))
        self.assertIn("cv", stages)
        self.assertLess(stages.index("cv"), stages.index("training"))
        text = tr.format_summary()
        self.assertTrue(text.startswith("ÇAPRAZ DOĞRULAMA"))
        self.assertIsNotNone(tr.test_predictions)

    def test_cv_is_off_by_default_and_summary_unchanged(self):
        res, stages = self._run(cv_folds=0)
        self.assertIsNone(res.training_result.cv_summary)
        self.assertNotIn("cv", stages)
        self.assertTrue(res.training_result.format_summary().startswith("Test doğruluğu"))


if __name__ == "__main__":
    unittest.main()
