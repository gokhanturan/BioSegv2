"""src.ml.input_format_control için birim testleri (Perch girdi biçimi kontrolü)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.detection.perch_classifier import WINDOW_SAMPLES
from src.ml import input_format_control as ifc
from src.ml.dataset_builder import TrainingSample
from src.ml import embedding_comparison as ec


class _FakeClassifier:
    def __init__(self):
        self.seen = []

    def predict_window_embedding(self, w):
        self.seen.append(w.copy())
        return np.array([w.mean(), np.abs(w).max()])


class TestFillToWindow(unittest.TestCase):
    def setUp(self):
        self.y = np.arange(1, 96001, dtype=np.float32)  # 3 sn @ 32 kHz

    def test_zero_pads_tail(self):
        out = ifc.fill_to_window(self.y, "sifir")
        self.assertEqual(len(out), WINDOW_SAMPLES)
        np.testing.assert_array_equal(out[:96000], self.y)
        self.assertTrue(np.all(out[96000:] == 0))

    def test_tile_repeats_segment(self):
        out = ifc.fill_to_window(self.y, "tekrar")
        self.assertEqual(len(out), WINDOW_SAMPLES)
        np.testing.assert_array_equal(out[:96000], self.y)
        np.testing.assert_array_equal(out[96000:], self.y[:WINDOW_SAMPLES - 96000])
        self.assertTrue(np.all(out != 0))

    def test_long_signal_untouched(self):
        y = np.ones(WINDOW_SAMPLES + 10, dtype=np.float32)
        for m in ifc.KIPLER:
            self.assertIs(ifc.fill_to_window(y, m), y)

    def test_bad_mode_and_empty(self):
        with self.assertRaises(ValueError):
            ifc.fill_to_window(self.y, "baska")
        with self.assertRaises(ValueError):
            ifc.fill_to_window(np.array([]), "sifir")

    def test_embedding_for_mode_feeds_full_window(self):
        clf = _FakeClassifier()
        e0 = ifc.embedding_for_mode(clf, self.y, "sifir")
        e1 = ifc.embedding_for_mode(clf, self.y, "tekrar")
        self.assertTrue(all(len(w) == WINDOW_SAMPLES for w in clf.seen))
        self.assertNotEqual(e0[0], e1[0])


class TestPaddingEffectRows(unittest.TestCase):
    def test_pairs_and_sign(self):
        rng = np.random.default_rng(0)
        rows = []
        base = {"LR_Perch_sifir": 0.80, "LR_Perch_tekrar": 0.85, "LR_BirdNET": 0.90}
        for rep in range(3):
            for fold in range(5):
                noise = rng.normal(0, 0.005)
                for name, v in base.items():
                    rows.append({"yontem": name, "tekrar": rep, "kat": fold, "makro_f1": v + noise * (1 + hash(name) % 2),
                                 "dogruluk": v, "n_test": 20, "n_train": 80})
        out = ifc.padding_effect_rows(rows)
        self.assertEqual([o["karsilastirma"] for o in out],
                         ["Perch tekrar − Perch sıfır", "BirdNET − Perch sıfır", "BirdNET − Perch tekrar"])
        self.assertGreater(out[0]["fark_ort"], 0.04)
        self.assertGreater(out[1]["fark_ort"], out[2]["fark_ort"])

    def test_missing_birdnet_is_skipped(self):
        rows = [{"yontem": n, "tekrar": r, "kat": k, "makro_f1": 0.8 + 0.01 * k, "dogruluk": 0.8,
                 "n_test": 20, "n_train": 80}
                for n in ("LR_Perch_sifir", "LR_Perch_tekrar") for r in range(2) for k in range(3)]
        out = ifc.padding_effect_rows(rows)
        self.assertEqual(len(out), 1)


class TestEndToEndSynthetic(unittest.TestCase):
    def test_run_embedding_comparison_row_names(self):
        rng = np.random.default_rng(1)
        samples, X = [], []
        for c in range(3):
            for r in range(4):
                for j in range(6):
                    samples.append(TrainingSample(filepath=f"s{c}_{r}_{j}.wav", species=f"sp{c}",
                                                  recording_id=f"rec{c}_{r}"))
                    X.append(rng.normal(c, 0.3, 8))
        X = np.array(X)
        rows, _, _ = ec.run_embedding_comparison(
            {"Perch_sifir": X, "Perch_tekrar": X + 0.01, "BirdNET": X}, samples,
            n_splits=2, n_repeats=2, add_concat=False)
        names = {r["yontem"] for r in rows}
        self.assertEqual(names, {"LR_Perch_sifir", "LR_Perch_tekrar", "LR_BirdNET"})
        self.assertEqual(len(ifc.padding_effect_rows(rows)), 3)


if __name__ == "__main__":
    unittest.main()
