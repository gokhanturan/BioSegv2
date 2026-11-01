"""Unit tests for src.detection.calibration."""

import os
import tempfile
import shutil
import unittest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.detection.calibration import (
    append_record, load_records, compute_calibration, MIN_SAMPLES_FOR_RELIABLE_FIT,
)


class TestCalibration(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.tmpdir, "calibration_log.csv")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_load_records_missing_file_returns_empty(self):
        self.assertEqual(load_records(self.log_path), [])

    def test_append_and_load_round_trip(self):
        append_record("Turdus merula", "Perch2", 87.5, True, log_path=self.log_path)
        append_record("Sitta krueperi", "BirdNET", 45.0, False, log_path=self.log_path)

        records = load_records(self.log_path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].species, "Turdus merula")
        self.assertEqual(records[0].source, "Perch2")
        self.assertAlmostEqual(records[0].confidence, 87.5)
        self.assertTrue(records[0].is_correct)
        self.assertFalse(records[1].is_correct)

    def test_compute_calibration_no_records_returns_none(self):
        self.assertIsNone(compute_calibration(log_path=self.log_path))

    def test_compute_calibration_filters_by_source(self):
        append_record("A", "Perch2", 90, True, log_path=self.log_path)
        append_record("B", "BirdNET", 20, False, log_path=self.log_path)

        perch_result = compute_calibration(source="Perch2", log_path=self.log_path)
        birdnet_result = compute_calibration(source="BirdNET", log_path=self.log_path)

        self.assertEqual(perch_result.n_samples, 1)
        self.assertEqual(birdnet_result.n_samples, 1)

    def test_calibration_curve_is_monotonically_increasing_on_realistic_data(self):
        rng = np.random.default_rng(42)
        for _ in range(60):
            conf = rng.uniform(0, 100)
            is_correct = rng.random() < (conf / 100.0)
            append_record("TestSpecies", "Perch2", conf, bool(is_correct), log_path=self.log_path)

        result = compute_calibration(source="Perch2", log_path=self.log_path)
        self.assertTrue(result.reliable)
        p_low = result.predict_probability(10)
        p_high = result.predict_probability(90)
        self.assertGreater(p_high, p_low)

    def test_below_min_samples_marked_unreliable(self):
        for _ in range(5):
            append_record("A", "Perch2", 80, True, log_path=self.log_path)

        result = compute_calibration(source="Perch2", log_path=self.log_path)
        self.assertLess(result.n_samples, MIN_SAMPLES_FOR_RELIABLE_FIT)
        self.assertFalse(result.reliable)

    def test_all_same_label_does_not_crash(self):
        """Tum ornekler 'dogru' isaretlenirse (celiskili veri) cokmemeli."""
        for _ in range(10):
            append_record("A", "Perch2", 80, True, log_path=self.log_path)

        result = compute_calibration(source="Perch2", log_path=self.log_path)
        self.assertIsNotNone(result)
        prob = result.predict_probability(50)
        self.assertGreater(prob, 0.5)

    def test_corrupt_row_is_skipped_gracefully(self):
        with open(self.log_path, "w", encoding="utf-8-sig") as f:
            f.write("timestamp,source,species,confidence,is_correct\n")
            f.write("2026-01-01,Perch2,Turdus merula,not_a_number,1\n")  # bozuk satir
            f.write("2026-01-01,Perch2,Turdus merula,90,1\n")  # gecerli satir

        records = load_records(self.log_path)
        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
