"""Unit tests for src.detection.turkish_lookup and perch_classifier helper functions."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.turkish_lookup import lookup, get_turkish_name, display_name
from src.detection.perch_classifier import (
    is_probable_species, generate_windows, merge_detections, WINDOW_SAMPLES
)


class TestTurkishLookup(unittest.TestCase):

    def test_lookup_known_species(self):
        info = lookup("Turdus merula")
        self.assertIsNotNone(info)
        self.assertEqual(info["turkish_name"], "Karatavuk")

    def test_lookup_case_insensitive(self):
        info = lookup("TURDUS MERULA")
        self.assertIsNotNone(info)

    def test_lookup_unknown_species_returns_none(self):
        info = lookup("Nonexistus speciesus")
        self.assertIsNone(info)

    def test_get_turkish_name_unknown_returns_none_not_error(self):
        # Must not raise; must not fabricate a name
        result = get_turkish_name("Totally Fake Species")
        self.assertIsNone(result)

    def test_display_name_falls_back_to_scientific_only(self):
        name = display_name("Totally Fake Species")
        self.assertEqual(name, "Totally Fake Species")

    def test_display_name_includes_turkish_when_known(self):
        name = display_name("Turdus merula")
        self.assertIn("Karatavuk", name)
        self.assertIn("Turdus merula", name)


class TestPerchHelpers(unittest.TestCase):

    def test_is_probable_species_true_for_binomial(self):
        self.assertTrue(is_probable_species("Aegithalos caudatus"))

    def test_is_probable_species_false_for_fsd50k_category(self):
        self.assertFalse(is_probable_species("Domestic_sounds_and_home_sounds"))
        self.assertFalse(is_probable_species("Music"))

    def test_generate_windows_short_audio_produces_one_padded_window(self):
        import numpy as np
        sr = 32000
        y = np.random.randn(int(sr * 1.7)).astype(np.float32)  # shorter than 5s window

        windows = generate_windows(y, sr, hop_seconds=2.5)

        self.assertEqual(len(windows), 1)
        _, _, chunk = windows[0]
        self.assertEqual(chunk.shape[0], WINDOW_SAMPLES)

    def test_generate_windows_long_audio_produces_multiple_overlapping_windows(self):
        import numpy as np
        sr = 32000
        y = np.random.randn(sr * 12).astype(np.float32)

        windows = generate_windows(y, sr, hop_seconds=2.5)

        self.assertGreater(len(windows), 1)

    def test_merge_detections_min_window_hits_filters_isolated_false_positive(self):
        detections = [
            (0.0, 5.0, "Lullula arborea", 95.0),
            (2.5, 7.5, "Lullula arborea", 92.0),
            (20.0, 25.0, "IsolatedFalsePositive", 99.0),
        ]

        merged_no_filter = merge_detections(detections, max_gap_seconds=2.5, min_window_hits=1)
        merged_filtered = merge_detections(detections, max_gap_seconds=2.5, min_window_hits=2)

        self.assertEqual(len(merged_no_filter), 2)
        self.assertEqual(len(merged_filtered), 1)
        self.assertEqual(merged_filtered[0][2], "Lullula arborea")


if __name__ == "__main__":
    unittest.main()
