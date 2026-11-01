"""Unit tests for src.detection.species_merge (Perch2 + BirdNET union logic)."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.species_merge import merge_across_sources, UnifiedDetection


class TestSpeciesMerge(unittest.TestCase):

    def test_cross_validation_when_both_models_agree(self):
        """Overlapping same-species detections from both models should be flagged cross-validated."""
        perch = [(0.0, 5.0, "Aegithalos caudatus", 100.0)]
        birdnet = [(2.0, 6.0, "Aegithalos caudatus", 77.0)]

        rows = merge_across_sources(perch, birdnet)

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].is_cross_validated)
        self.assertTrue(rows[0].has_perch)
        self.assertTrue(rows[0].has_birdnet)

    def test_birdnet_only_detection_is_not_lost(self):
        """A region only BirdNET detects (Perch missed it) must still appear as its own row."""
        perch = [(0.0, 5.0, "Aegithalos caudatus", 100.0)]
        birdnet = [
            (2.0, 6.0, "Aegithalos caudatus", 77.0),
            (60.0, 63.0, "Sitta krueperi", 88.0),  # non-overlapping, Perch missed this
        ]

        rows = merge_across_sources(perch, birdnet)

        self.assertEqual(len(rows), 2)
        missed_row = [r for r in rows if r.start_s == 60.0][0]
        self.assertFalse(missed_row.has_perch)
        self.assertTrue(missed_row.has_birdnet)
        self.assertEqual(missed_row.birdnet_species, "Sitta krueperi")

    def test_perch_only_detection_is_not_lost(self):
        """A region only Perch detects (BirdNET missed it) must still appear as its own row."""
        perch = [(0.0, 5.0, "Parus major", 100.0)]
        birdnet = []

        rows = merge_across_sources(perch, birdnet)

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].has_perch)
        self.assertFalse(rows[0].has_birdnet)
        self.assertFalse(rows[0].is_cross_validated)

    def test_non_overlapping_different_species_are_separate_rows(self):
        """Different species at different times should never be merged into one row."""
        perch = [(0.0, 5.0, "Parus major", 100.0)]
        birdnet = [(50.0, 55.0, "Corvus corax", 90.0)]

        rows = merge_across_sources(perch, birdnet)

        self.assertEqual(len(rows), 2)

    def test_fsd50k_non_species_category_flagged(self):
        """FSD50K-style category labels (e.g. 'Music') must be flagged as non-species."""
        perch = [(0.0, 5.0, "Music", 90.0)]
        birdnet = []

        rows = merge_across_sources(perch, birdnet)

        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0].perch_is_species)

    def test_best_species_for_labeling_priority(self):
        """Cross-validated species should be preferred; otherwise Perch (if real species) then BirdNET."""
        # Cross-validated case
        rows = merge_across_sources(
            [(0.0, 5.0, "Turdus merula", 100.0)],
            [(1.0, 6.0, "Turdus merula", 80.0)],
        )
        self.assertEqual(rows[0].best_species_for_labeling, "Turdus merula")

        # Perch non-species + BirdNET real species -> BirdNET should win
        rows = merge_across_sources(
            [(0.0, 5.0, "Music", 90.0)],
            [(1.0, 6.0, "Corvus corax", 80.0)],
        )
        # These don't overlap in species so may end up as separate rows;
        # verify at least one row correctly prioritizes a real species.
        species_suggestions = [r.best_species_for_labeling for r in rows if r.best_species_for_labeling]
        self.assertIn("Corvus corax", species_suggestions)

    def test_empty_inputs_produce_no_rows(self):
        rows = merge_across_sources([], [])
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
