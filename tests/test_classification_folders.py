"""src.data.classification_folders için birim testleri."""

import csv
import os
import shutil
import tempfile
import unittest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.classification_folders import build_classification_folders, _sanitize_folder_name


class TestClassificationFolders(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.segments_dir = os.path.join(self.tmpdir, "segments")
        os.makedirs(self.segments_dir)
        self.csv_path = os.path.join(self.tmpdir, "bird_sound_dataset.csv")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_fake_audio(self, filename: str, content: bytes = b"FAKE"):
        with open(os.path.join(self.segments_dir, filename), "wb") as f:
            f.write(content)

    def _write_csv(self, rows):
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Filename", "Scientific_Name"])
            for row in rows:
                writer.writerow(row)

    def test_copies_files_into_species_folders(self):
        self._write_fake_audio("a.wav")
        self._write_fake_audio("b.wav")
        self._write_csv([["a.wav", "Turdus merula"], ["b.wav", "Lullula arborea"]])

        result = build_classification_folders(csv_path=self.csv_path, segments_dir=self.segments_dir)

        self.assertEqual(result.species_folders_created, 2)
        self.assertEqual(result.files_copied, 2)
        self.assertTrue(os.path.exists(os.path.join(self.segments_dir, "classification", "Turdus merula", "a.wav")))
        self.assertTrue(os.path.exists(os.path.join(self.segments_dir, "classification", "Lullula arborea", "b.wav")))

    def test_does_not_delete_or_move_original_segment_files(self):
        self._write_fake_audio("a.wav")
        self._write_csv([["a.wav", "Turdus merula"]])

        build_classification_folders(csv_path=self.csv_path, segments_dir=self.segments_dir)

        self.assertTrue(os.path.exists(os.path.join(self.segments_dir, "a.wav")))

    def test_repeated_run_skips_unchanged_files(self):
        self._write_fake_audio("a.wav")
        self._write_csv([["a.wav", "Turdus merula"]])

        first = build_classification_folders(csv_path=self.csv_path, segments_dir=self.segments_dir)
        second = build_classification_folders(csv_path=self.csv_path, segments_dir=self.segments_dir)

        self.assertEqual(first.files_copied, 1)
        self.assertEqual(second.files_copied, 0)
        self.assertEqual(second.files_already_present, 1)

    def test_skips_rows_with_missing_scientific_name(self):
        self._write_fake_audio("a.wav")
        self._write_csv([["a.wav", ""]])

        result = build_classification_folders(csv_path=self.csv_path, segments_dir=self.segments_dir)

        self.assertEqual(result.skipped_missing_scientific_name, 1)
        self.assertEqual(result.files_copied, 0)

    def test_skips_rows_with_missing_source_audio_file(self):
        # CSV'de var ama segments/ klasöründe dosya YOK
        self._write_csv([["missing.wav", "Turdus merula"]])

        result = build_classification_folders(csv_path=self.csv_path, segments_dir=self.segments_dir)

        self.assertEqual(result.skipped_missing_source_file, 1)
        self.assertEqual(result.files_copied, 0)

    def test_missing_csv_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            build_classification_folders(
                csv_path=os.path.join(self.tmpdir, "does_not_exist.csv"),
                segments_dir=self.segments_dir,
            )

    def test_sanitize_folder_name_removes_invalid_characters(self):
        self.assertEqual(_sanitize_folder_name('Turdus: merula?'), 'Turdus merula')
        self.assertEqual(_sanitize_folder_name('   '), 'Unlabeled')

    def test_species_counts_reflect_multiple_records_same_species(self):
        self._write_fake_audio("a.wav")
        self._write_fake_audio("b.wav")
        self._write_csv([["a.wav", "Turdus merula"], ["b.wav", "Turdus merula"]])

        result = build_classification_folders(csv_path=self.csv_path, segments_dir=self.segments_dir)

        self.assertEqual(result.species_counts["Turdus merula"], 2)
        self.assertEqual(result.species_folders_created, 1)


if __name__ == "__main__":
    unittest.main()