"""src.ml.dataset_builder için birim testleri."""

import csv
import os
import shutil
import tempfile
import unittest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.dataset_builder import build_training_samples


class TestDatasetBuilder(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.segments_dir = os.path.join(self.tmpdir, "segments")
        self.classification_dir = os.path.join(self.segments_dir, "classification")
        self.csv_path = os.path.join(self.tmpdir, "bird_sound_dataset.csv")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _make_species_file(self, species: str, filename: str):
        folder = os.path.join(self.classification_dir, species)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, filename), "wb") as f:
            f.write(b"FAKE")

    def _write_csv(self, rows):
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Filename", "Scientific_Name", "Original_Recording_Name"])
            for row in rows:
                writer.writerow(row)

    def test_excludes_species_below_min_samples_threshold(self):
        for fname in ["a1.wav", "a2.wav", "a3.wav"]:
            self._make_species_file("Turdus merula", fname)
        self._make_species_file("Lullula arborea", "b1.wav")

        self._write_csv([
            ["a1.wav", "Turdus merula", "REC_A"],
            ["a2.wav", "Turdus merula", "REC_A"],
            ["a3.wav", "Turdus merula", "REC_B"],
            ["b1.wav", "Lullula arborea", "REC_C"],
        ])

        samples, excluded, raw_counts = build_training_samples(
            csv_path=self.csv_path, classification_dir=self.classification_dir,
            min_samples_per_class=2,
        )

        self.assertEqual(len(samples), 3)
        self.assertEqual(excluded, ["Lullula arborea"])
        self.assertEqual(raw_counts["Turdus merula"], 3)
        self.assertEqual(raw_counts["Lullula arborea"], 1)

    def test_recording_id_assigned_correctly_for_grouping(self):
        self._make_species_file("Turdus merula", "a1.wav")
        self._make_species_file("Turdus merula", "a2.wav")
        self._write_csv([
            ["a1.wav", "Turdus merula", "REC_A"],
            ["a2.wav", "Turdus merula", "REC_A"],
        ])

        samples, _, _ = build_training_samples(
            csv_path=self.csv_path, classification_dir=self.classification_dir,
            min_samples_per_class=1,
        )

        self.assertTrue(all(s.recording_id == "REC_A" for s in samples))

    def test_skips_rows_missing_source_file(self):
        self._write_csv([["missing.wav", "Turdus merula", "REC_A"]])

        samples, _, _ = build_training_samples(
            csv_path=self.csv_path, classification_dir=self.classification_dir,
            min_samples_per_class=1,
        )

        self.assertEqual(len(samples), 0)

    def test_missing_csv_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            build_training_samples(
                csv_path=os.path.join(self.tmpdir, "nope.csv"),
                classification_dir=self.classification_dir,
            )

    def test_falls_back_to_filename_when_recording_id_missing(self):
        self._make_species_file("Turdus merula", "a1.wav")
        self._write_csv([["a1.wav", "Turdus merula", ""]])

        # Burada sadece recording_id -> dosya adı yedeklemesini izole test
        # ediyoruz; min_recordings_per_class=1 ile yeni (varsayılan 2)
        # kayıt-sayısı şartını bilerek devre dışı bırakıyoruz.
        samples, _, _ = build_training_samples(
            csv_path=self.csv_path, classification_dir=self.classification_dir,
            min_samples_per_class=1, min_recordings_per_class=1,
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].recording_id, "a1.wav")

    def test_excludes_species_with_too_few_distinct_recordings(self):
        """
        KRİTİK: bir tür yeterli segmente (>=min_samples_per_class) sahip
        olsa bile, hepsi TEK bir kayıttan geliyorsa (train/test'e ayrı ayrı
        bölünemez) dışlanmalı — gerçek bir vakada bu eksiklik %0 test
        doğruluğuna yol açmıştı.
        """
        for i in range(15):
            self._make_species_file("Turdus merula", f"a{i}.wav")
        self._write_csv([[f"a{i}.wav", "Turdus merula", "TEK_KAYIT"] for i in range(15)])

        samples, excluded, _ = build_training_samples(
            csv_path=self.csv_path, classification_dir=self.classification_dir,
            min_samples_per_class=10, min_recordings_per_class=2,
        )

        self.assertEqual(len(samples), 0)
        self.assertIn("Turdus merula", excluded)

    def test_bird_only_restriction_disabled_by_default(self):
        """restrict_to_known_bird_species varsayılan olarak KAPALI (False)
        olmalı — mevcut davranışı bozmamak için."""
        self._make_species_file("Sahte Tur", "a1.wav")
        self._make_species_file("Sahte Tur", "a2.wav")
        self._write_csv([
            ["a1.wav", "Sahte Tur", "REC_A"],
            ["a2.wav", "Sahte Tur", "REC_B"],
        ])

        samples, excluded, _ = build_training_samples(
            csv_path=self.csv_path, classification_dir=self.classification_dir,
            min_samples_per_class=1, min_recordings_per_class=2,
        )
        self.assertEqual(len(samples), 2)
        self.assertEqual(excluded, [])

    def test_bird_only_restriction_excludes_non_bird_species(self):
        """
        KRİTİK: `restrict_to_known_bird_species=True` iken, projenin 518
        türlük kuş veritabanında bulunmayan bir isim (örn. yanlışlıkla
        etiketlenmiş bir kurbağa türü) EĞİTİM SETİNDEN dışlanmalı — Perch/
        BirdNET çok-taksonlu modeller olduğu için bu garanti gerekiyor.
        """
        self._make_species_file("Turdus merula", "a1.wav")
        self._make_species_file("Turdus merula", "a2.wav")
        self._make_species_file("Rana temporaria", "b1.wav")  # gerçek bir kuş değil
        self._make_species_file("Rana temporaria", "b2.wav")
        self._write_csv([
            ["a1.wav", "Turdus merula", "REC_A"],
            ["a2.wav", "Turdus merula", "REC_B"],
            ["b1.wav", "Rana temporaria", "REC_C"],
            ["b2.wav", "Rana temporaria", "REC_D"],
        ])

        samples, excluded, _ = build_training_samples(
            csv_path=self.csv_path, classification_dir=self.classification_dir,
            min_samples_per_class=1, min_recordings_per_class=2,
            restrict_to_known_bird_species=True,
        )

        species_kept = {s.species for s in samples}
        self.assertEqual(species_kept, {"Turdus merula"})
        self.assertIn("Rana temporaria", excluded)


if __name__ == "__main__":
    unittest.main()

def test_default_min_segments_is_30_everywhere():
    import inspect
    from src.data import dataset_summary
    from src.ml import cv_runner, pipeline, classification_diagnosis
    from src.ml.dataset_builder import DEFAULT_MIN_SEGMENTS, build_training_samples
    assert DEFAULT_MIN_SEGMENTS == 30 == dataset_summary.MIN_SEGMENT_FOR_TRAINING
    for fn in (build_training_samples, cv_runner.run_evaluation, classification_diagnosis.species_support):
        p = [v for k, v in inspect.signature(fn).parameters.items() if k in ("min_samples_per_class", "min_segment")]
        assert p and p[0].default == 30, fn
    assert inspect.signature(pipeline.run_training_pipeline).parameters["min_samples_per_class"].default == 30 \
        if hasattr(pipeline, "run_training_pipeline") else True
