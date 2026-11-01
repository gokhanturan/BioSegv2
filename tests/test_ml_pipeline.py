"""src.ml.pipeline için birim testleri (uçtan uca eğitim orkestrasyonu)."""

import csv
import os
import shutil
import tempfile
import unittest
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tests.birdnet_fakes import fake_birdnet

from src.ml.pipeline import run_full_training_pipeline, InsufficientDataError


class TestTrainingPipeline(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.segments_dir = os.path.join(self.tmpdir, "segments")
        self.classification_dir = os.path.join(self.segments_dir, "classification")
        self.csv_path = os.path.join(self.tmpdir, "bird_sound_dataset.csv")
        self.save_dir = os.path.join(self.tmpdir, "trained_classifier")
        self.species_list = ["Turdus merula", "Lullula arborea"]

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _build_synthetic_dataset(self, n_recordings=8, segments_per_recording=2):
        rows = []
        for sp in self.species_list:
            folder = os.path.join(self.classification_dir, sp)
            os.makedirs(folder, exist_ok=True)
            for rec_i in range(n_recordings):
                for seg_i in range(segments_per_recording):
                    fname = f"{sp}_{rec_i}_{seg_i}.wav"
                    with open(os.path.join(folder, fname), "wb") as f:
                        f.write(b"FAKE_AUDIO")
                    rows.append([fname, sp, f"REC_{sp}_{rec_i}"])

        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Filename", "Scientific_Name", "Original_Recording_Name"])
            for row in rows:
                writer.writerow(row)

    def _fake_embedding_extractor(self, seed=42, dim=1024, spread=3.0, noise=0.3):
        rng = np.random.default_rng(seed)
        centers = {sp: rng.standard_normal(dim) * spread for sp in self.species_list}

        def _fake_extract(classifier, filepath, hop_seconds=2.5):
            for sp in self.species_list:
                if sp in filepath:
                    return centers[sp] + np.random.randn(dim) * noise
            raise ValueError("bilinmeyen tür")

        return _fake_extract

    def test_full_pipeline_runs_end_to_end_with_mocked_embeddings(self):
        self._build_synthetic_dataset()
        fake_extract = self._fake_embedding_extractor()

        with fake_birdnet(fake_extract):

            progress_stages = []

            def on_progress(stage, current, total):
                progress_stages.append(stage)

            result = run_full_training_pipeline(
                min_samples_per_class=5,
                test_size=0.25, val_size=0.2, max_epochs=25,
                progress_callback=on_progress,
                csv_path=self.csv_path,
                classification_dir=self.classification_dir,
                save_dir=self.save_dir,
            )

        self.assertGreater(result.training_result.test_accuracy, 0.7)
        self.assertEqual(len(result.training_result.class_names), 2)
        self.assertTrue(os.path.exists(os.path.join(result.save_dir, "model.keras")))
        self.assertTrue(os.path.exists(os.path.join(result.save_dir, "classes.json")))
        self.assertIn("samples", progress_stages)
        self.assertIn("embedding", progress_stages)
        self.assertIn("training", progress_stages)
        self.assertIn("done", progress_stages)

    def test_no_data_leakage_across_splits_in_full_pipeline(self):
        """Tam hat üzerinden de grup-farkında bölmenin çalıştığını doğrular."""
        self._build_synthetic_dataset(n_recordings=10, segments_per_recording=3)
        fake_extract = self._fake_embedding_extractor()

        with fake_birdnet(fake_extract):
            result = run_full_training_pipeline(
                min_samples_per_class=5,
                test_size=0.3, val_size=0.2, max_epochs=15,
                csv_path=self.csv_path,
                classification_dir=self.classification_dir,
                save_dir=self.save_dir,
            )

        tr = result.training_result
        # kayıt sayıları, segment sayılarından mantıklı biçimde az
        # olmalı (her kayıttan birden fazla segment var)
        self.assertLessEqual(tr.n_train_recordings, tr.n_train)
        self.assertLessEqual(tr.n_test_recordings, tr.n_test)
        self.assertGreater(tr.n_train_recordings, 0)
        self.assertGreater(tr.n_test_recordings, 0)

    def test_insufficient_data_raises_clear_error(self):
        # Hiçbir dosya/CSV yok
        with self.assertRaises(FileNotFoundError):
            run_full_training_pipeline(
                csv_path=self.csv_path,  # henüz oluşturulmadı
                classification_dir=self.classification_dir,
                save_dir=self.save_dir,
            )

    def test_single_species_raises_insufficient_data_error(self):
        # Sadece 1 tür var - en az 2 tür gerekir
        folder = os.path.join(self.classification_dir, "Turdus merula")
        os.makedirs(folder, exist_ok=True)
        rows = []
        for i in range(10):
            fname = f"a{i}.wav"
            with open(os.path.join(folder, fname), "wb") as f:
                f.write(b"FAKE")
            rows.append([fname, "Turdus merula", f"REC_{i}"])

        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Filename", "Scientific_Name", "Original_Recording_Name"])
            for row in rows:
                writer.writerow(row)

        with self.assertRaises(InsufficientDataError):
            run_full_training_pipeline(
                min_samples_per_class=5,
                csv_path=self.csv_path,
                classification_dir=self.classification_dir,
                save_dir=self.save_dir,
            )

    def test_all_embeddings_failing_raises_clear_actionable_error(self):
        """
        Perch 2.0 modeli yüklenemediği (ya da başka sistemik bir nedenle
        HER dosya için embedding çıkarma başarısız olduğu) durumda,
        akış sessizce boş veriyle devam edip anlaşılmaz bir sklearn
        hatası vermek yerine, AÇIK ve EYLEME dönüştürülebilir bir hata
        fırlatmalı.
        """
        self._build_synthetic_dataset()

        def always_fails(classifier, filepath, hop_seconds=2.5):
            raise RuntimeError("Perch 2.0 modeli yüklenemedi (simüle edilmiş hata)")

        with fake_birdnet(always_fails):
            with self.assertRaises(InsufficientDataError) as ctx:
                run_full_training_pipeline(
                    min_samples_per_class=5,
                    csv_path=self.csv_path,
                    classification_dir=self.classification_dir,
                    save_dir=self.save_dir,
                )

        error_message = str(ctx.exception)
        self.assertIn("embedding çıkarılamadı", error_message.lower())
        self.assertIn("simüle edilmiş hata", error_message)


if __name__ == "__main__":
    unittest.main()