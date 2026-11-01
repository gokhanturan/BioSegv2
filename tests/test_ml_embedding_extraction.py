"""src.ml.embedding_extraction için birim testleri (Perch 2.0 embedding çıkarma)."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.ml.dataset_builder import TrainingSample
from src.ml.embedding_extraction import extract_embeddings_for_samples


class TestExtractEmbeddingsForSamples(unittest.TestCase):

    def setUp(self):
        self.samples = [
            TrainingSample(filepath="a.wav", species="Turdus merula", recording_id="RECA"),
            TrainingSample(filepath="b.wav", species="Lullula arborea", recording_id="RECB"),
        ]

    def test_calls_extract_embedding_for_file_per_sample(self):
        fake_emb = np.ones(1536)
        with patch("src.ml.embedding_extraction.extract_embedding_for_file", return_value=fake_emb) as mock_fn:
            X, kept, n_fail, err = extract_embeddings_for_samples(
                classifier=object(), samples=self.samples,
            )
        self.assertEqual(mock_fn.call_count, 2)
        self.assertEqual(X.shape, (2, 1536))
        self.assertEqual(len(kept), 2)
        self.assertEqual(n_fail, 0)
        self.assertIsNone(err)

    def test_failures_are_counted_and_reported(self):
        def fail_second(classifier, filepath):
            if "b.wav" in filepath:
                raise RuntimeError("simüle edilmiş hata")
            return np.ones(1536)

        with patch("src.ml.embedding_extraction.extract_embedding_for_file", side_effect=fail_second):
            X, kept, n_fail, err = extract_embeddings_for_samples(
                classifier=object(), samples=self.samples,
            )
        self.assertEqual(n_fail, 1)
        self.assertIn("simüle edilmiş hata", err)
        self.assertEqual(len(kept), 1)

    def test_all_failures_returns_empty_result_without_crashing(self):
        with patch("src.ml.embedding_extraction.extract_embedding_for_file",
                   side_effect=RuntimeError("model yüklenemedi")):
            X, kept, n_fail, err = extract_embeddings_for_samples(
                classifier=object(), samples=self.samples,
            )
        self.assertEqual(len(kept), 0)
        self.assertEqual(n_fail, 2)
        self.assertIn("model yüklenemedi", err)

    def test_progress_callback_invoked_for_each_sample(self):
        calls = []

        def on_progress(current, total):
            calls.append((current, total))

        with patch("src.ml.embedding_extraction.extract_embedding_for_file", return_value=np.ones(1536)):
            extract_embeddings_for_samples(
                classifier=object(), samples=self.samples, progress_callback=on_progress,
            )
        self.assertEqual(calls, [(1, 2), (2, 2)])


if __name__ == "__main__":
    unittest.main()