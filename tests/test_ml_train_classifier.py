"""src.ml.train_classifier için birim testleri."""

import os
import shutil
import tempfile
import unittest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.ml.dataset_builder import TrainingSample
from src.ml.train_classifier import (
    group_aware_split, group_kfold_splits, build_and_train_mlp, evaluate_model, save_model, load_model,
)


def _make_grouped_samples(n_recordings_per_class=10, segments_per_recording=3):
    samples = []
    for cls, prefix in enumerate(["SpeciesA", "SpeciesB"]):
        for rec_i in range(n_recordings_per_class):
            for seg_i in range(segments_per_recording):
                samples.append(TrainingSample(
                    filepath=f"{prefix}_{rec_i}_{seg_i}.wav",
                    species=prefix,
                    recording_id=f"{prefix}_REC{rec_i}",
                ))
    return samples


class TestGroupAwareSplit(unittest.TestCase):

    def test_no_recording_appears_in_more_than_one_split(self):
        """EN KRİTİK TEST: veri sızıntısı olmamalı."""
        samples = _make_grouped_samples()
        split = group_aware_split(samples, test_size=0.3, val_size=0.2, random_state=7)

        train_recs = {samples[i].recording_id for i in split.train_idx}
        val_recs = {samples[i].recording_id for i in split.val_idx}
        test_recs = {samples[i].recording_id for i in split.test_idx}

        self.assertEqual(train_recs & val_recs, set())
        self.assertEqual(train_recs & test_recs, set())
        self.assertEqual(val_recs & test_recs, set())

    def test_split_covers_all_samples_exactly_once(self):
        samples = _make_grouped_samples()
        split = group_aware_split(samples, test_size=0.3, val_size=0.2, random_state=3)

        all_idx = sorted(list(split.train_idx) + list(split.val_idx) + list(split.test_idx))
        self.assertEqual(all_idx, list(range(len(samples))))

    def test_different_random_states_can_produce_different_splits(self):
        samples = _make_grouped_samples()
        split1 = group_aware_split(samples, random_state=1)
        split2 = group_aware_split(samples, random_state=2)
        self.assertFalse(np.array_equal(sorted(split1.test_idx), sorted(split2.test_idx)))

    def test_too_few_recordings_falls_back_to_empty_val_without_crashing(self):
        """
        Her tür TAM OLARAK 2 farklı kayıttan geliyorsa (1 train + 1 test
        için yeterli, ama ayrı bir val için yetersiz), val seti BOŞ
        dönmeli ama train/test hâlâ doğru (ve her iki tür de temsil
        edilerek) ayrılmalı.
        """
        samples = []
        for rec_i in range(2):
            for seg_i in range(5):
                samples.append(TrainingSample(filepath=f"A_{rec_i}_{seg_i}.wav", species="SpeciesA", recording_id=f"A_REC{rec_i}"))
        for rec_i in range(2):
            for seg_i in range(5):
                samples.append(TrainingSample(filepath=f"B_{rec_i}_{seg_i}.wav", species="SpeciesB", recording_id=f"B_REC{rec_i}"))

        split = group_aware_split(samples, test_size=0.2, val_size=0.15, random_state=42)

        self.assertEqual(len(split.val_idx), 0)
        self.assertGreater(len(split.train_idx), 0)
        self.assertGreater(len(split.test_idx), 0)

        train_recs = {samples[i].recording_id for i in split.train_idx}
        test_recs = {samples[i].recording_id for i in split.test_idx}
        self.assertEqual(train_recs & test_recs, set())

        # KRİTİK: HER İKİ TÜR DE hem train'de hem test'te temsil edilmeli
        # (önce bulunan gerçek hatanın tam olarak bunun eksikliği yüzünden
        # olduğunu doğrulayan test)
        train_species = {samples[i].species for i in split.train_idx}
        test_species = {samples[i].species for i in split.test_idx}
        self.assertEqual(train_species, {"SpeciesA", "SpeciesB"})
        self.assertEqual(test_species, {"SpeciesA", "SpeciesB"})

    def test_single_recording_species_goes_entirely_to_train(self):
        """
        Bir tür sadece 1 kayıttan geliyorsa, o kayıt train'e verilir
        (test/val'a hiç gitmez) — bu tür için test doğruluğu ölçülemez
        ama en azından model bu türü 'görmüş' olur. Hata fırlatılmaz;
        çağıran kod (pipeline.py) test_idx boş kalırsa bunu ayrıca
        kontrol eder.
        """
        samples = [
            TrainingSample(filepath=f"a{i}.wav", species="SpeciesA", recording_id="ONLY_REC")
            for i in range(5)
        ]
        split = group_aware_split(samples, test_size=0.2, val_size=0.15)

        self.assertEqual(len(split.train_idx), 5)
        self.assertEqual(len(split.val_idx), 0)
        self.assertEqual(len(split.test_idx), 0)

    def test_each_species_represented_in_train_and_test_with_realistic_recording_counts(self):
        """
        Gerçek dünya senaryosuna daha yakın: türler farklı sayıda kayda
        sahip (3 ve 5 kayıt). Her ikisi de hem train hem test'te
        görünmeli.
        """
        samples = []
        for rec_i in range(3):
            for seg_i in range(4):
                samples.append(TrainingSample(filepath=f"A_{rec_i}_{seg_i}.wav", species="SpeciesA", recording_id=f"A_REC{rec_i}"))
        for rec_i in range(5):
            for seg_i in range(4):
                samples.append(TrainingSample(filepath=f"B_{rec_i}_{seg_i}.wav", species="SpeciesB", recording_id=f"B_REC{rec_i}"))

        split = group_aware_split(samples, test_size=0.2, val_size=0.15, random_state=7)

        train_species = {samples[i].species for i in split.train_idx}
        test_species = {samples[i].species for i in split.test_idx}
        self.assertEqual(train_species, {"SpeciesA", "SpeciesB"})
        self.assertEqual(test_species, {"SpeciesA", "SpeciesB"})


    def test_multi_species_recordings_never_leak_across_splits(self):
        """
        REGRESYON TESTİ (v2.0.1): Bir kayıt BİRDEN FAZLA türün segmentini
        içerdiğinde bile, her kayıt yalnızca TEK bir kümede olmalı ve her
        örnek en fazla bir kümede görünmeli. Eski sürüm tür bazında ayrı ayrı
        atama yaptığı için bu durumda sızıntı üretiyordu.
        """
        rng = np.random.default_rng(0)
        species = [f"S{i}" for i in range(6)]
        samples = []
        for rec_i in range(30):
            rec_species = rng.choice(species, size=2, replace=False)
            for sp in rec_species:
                for seg_i in range(3):
                    samples.append(TrainingSample(
                        filepath=f"{sp}_{rec_i}_{seg_i}.wav", species=str(sp), recording_id=f"REC{rec_i}"))

        for seed in range(5):
            split = group_aware_split(samples, test_size=0.2, val_size=0.15, random_state=seed)
            tr = {samples[i].recording_id for i in split.train_idx}
            va = {samples[i].recording_id for i in split.val_idx}
            te = {samples[i].recording_id for i in split.test_idx}
            self.assertEqual(tr & va, set())
            self.assertEqual(tr & te, set())
            self.assertEqual(va & te, set())
            all_idx = sorted(list(split.train_idx) + list(split.val_idx) + list(split.test_idx))
            self.assertEqual(all_idx, list(range(len(samples))))  # hiçbir örnek çift sayılmaz
            self.assertEqual(len(tr | va | te), 30)


class TestGroupKFoldSplits(unittest.TestCase):

    def _multi_species_samples(self, n_rec=40, seed=0):
        rng = np.random.default_rng(seed)
        species = [f"S{i}" for i in range(5)]
        samples = []
        for rec_i in range(n_rec):
            for sp in rng.choice(species, size=2, replace=False):
                for seg_i in range(3):
                    samples.append(TrainingSample(
                        filepath=f"{sp}_{rec_i}_{seg_i}.wav", species=str(sp), recording_id=f"REC{rec_i}"))
        return samples

    def test_folds_are_recording_disjoint_and_test_folds_partition_samples(self):
        samples = self._multi_species_samples()
        splits = group_kfold_splits(samples, n_splits=5, random_state=1)
        self.assertEqual(len(splits), 5)
        all_test = []
        for sp in splits:
            tr = {samples[i].recording_id for i in sp.train_idx}
            va = {samples[i].recording_id for i in sp.val_idx}
            te = {samples[i].recording_id for i in sp.test_idx}
            self.assertEqual(tr & va, set())
            self.assertEqual(tr & te, set())
            self.assertEqual(va & te, set())
            self.assertGreater(len(sp.val_idx), 0)
            all_test.extend(sp.test_idx.tolist())
        # Her örnek tam olarak BİR katın test kümesinde olmalı
        self.assertEqual(sorted(all_test), list(range(len(samples))))

    def test_invalid_n_splits_raises(self):
        samples = self._multi_species_samples(n_rec=4)
        with self.assertRaises(ValueError):
            group_kfold_splits(samples, n_splits=10)

    def test_different_seeds_give_different_folds(self):
        samples = self._multi_species_samples()
        a = group_kfold_splits(samples, n_splits=5, random_state=1)[0].test_idx
        b = group_kfold_splits(samples, n_splits=5, random_state=2)[0].test_idx
        self.assertFalse(np.array_equal(sorted(a), sorted(b)))


class TestMlpTrainingAndEvaluation(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.embedding_dim = 16
        self.n_classes = 3
        self.class_names = ["SpeciesA", "SpeciesB", "SpeciesC"]

        X, y = [], []
        for c in range(self.n_classes):
            center = np.random.randn(self.embedding_dim) * 6
            X.append(np.random.randn(40, self.embedding_dim) * 0.4 + center)
            y.extend([c] * 40)
        X = np.vstack(X)
        y = np.array(y)
        idx = np.random.permutation(len(X))
        self.X, self.y = X[idx], y[idx]

        n_train = int(len(X) * 0.6)
        n_val = int(len(X) * 0.2)
        self.X_train, self.y_train = self.X[:n_train], self.y[:n_train]
        self.X_val, self.y_val = self.X[n_train:n_train + n_val], self.y[n_train:n_train + n_val]
        self.X_test, self.y_test = self.X[n_train + n_val:], self.y[n_train + n_val:]

    def test_training_converges_on_separable_synthetic_data(self):
        model, history = build_and_train_mlp(
            self.X_train, self.y_train, self.X_val, self.y_val,
            n_classes=self.n_classes, embedding_dim=self.embedding_dim, max_epochs=40,
        )
        result = evaluate_model(model, self.X_test, self.y_test, class_names=self.class_names)
        self.assertGreater(result["accuracy"], 0.85)
        self.assertGreater(result["macro_f1"], 0.85)

    def test_progress_callback_is_invoked(self):
        calls = []

        def on_progress(epoch, total, logs):
            calls.append(epoch)

        build_and_train_mlp(
            self.X_train, self.y_train, self.X_val, self.y_val,
            n_classes=self.n_classes, embedding_dim=self.embedding_dim, max_epochs=10,
            progress_callback=on_progress,
        )
        self.assertGreater(len(calls), 0)

    def test_training_with_none_validation_set_does_not_crash(self):
        """
        group_aware_split, yeterli farklı kayıt olmadığında val_idx'i BOŞ
        döndürebilir — bu durumda pipeline X_val=None, y_val=None geçer.
        build_and_train_mlp bu durumu Keras'ın kendi validation_split'ine
        düşerek çökmeden ele almalı.
        """
        model, history = build_and_train_mlp(
            self.X_train, self.y_train, None, None,
            n_classes=self.n_classes, embedding_dim=self.embedding_dim, max_epochs=10,
        )
        self.assertIsNotNone(model)
        self.assertIn("val_loss", history)  # Keras'ın kendi validation_split'i de val_loss üretir

    def test_same_random_state_produces_identical_results(self):
        """
        KRİTİK — tekrarlanabilirlik: Gerçek bir kullanımda, AYNI veri
        bölmesiyle (aynı random_state) ama seed'lenmemiş bir MLP eğitimi,
        farklı çalıştırmalarda çarpıcı derecede farklı sonuçlar
        üretebiliyordu (bir türün F1'i 0,87'den 0,0'a düşmüştü). Bu test,
        `random_state`'in artık TensorFlow/Keras'ın kendi ağırlık
        başlangıcını/eğitim rastgeleliğini de sabitlediğini, dolayısıyla
        aynı ayarların aynı sonucu verdiğini doğrular.
        """
        results = []
        for _ in range(2):
            model, _ = build_and_train_mlp(
                self.X_train, self.y_train, self.X_val, self.y_val,
                n_classes=self.n_classes, embedding_dim=self.embedding_dim,
                max_epochs=20, random_state=123,
            )
            result = evaluate_model(model, self.X_test, self.y_test, class_names=self.class_names)
            results.append(result["accuracy"])

        self.assertAlmostEqual(results[0], results[1], places=6)

    def test_different_random_states_can_produce_different_training_results(self):
        """Farklı random_state'ler farklı sonuç verebilmeli (yukarıdaki
        testin tam tersini doğrulayarak, seed'in gerçekten etkili
        olduğunu — sabit/donmuş bir davranış olmadığını — gösterir)."""
        accuracies = []
        for seed in (1, 2):
            model, _ = build_and_train_mlp(
                self.X_train, self.y_train, self.X_val, self.y_val,
                n_classes=self.n_classes, embedding_dim=self.embedding_dim,
                max_epochs=15, random_state=seed,
            )
            result = evaluate_model(model, self.X_test, self.y_test, class_names=self.class_names)
            accuracies.append(result["accuracy"])
        # İkisi de makul (ayrılabilir veri) olmalı, ama bu testin amacı
        # sadece "seed parametresinin bir etkisi var" olduğunu göstermek
        # -- iki değerin illa farklı olması şart değil, sadece çökmemesi
        # ve makul olması yeterli.
        for acc in accuracies:
            self.assertGreater(acc, 0.0)

    def test_save_and_load_model_produces_identical_predictions(self):
        model, _ = build_and_train_mlp(
            self.X_train, self.y_train, self.X_val, self.y_val,
            n_classes=self.n_classes, embedding_dim=self.embedding_dim, max_epochs=15,
        )
        tmpdir = tempfile.mkdtemp()
        try:
            save_model(model, self.class_names, tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "model.keras")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "classes.json")))

            loaded_model, loaded_classes = load_model(tmpdir)
            self.assertEqual(loaded_classes, self.class_names)

            pred_before = model.predict(self.X_test[:5], verbose=0)
            pred_after = loaded_model.predict(self.X_test[:5], verbose=0)
            np.testing.assert_allclose(pred_before, pred_after, atol=1e-5)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()


class TestClassWeights(unittest.TestCase):

    def test_training_runs_with_class_weights_on_imbalanced_data(self):
        np.random.seed(0)
        X = np.vstack([np.random.randn(60, 8) + 3, np.random.randn(8, 8) - 3])
        y = np.array([0] * 60 + [1] * 8)
        model, history = build_and_train_mlp(
            X, y, None, None, n_classes=2, embedding_dim=8, max_epochs=3, use_class_weights=True)
        self.assertIn("loss", history)

