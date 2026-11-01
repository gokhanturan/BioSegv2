"""src.data.dataset_summary için birim testleri."""

import csv
import os
import tempfile
import unittest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_summary import (
    load_dataset, compute_summary, compute_richness_by_group,
    compute_group_richness, compute_species_accumulation_curve,
    LOW_SAMPLE_THRESHOLD, GOOD_SAMPLE_THRESHOLD,
)

HEADERS = [
    "Original_Recording_Name", "Original_File_Path", "Original_Duration",
    "Channels", "File_Format", "Bit_Depth", "Recording_Date", "Recording_Time",
    "Filename", "File_Path", "Start_Time", "End_Time", "Duration",
    "Call_Type", "type_Count", "Order", "Family", "Scientific_Name", "Turkish_Name",
    "Background_Order", "Background_Family", "Background_Scientific_Name",
    "Background_Turkish_Name", "Background_Noise_Type",
    "Notes_For_Other_Species", "Location",
    "Recordist", "Ornithologist", "Verification_Status", "Confidence_Level",
    "Microphone", "Recorder",
    "Habitat_ID", "Habitat_Type", "Code", "UTM_Zone", "UTM_Easting", "UTM_Northing",
    "Lat", "Lon", "Google_Maps", "Notes", "Project",
]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(HEADERS)
        for row_dict in rows:
            row = [""] * len(HEADERS)
            for k, v in row_dict.items():
                row[HEADERS.index(k)] = v
            writer.writerow(row)


class TestDatasetSummary(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "test_dataset.csv")

    def tearDown(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        os.rmdir(self.tmpdir)

    def test_missing_file_returns_none(self):
        self.assertIsNone(load_dataset("/nonexistent/path.csv"))

    def test_empty_file_returns_none(self):
        with open(self.csv_path, "w", encoding="utf-8-sig") as f:
            f.write("")
        self.assertIsNone(load_dataset(self.csv_path))

    def test_basic_counts(self):
        _write_csv(self.csv_path, [
            {"Turkish_Name": "Karatavuk", "Location": "Burdur", "Habitat_Type": "Orman"},
            {"Turkish_Name": "Karatavuk", "Location": "Burdur", "Habitat_Type": "Orman"},
            {"Turkish_Name": "Orman Toygarı", "Location": "Isparta", "Habitat_Type": "Sulak Alan"},
        ])
        df = load_dataset(self.csv_path)
        summary = compute_summary(df)

        self.assertEqual(summary.total_records, 3)
        self.assertEqual(summary.unique_species, 2)
        self.assertEqual(summary.unique_locations, 2)
        self.assertEqual(summary.unique_habitats, 2)
        self.assertEqual(summary.species_counts["Karatavuk"], 2)
        self.assertEqual(summary.species_counts["Orman Toygarı"], 1)

    def test_low_and_well_sampled_species_classification(self):
        rows = [{"Turkish_Name": "Karatavuk", "Location": "Burdur"} for _ in range(GOOD_SAMPLE_THRESHOLD)]
        rows.append({"Turkish_Name": "Nadir Tür", "Location": "Burdur"})
        _write_csv(self.csv_path, rows)

        df = load_dataset(self.csv_path)
        summary = compute_summary(df)

        well_sampled_names = [name for name, _ in summary.well_sampled_species]
        low_sampled_names = [name for name, _ in summary.low_sample_species]

        self.assertIn("Karatavuk", well_sampled_names)
        self.assertIn("Nadir Tür", low_sampled_names)
        self.assertNotIn("Karatavuk", low_sampled_names)

    def test_missing_column_does_not_crash(self):
        # Location sütunu olmadan da çalışması gerekir (boş Series döner)
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Turkish_Name"])
            writer.writerow(["Karatavuk"])
        df = load_dataset(self.csv_path)
        summary = compute_summary(df)
        self.assertEqual(summary.total_records, 1)
        self.assertEqual(summary.unique_locations, 0)

    def test_blank_values_grouped_as_unspecified(self):
        _write_csv(self.csv_path, [
            {"Turkish_Name": "Karatavuk", "Location": ""},
            {"Turkish_Name": "", "Location": "Burdur"},
        ])
        df = load_dataset(self.csv_path)
        summary = compute_summary(df)
        # Boş değerler "(belirtilmemiş)" olarak gruplanmalı, farklı tür
        # sayısına dahil edilmemeli
        self.assertEqual(summary.unique_species, 1)

    def test_habitat_richness_counts_unique_species_only(self):
        _write_csv(self.csv_path, [
            {"Turkish_Name": "Karatavuk", "Habitat_ID": "12", "Habitat_Type": "Orman"},
            {"Turkish_Name": "Karatavuk", "Habitat_ID": "12", "Habitat_Type": "Orman"},  # tekrar
            {"Turkish_Name": "Orman Toygarı", "Habitat_ID": "12", "Habitat_Type": "Orman"},
            {"Turkish_Name": "Anadolu Sıvacısı", "Habitat_ID": "69", "Habitat_Type": "Sulak Alan"},
        ])
        df = load_dataset(self.csv_path)
        summary = compute_summary(df)

        self.assertEqual(summary.habitat_richness["12 (Orman)"], 2)
        self.assertEqual(summary.habitat_richness["69 (Sulak Alan)"], 1)

    def test_richness_by_group_ignores_blank_species(self):
        _write_csv(self.csv_path, [
            {"Turkish_Name": "Karatavuk", "Habitat_ID": "12"},
            {"Turkish_Name": "", "Habitat_ID": "12"},
        ])
        df = load_dataset(self.csv_path)
        richness = compute_richness_by_group(df, "Habitat_ID")
        self.assertEqual(richness["12"], 1)

    def test_richness_by_group_missing_columns_returns_empty(self):
        _write_csv(self.csv_path, [{"Turkish_Name": "Karatavuk"}])
        df = load_dataset(self.csv_path)
        richness = compute_richness_by_group(df, "Nonexistent_Column")
        self.assertEqual(len(richness), 0)

    def test_format_richness_summary_is_separate_from_text_summary(self):
        _write_csv(self.csv_path, [
            {"Turkish_Name": "Karatavuk", "Habitat_ID": "12", "Habitat_Type": "Orman"},
            {"Turkish_Name": "Orman Toygarı", "Habitat_ID": "12", "Habitat_Type": "Orman"},
        ])
        df = load_dataset(self.csv_path)
        summary = compute_summary(df)

        text_summary = summary.format_text_summary()
        richness_summary = summary.format_richness_summary()

        # Genel özet metni tür zenginliği ayrıntısı İÇERMEMELİ
        # (ayrı pencereye taşındı); zenginlik özeti İÇERMELİ.
        self.assertNotIn("zengin", text_summary.lower())
        self.assertIn("zengin", richness_summary.lower())
        self.assertIn("Orman", richness_summary)

    def test_group_richness_counts_observed_species_and_recordings(self):
        """Elle sayım: 5 kayıt, 4 farklı tür -> belirlenen zenginlik 4, kayıt sayısı 5."""
        rows = []
        for rec, species_list in [
            ("R1", ["SpeciesA", "SpeciesB"]), ("R2", ["SpeciesA", "SpeciesC"]),
            ("R3", ["SpeciesA"]), ("R4", ["SpeciesD"]), ("R5", ["SpeciesD"]),
        ]:
            for sp in species_list:
                rows.append({"Habitat_ID": "H1", "Turkish_Name": sp, "Original_Recording_Name": rec})
        _write_csv(self.csv_path, rows)

        df = load_dataset(self.csv_path)
        results = compute_group_richness(df, "Habitat_ID")

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.observed_richness, 4)
        self.assertEqual(r.n_recordings, 5)

    def test_effort_normalized_richness_uses_unique_recording_duration_only(self):
        """
        KRİTİK: aynı kayıttan gelen birden fazla segmentin süresi TEKRAR
        SAYILMAMALI — toplam süre, benzersiz kayıtların süreleri toplamı
        olmalı, segment sayısına göre değil.
        """
        rows = [
            {"Habitat_ID": "H4", "Turkish_Name": "SpeciesA", "Original_Recording_Name": "R1", "Original_Duration": "600"},
            {"Habitat_ID": "H4", "Turkish_Name": "SpeciesB", "Original_Recording_Name": "R1", "Original_Duration": "600"},  # AYNI kayıt, tekrar
            {"Habitat_ID": "H4", "Turkish_Name": "SpeciesC", "Original_Recording_Name": "R2", "Original_Duration": "1200"},
        ]
        _write_csv(self.csv_path, rows)
        df = load_dataset(self.csv_path)
        results = compute_group_richness(df, "Habitat_ID")

        r = results[0]
        # Toplam süre (600 + 1200) / 3600 saat olmalı, (600+600+1200) DEĞİL
        self.assertAlmostEqual(r.total_duration_hours, (600 + 1200) / 3600, places=6)

    def test_missing_recording_column_returns_empty_list(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Turkish_Name", "Habitat_ID"])
            writer.writerow(["SpeciesA", "H1"])
        df = load_dataset(self.csv_path)
        results = compute_group_richness(df, "Habitat_ID")
        self.assertEqual(results, [])

    def test_accumulation_curve_reaches_true_species_count(self):
        """Tüm kayıtlar dahil edildiğinde, kümülatif tür sayısı gerçek S_obs'a eşit olmalı."""
        rows = []
        for i in range(8):
            rows.append({"Habitat_ID": "H1", "Turkish_Name": f"Species{i}", "Original_Recording_Name": f"R{i}"})
        _write_csv(self.csv_path, rows)

        df = load_dataset(self.csv_path)
        curve = compute_species_accumulation_curve(df, "Habitat_ID", "H1", n_permutations=50, random_state=1)

        self.assertIsNotNone(curve)
        self.assertEqual(curve.x, list(range(1, 9)))
        self.assertAlmostEqual(curve.y_mean[-1], 8.0, places=6)
        self.assertAlmostEqual(curve.y_mean[0], 1.0, places=6)  # ilk kayıtta hep 1 tür

    def test_accumulation_curve_detects_leveling_off(self):
        """Yeni tür eklenmeyi bıraktığında, eğri 'düzleşiyor' olarak işaretlenmeli."""
        rows = []
        species_pool = [f"Species{i}" for i in range(5)]
        # ilk 5 kayıtta hepsi yeni, sonraki 15 kayıtta hep tekrar
        for i in range(5):
            rows.append({"Habitat_ID": "H1", "Turkish_Name": species_pool[i], "Original_Recording_Name": f"R{i}"})
        for i in range(5, 20):
            rows.append({"Habitat_ID": "H1", "Turkish_Name": species_pool[i % 5], "Original_Recording_Name": f"R{i}"})
        _write_csv(self.csv_path, rows)

        df = load_dataset(self.csv_path)
        curve = compute_species_accumulation_curve(df, "Habitat_ID", "H1", n_permutations=100, random_state=3)

        self.assertTrue(curve.is_leveling_off)
        self.assertAlmostEqual(curve.y_mean[-1], 5.0, places=6)

    def test_accumulation_curve_detects_still_rising(self):
        """Her kayıtta yeni bir tür varsa, eğri düzleşmemiş (hâlâ dik) olarak işaretlenmeli."""
        rows = []
        for i in range(12):
            rows.append({"Habitat_ID": "H1", "Turkish_Name": f"UniqueSpecies{i}", "Original_Recording_Name": f"R{i}"})
        _write_csv(self.csv_path, rows)

        df = load_dataset(self.csv_path)
        curve = compute_species_accumulation_curve(df, "Habitat_ID", "H1", n_permutations=50, random_state=5)

        self.assertFalse(curve.is_leveling_off)

    def test_accumulation_curve_missing_group_returns_none(self):
        _write_csv(self.csv_path, [
            {"Habitat_ID": "H1", "Turkish_Name": "SpeciesA", "Original_Recording_Name": "R1"},
        ])
        df = load_dataset(self.csv_path)
        curve = compute_species_accumulation_curve(df, "Habitat_ID", "NONEXISTENT_HABITAT")
        self.assertIsNone(curve)


if __name__ == "__main__":
    unittest.main()

class TestPointBasedSummary(unittest.TestCase):
    def test_compute_summary_has_point_based_results_and_honest_warnings(self):
        import tempfile, os
        rows = []
        for pid, loc, typ, sps in [
            ("1", "A", "Orman", ["s1", "s2"]), ("2", "A", "Orman", ["s1", "s3"]),
            ("3", "B", "Makilik", ["s4"]),
        ]:
            for sp in sps:
                rows.append({"Habitat_ID": pid, "Location": loc, "Habitat_Type": typ, "Turkish_Name": sp})
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "d.csv")
        _write_csv(path, rows)
        summary = compute_summary(load_dataset(path))
        self.assertEqual(summary.n_points, 3)
        self.assertEqual(summary.n_species_observed, 4)
        self.assertEqual({e.group: e.m for e in summary.type_richness}, {"Orman": 2, "Makilik": 1})
        self.assertIsNotNone(summary.location_rarefaction)
        text = summary.format_richness_summary()
        self.assertIn("belirlenen toplam tür: 4", text)
        self.assertNotIn("Chao2", text)
        self.assertNotIn("tür/saat", text)

    def test_missing_columns_do_not_crash(self):
        import tempfile, os
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "d.csv")
        _write_csv(path, [{"Turkish_Name": "Karatavuk"}])
        summary = compute_summary(load_dataset(path))
        self.assertEqual(summary.n_species_observed, 0)
        self.assertEqual(summary.n_points, 0)


def test_extra_summary_stats_real_shape(tmp_path):
    import pandas as pd
    from src.data.dataset_summary import compute_summary
    rows = []
    for i in range(36):   # tür A: 36 segment, 3 kayıt -> süzgeci geçer
        rows.append({"Turkish_Name": "A", "Original_Recording_Name": f"r{i % 3}", "Habitat_ID": "1", "Call_Type": "Courtship"})
    for i in range(36):   # tür B: 36 segment ama tek kayıt -> geçmez
        rows.append({"Turkish_Name": "B", "Original_Recording_Name": "r9", "Habitat_ID": "2", "Call_Type": "Call"})
    rows.append({"Turkish_Name": "C", "Original_Recording_Name": "r1", "Habitat_ID": "", "Call_Type": "Call"})
    s = compute_summary(pd.DataFrame(rows))
    assert s.n_recordings == 4 and s.labeled_points == 2
    assert s.trainable_species == 1 and s.trainable_segments == 36
    assert s.median_per_species == 36 and s.min_per_species == 1
    t = s.format_text_summary()
    assert "Toplam segment: 73" in t and "medyan 36" in t and "1 tür, 36 segment" in t
    assert "zengin" not in t.lower()
