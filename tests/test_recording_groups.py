import pandas as pd

from src.data.recording_groups import canonical_recording_ids, merged_pairs


def _df(rows):
    return pd.DataFrame(rows, columns=["Original_Recording_Name", "Original_Duration", "Habitat_ID"])


def test_pair_merged_when_same_point_and_duration():
    df = _df([("Tsu5_1L.wav", "187.241", "224"), ("Tsu5_1M.wav", "187.241", "224"), ("Tsu5_2L.wav", "562.5", "224")])
    c = list(canonical_recording_ids(df))
    assert c[0] == c[1] == "Tsu5_1[LM].wav"
    assert c[2] == "Tsu5_2L.wav"


def test_merged_by_name_even_if_duration_or_point_differs_but_conflict_reported():
    # Kullanıcının kuralı: ad tabanı aynı + L ve M varsa aynı kayıt. Süre/nokta farkı yalnızca uyarılır.
    from src.data.recording_groups import merge_conflicts
    df = _df([("A_1L.wav", "100", "1"), ("A_1M.wav", "101", "1")])
    assert list(canonical_recording_ids(df)) == ["A_1[LM].wav"] * 2
    assert any("süreleri farklı" in w for w in merge_conflicts(df))
    df = _df([("A_1L.wav", "100", "1"), ("A_1M.wav", "100", "2")])
    assert list(canonical_recording_ids(df)) == ["A_1[LM].wav"] * 2
    assert any("Habitat_ID" in w for w in merge_conflicts(df))
    df = _df([("A_1L.wav", "100", "1"), ("A_1M.wav", "100", "1")])
    assert merge_conflicts(df) == []


def test_missing_duration_or_point_still_merged():
    df = _df([("A_1L.wav", "", ""), ("A_1M.wav", "", "")])
    assert list(canonical_recording_ids(df)) == ["A_1[LM].wav"] * 2


def test_only_one_channel_not_merged_and_other_names_untouched():
    df = _df([("A_1L.wav", "100", "1"), ("A_2L.wav", "100", "1"), ("rec_final.wav", "30", "1")])
    assert list(canonical_recording_ids(df)) == ["A_1L.wav", "A_2L.wav", "rec_final.wav"]


def test_missing_columns_and_merged_pairs_report():
    df = pd.DataFrame({"Original_Recording_Name": ["x.wav"]})
    assert list(canonical_recording_ids(df)) == ["x.wav"]
    df = _df([("T_3L.wav", "423.451", "1"), ("T_3M.wav", "423.451", "1")])
    assert merged_pairs(df) == [("T_3[LM].wav", ["T_3L.wav", "T_3M.wav"])]


def test_dataset_builder_uses_group_id(tmp_path):
    """Segmentler L ve M altında olsa da recording_id aynı çıkar (bölmede aynı tarafa düşer)."""
    import os
    from src.ml import dataset_builder as DB
    cls = tmp_path / "cls"
    rows = []
    for i, (rec, sp) in enumerate([("T_1L.wav", "Parus major"), ("T_1M.wav", "Parus major"),
                                   ("T_2L.wav", "Parus major"), ("U_1L.wav", "Fringilla coelebs"),
                                   ("U_2L.wav", "Fringilla coelebs")]):
        fn = f"seg{i}.wav"
        folder = cls / DB._sanitize_folder_name(sp)
        os.makedirs(folder, exist_ok=True)
        (folder / fn).write_bytes(b"x")
        rows.append({"Scientific_Name": sp, "Filename": fn, "Original_Recording_Name": rec,
                     "Original_Duration": "100" if rec.startswith("T_1") else "50", "Habitat_ID": "1"})
    csv = tmp_path / "d.csv"
    pd.DataFrame(rows).to_csv(csv, sep=";", index=False, encoding="utf-8-sig")
    samples, _, _ = DB.build_training_samples(csv_path=str(csv), classification_dir=str(cls),
                                              min_samples_per_class=1, min_recordings_per_class=1)
    ids = {s.recording_id for s in samples if s.species == "Parus major"}
    assert "T_1[LM].wav" in ids and "T_1L.wav" not in ids and len(ids) == 2
