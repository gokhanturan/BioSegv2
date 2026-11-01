import pandas as pd

from src.data import point_effort_report as per


def _df():
    rows = [
        # P1: 2 kayıt (R1 600 sn, R2 300 sn), R1 iki segment
        ("R1", "600", "P1", "3", "A"), ("R1", "600", "P1", "3", "B"), ("R2", "300", "P1", "6", "A"),
        # P2: 1 kayıt, süre virgüllü
        ("R3", "120,5", "P2", "3", "C"),
        # P3: süre okunamıyor
        ("R4", "", "P3", "3", "A"),
        # Habitat_ID boş
        ("R5", "100", "", "3", "A"),
    ]
    return pd.DataFrame(rows, columns=["Original_Recording_Name", "Original_Duration", "Habitat_ID",
                                       "Duration", "Turkish_Name"])


def test_parse_seconds():
    assert per.parse_seconds("12,5") == 12.5
    assert per.parse_seconds("1:30") == 90
    assert per.parse_seconds("0:01:30") == 90
    assert per.parse_seconds("abc") is None
    assert per.parse_seconds("-3") is None
    assert per.parse_seconds(None) is None


def test_per_point_counts_and_durations():
    rep = per.build_effort_report(_df())
    pt = rep.per_point.set_index("Habitat_ID")
    assert set(pt.index) == {"P1", "P2", "P3"}
    assert pt.loc["P1", "Kayit_Sayisi"] == 2
    assert pt.loc["P1", "Toplam_Sure_sn"] == 900          # R1 bir kez sayılır (600), R2 300
    assert pt.loc["P1", "Segment_Sayisi"] == 3
    assert pt.loc["P1", "Tur_Sayisi"] == 2
    assert abs(pt.loc["P2", "Toplam_Sure_sn"] - 120.5) < 1e-9
    assert pd.isna(pt.loc["P3", "Toplam_Sure_sn"])
    assert pd.isna(pt.loc["P3", "Etiketli_Sure_Orani"])


def test_label_ratio_uses_same_recordings():
    rep = per.build_effort_report(_df())
    pt = rep.per_point.set_index("Habitat_ID")
    # P1: etiketli segment süresi 3+3+6=12 sn, kayıt toplamı 900 sn
    assert abs(pt.loc["P1", "Etiketli_Sure_Orani"] - 12 / 900) < 1e-9


def test_warnings_for_missing_id_and_duration():
    rep = per.build_effort_report(_df())
    text = " ".join(rep.warnings)
    assert "Habitat_ID boş" in text
    assert "okunamadı" in text


def test_recording_under_two_points_warns():
    df = pd.DataFrame([("R1", "100", "P1", "3", "A"), ("R1", "100", "P2", "3", "B")],
                      columns=["Original_Recording_Name", "Original_Duration", "Habitat_ID", "Duration", "Turkish_Name"])
    rep = per.build_effort_report(df)
    assert any("birden çok Habitat_ID" in w for w in rep.warnings)


def test_numeric_habitat_id_normalized():
    df = pd.DataFrame([("R1", "100", "12.0", "3", "A"), ("R2", "100", "12", "3", "A")],
                      columns=["Original_Recording_Name", "Original_Duration", "Habitat_ID", "Duration", "Turkish_Name"])
    rep = per.build_effort_report(df)
    assert list(rep.per_point["Habitat_ID"]) == ["12"]
    assert rep.per_point.iloc[0]["Kayit_Sayisi"] == 2


def test_missing_columns_and_empty():
    assert per.build_effort_report(pd.DataFrame({"x": [1]})).warnings
    empty = pd.DataFrame(columns=["Original_Recording_Name", "Original_Duration", "Habitat_ID"])
    assert per.build_effort_report(empty).per_point.empty


def test_format_and_csv(tmp_path):
    rep = per.build_effort_report(_df())
    txt = rep.format()
    assert "SINIRLAMA" in txt and "P1" in txt
    files = per.write_csvs(str(tmp_path), rep)
    assert "nokta_kayit_suresi.csv" in files and (tmp_path / "nokta_kayit_ayrintisi.csv").exists()


def test_lm_pair_counted_once_in_effort_report():
    df = pd.DataFrame([
        ("T1_1L.wav", "100", "P1", "3", "A"), ("T1_1M.wav", "100", "P1", "3", "A"),
        ("T1_2L.wav", "50", "P1", "3", "B"),
    ], columns=["Original_Recording_Name", "Original_Duration", "Habitat_ID", "Duration", "Turkish_Name"])
    rep = per.build_effort_report(df)
    row = rep.per_point.iloc[0]
    assert row["Kayit_Sayisi"] == 2 and row["Toplam_Sure_sn"] == 150
    assert any("L/M" in w for w in rep.warnings)
