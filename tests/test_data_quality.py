import os
import pandas as pd
import pytest

from src.data import data_quality as dq


def _df(rows):
    cols = ["Original_Recording_Name", "Filename", "Start_Time", "End_Time", "Scientific_Name",
            "Turkish_Name", "Location", "Habitat_ID", "Habitat_Type", "Call_Type", "File_Path"]
    return pd.DataFrame(rows, columns=cols, dtype=str)


def _row(rec, fn, s, e, sp, loc="Isparta", hid="1", ht="Orman", fp=""):
    return [rec, fn, s, e, sp, sp.lower(), loc, hid, ht, "Call", fp]


def test_clean_data():
    df = _df([_row("r1", "a1", "0", "3", "A"), _row("r1", "a2", "10", "13", "A"), _row("r2", "b", "0", "3", "B")])
    r = dq.audit(df, check_files=False)
    assert r.n_duplicate_extra == 0 and r.n_overlap_pairs == 0 and not r.inconsistencies
    assert not any(r.empty_counts.values()) and r.impact is None
    assert len(r.suspicious_rows()) == 0


def test_duplicates_overlaps_and_readonly():
    rows = [_row("r1", f"a{i}", "0", "3", "A") for i in range(2)]          # 1 yinelenen
    rows += [_row("r1", "c", "2", "5", "B")]                               # örtüşür
    df = _df(rows)
    before = df.copy()
    r = dq.audit(df, check_files=False)
    assert r.n_duplicate_extra == 1 and r.n_duplicate_recordings == 1
    assert r.n_overlap_pairs >= 1
    assert df.equals(before)                      # girdi değişmedi
    sus = r.suspicious_rows()
    assert set(sus["Neden"]) >= {"Yinelenen segment"} and "CSV_Satir" in sus


def test_inconsistency_and_empty():
    df = _df([_row("r1", "a", "0", "3", "A", loc="Isparta"), _row("r1", "b", "5", "8", "A", loc="Burdur", ht="Su"),
              _row("r2", "c", "0", "3", "A", hid="")])
    r = dq.audit(df, check_files=False)
    assert any("konuma" in x for x in r.inconsistencies)
    assert r.empty_counts["Habitat_ID"] == 1


def test_impact_on_trainable_species():
    n = dq.MIN_SEGMENT                                   # eşik: tam eşikte, biri yinelenen
    rows = [_row("r1", f"a{i}", str(3 * i), str(3 * i + 3), "A") for i in range(n // 2)]
    rows += [_row("r2", f"b{i}", str(3 * i), str(3 * i + 3), "A") for i in range(n - n // 2 - 1)]
    rows += [_row("r2", "dup", "0", "3", "A")]       # eşiği tamamlayan son segment yinelenen
    r = dq.audit(_df(rows), check_files=False)
    assert r.impact["species_now"] == 1 and r.impact["species_dedup"] == 0
    assert r.impact["species_lost"] == ["A"]
    assert "hiçbir şey silinmedi" in r.format_text()


def test_file_check(tmp_path):
    (tmp_path / "x.wav").write_bytes(b"1")
    df = _df([_row("r1", "x.wav", "0", "3", "A"), _row("r1", "y.wav", "5", "8", "A")])
    r = dq.audit(df, search_dirs=[str(tmp_path)])
    assert (r.files_checked, r.files_missing) == (2, 1)


def test_missing_columns_skipped():
    r = dq.audit(pd.DataFrame({"Turkish_Name": ["A", "B"]}), check_files=False)
    assert "yinelenen" in r.skipped and "Atlandı" in r.format_text()


def test_dialog_offscreen(tmp_path):
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from src.ui.data_quality_dialog import DataQualityDialog
    p = tmp_path / "d.csv"
    _df([_row("r1", "a", "0", "3", "A"), _row("r1", "b", "0", "3", "A")]).to_csv(p, sep=";", index=False, encoding="utf-8-sig")
    d = DataQualityDialog(str(p))
    assert "1 fazlalık satır" in d.text.toPlainText() and d.btn_export.isEnabled()
    d2 = DataQualityDialog(str(tmp_path / "yok.csv"))
    assert "bulunamadı" in d2.text.toPlainText()


def test_report_is_dynamic_clean_vs_dirty():
    clean = _df([_row("r1", "a", "0", "3", "A"), _row("r2", "b", "0", "3", "B")])
    t = dq.audit(clean, check_files=False).format_text()
    assert "sorun bulunmadı" in t and "Yinelenen olmadığı için etki yok" in t
    dirty = _df([_row("r1", "a", "0", "3", "A"), _row("r1", "a_2", "0", "3", "A"), _row("r2", "b", "0", "3", "B")])
    r = dq.audit(dirty, check_files=False)
    t = r.format_text()
    assert "1 yinelenen grup" in t and "1 grup × 2 satır" in t and "Filename" in t
    assert r.n_species == 2 and r.n_recordings == 2 and "A: 1 / 2" in t


# ---------------------------------------------------------------- eksik dosya raporu
def test_missing_rows_listed_with_excel_row_numbers_and_orphans(tmp_path):
    for n in ("a.wav", "c.wav", "yetim.wav"):
        (tmp_path / n).write_bytes(b"1")
    df = _df([_row("r1", "a.wav", "0", "3", "A"),        # satır 2: var
              _row("r1", "b.wav", "5", "8", "A"),        # satır 3: EKSİK
              _row("r2", "c.wav", "0", "3", "B")])       # satır 4: var
    r = dq.audit(df, search_dirs=[str(tmp_path)])
    assert list(r.missing_rows.index + 2) == [3]
    assert r.n_files_on_disk == 3 and r.orphan_files == ["yetim.wav"]
    t = r.format_text()
    assert "CSV'de 3 satır, segment klasöründe 3 wav" in t      # sayılar eşit ama içerik farklı
    assert "satır 3" in t and "b.wav" in t and "yetim.wav" in t
    assert "1 satırın ses dosyası eksik" in t and "1 wav dosyası CSV'de yok" in t
    sus = r.suspicious_rows()
    assert "Ses dosyası eksik" in set(sus["Neden"]) and int(sus["CSV_Satir"].iloc[0]) == 3


def test_count_mismatch_is_stated(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"1")
    df = _df([_row("r1", "a.wav", "0", "3", "A"), _row("r1", "b.wav", "5", "8", "A")])
    t = dq.audit(df, search_dirs=[str(tmp_path)]).format_text()
    assert "CSV'de 2 satır, segment klasöründe 1 wav" in t and "fark: +1" in t


# ---------------------------------------------------------------- yinelenenleri silme
def _write_csv(path, rows):
    _df(rows).to_csv(path, sep=";", index=False, encoding="utf-8-sig")


def _segdir(tmp_path, names, species="A"):
    seg = tmp_path / "segments"
    cls = seg / "classification" / species
    cls.mkdir(parents=True)
    for n in names:
        (seg / n).write_bytes(b"x")
        (cls / n).write_bytes(b"x")
    return seg


def test_plan_does_not_touch_anything_and_keeps_first_row(tmp_path):
    seg = _segdir(tmp_path, ["a1.wav", "a2.wav", "b.wav"])
    csv_p = tmp_path / "d.csv"
    _write_csv(csv_p, [_row("r1", "a1.wav", "0", "3", "A"), _row("r1", "a2.wav", "0", "3", "A"),
                       _row("r2", "b.wav", "0", "3", "B")])
    before = csv_p.read_bytes()
    plan = dq.plan_duplicate_removal(str(csv_p), [str(seg)])
    assert plan.row_numbers == [3] and plan.n_groups == 1
    assert sorted(os.path.basename(p) for p in plan.files_to_move) == ["a2.wav", "a2.wav"]   # segments + classification
    assert csv_p.read_bytes() == before and (seg / "a2.wav").exists()


def test_apply_removes_rows_backs_up_csv_and_moves_files(tmp_path):
    seg = _segdir(tmp_path, ["a1.wav", "a2.wav", "b.wav"])
    csv_p = tmp_path / "d.csv"
    _write_csv(csv_p, [_row("r1", "a1.wav", "0", "3", "A"), _row("r1", "a2.wav", "0", "3", "A"),
                       _row("r2", "b.wav", "0", "3", "B")])
    res = dq.apply_duplicate_removal(str(csv_p), [str(seg)])
    left = pd.read_csv(csv_p, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert list(left.Filename) == ["a1.wav", "b.wav"] and res.rows_removed == 1
    assert os.path.exists(res.backup_csv) and len(pd.read_csv(res.backup_csv, sep=";", encoding="utf-8-sig")) == 3
    assert not (seg / "a2.wav").exists() and not (seg / "classification" / "A" / "a2.wav").exists()
    assert (seg / "a1.wav").exists() and (seg / "b.wav").exists()                  # korunan satırın dosyası yerinde
    assert (tmp_path / "segments" / os.path.basename(res.backup_dir) / "a2.wav").exists()   # geri alınabilir
    assert dq.audit(left, search_dirs=[str(seg)]).n_duplicate_extra == 0


def test_shared_filename_is_never_moved(tmp_path):
    seg = _segdir(tmp_path, ["same.wav"])
    csv_p = tmp_path / "d.csv"
    _write_csv(csv_p, [_row("r1", "same.wav", "0", "3", "A"), _row("r1", "same.wav", "0", "3", "A")])
    res = dq.apply_duplicate_removal(str(csv_p), [str(seg)])
    assert res.rows_removed == 1 and res.files_moved == [] and res.files_kept_shared == ["same.wav"]
    assert (seg / "same.wav").exists()


def test_apply_without_duplicates_changes_nothing(tmp_path):
    csv_p = tmp_path / "d.csv"
    _write_csv(csv_p, [_row("r1", "a.wav", "0", "3", "A"), _row("r1", "b.wav", "5", "8", "A")])
    before = csv_p.read_bytes()
    res = dq.apply_duplicate_removal(str(csv_p), [])
    assert res.rows_removed == 0 and csv_p.read_bytes() == before and res.backup_csv == ""


def test_other_cells_survive_roundtrip(tmp_path):
    csv_p = tmp_path / "d.csv"
    rows = [_row("r1", "a1", "0", "3", "A"), _row("r1", "a2", "0", "3", "A"), _row("r2", "b", "7,5", "9", "B")]
    rows[2][6] = "Isparta; Merkez"            # ayraç içeren hücre
    _write_csv(csv_p, rows)
    dq.apply_duplicate_removal(str(csv_p), [])
    left = pd.read_csv(csv_p, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert left.loc[1, "Start_Time"] == "7,5" and left.loc[1, "Location"] == "Isparta; Merkez"


_QAPP = []   # QApplication'a referans tutulmazsa çöp toplayıcı onu yok eder ve süreç çöker


def _dialog(tmp_path, rows, seg=None):
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    _QAPP.append(QApplication.instance() or QApplication([]))
    from src.ui.data_quality_dialog import DataQualityDialog
    p = tmp_path / "d.csv"
    _write_csv(p, rows)
    return DataQualityDialog(str(p), search_dirs=[str(seg)] if seg else []), p


def test_dialog_delete_needs_two_yes_answers(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox
    seg = _segdir(tmp_path, ["a1.wav", "a2.wav"])
    d, p = _dialog(tmp_path, [_row("r1", "a1.wav", "0", "3", "A"), _row("r1", "a2.wav", "0", "3", "A")], seg)
    assert d.btn_fix.isEnabled()
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.Ok)
    calls = []

    def answers(seq):
        it = iter(seq)

        def q(*a, **k):
            calls.append(a[1])
            return next(it)
        return q

    monkeypatch.setattr(QMessageBox, "question", answers([QMessageBox.No]))          # 1. onayda vazgeç
    d._fix_duplicates()
    assert len(calls) == 1 and len(pd.read_csv(p, sep=";")) == 2

    calls.clear()
    monkeypatch.setattr(QMessageBox, "question", answers([QMessageBox.Yes, QMessageBox.No]))   # 2. onayda vazgeç
    d._fix_duplicates()
    assert len(calls) == 2 and len(pd.read_csv(p, sep=";")) == 2 and (seg / "a2.wav").exists()

    calls.clear()
    monkeypatch.setattr(QMessageBox, "question", answers([QMessageBox.Yes, QMessageBox.Yes]))
    d._fix_duplicates()
    assert len(calls) == 2 and len(pd.read_csv(p, sep=";")) == 1 and not (seg / "a2.wav").exists()
    assert not d.btn_fix.isEnabled()                                                  # yenilenen rapor: yinelenen kalmadı


def test_turkish_dotted_capital_i_is_not_orphan(tmp_path):
    """'İspinoz' gibi adlar pandas/Python küçültme farkı yüzünden yetim sayılmamalı (181 yanlış alarm)."""
    for n in ("A_İspinoz_1.wav", "B_İskete.wav", "yetim.wav"):
        (tmp_path / n).write_bytes(b"1")
    df = _df([_row("r1", "A_İspinoz_1.wav", "0", "3", "A"),
              _row("r2", "B_İskete.wav", "0", "3", "B")])
    r = dq.audit(df, search_dirs=[str(tmp_path)])
    assert r.orphan_files == ["yetim.wav"]
    assert r.n_unique_filenames == 2


def test_audit_threshold_matches_classifier_default():
    """Denetimin 'eğitim süzgeci' eşiği gerçek sınıflandırıcıyla aynı olmalı (eskiden 10 idi, doğrusu 30)."""
    from src.ml.dataset_builder import DEFAULT_MIN_SEGMENTS
    assert dq.MIN_SEGMENT == DEFAULT_MIN_SEGMENTS == 30
