"""Ön Hazırlık: varsayılan etiket bilgileri, yedekten geri yükleme, model durumu, veri klasörleri."""
import json
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.data import on_hazirlik
from src.data.catalog_manager import CatalogStore


def _write(p, data):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ---------------------------------------------------------------- varsayılanlar
def test_defaults_roundtrip_and_cleaning(tmp_path):
    assert on_hazirlik.load_label_defaults(str(tmp_path)) == {}
    on_hazirlik.save_label_defaults(str(tmp_path), {"Recordist": " Ali ", "Microphone": "", "Bilinmeyen": "x"})
    assert on_hazirlik.load_label_defaults(str(tmp_path)) == {"Recordist": "Ali"}


def test_defaults_corrupt_file_is_ignored(tmp_path):
    (tmp_path / on_hazirlik.DEFAULTS_FILE).write_text("{bozuk", encoding="utf-8")
    assert on_hazirlik.load_label_defaults(str(tmp_path)) == {}


# ---------------------------------------------------------------- yedekler
def _store_with_backups(tmp_path):
    root = str(tmp_path)
    _write(tmp_path / "bird_species_data.json", [])
    _write(tmp_path / "habitat_data.json", [])
    _write(tmp_path / "bird_sound_config.json", {"project": "A"})
    st = CatalogStore(root)
    st.set_text("project", "B")
    st.save()
    time.sleep(0.01)
    st.set_text("project", "C")
    st.save()
    return root, st


def test_list_and_restore_backup(tmp_path):
    root, _ = _store_with_backups(tmp_path)
    backups = on_hazirlik.list_backups(root)
    assert len(backups) == 2 and backups[0]["when"] >= backups[1]["when"]
    oldest = backups[-1]  # içinde "A" var
    assert on_hazirlik.restore_backup(root, oldest["path"]) == "bird_sound_config.json"
    assert json.load(open(os.path.join(root, "bird_sound_config.json"), encoding="utf-8"))["project"] == "A"
    # geri yüklemeden önce geçerli dosya ("C") yedeklendi -> geri alınabilir
    assert len(on_hazirlik.list_backups(root)) == 3


def test_restore_rejects_corrupt_and_foreign_files(tmp_path):
    root, _ = _store_with_backups(tmp_path)
    bad = on_hazirlik.list_backups(root)[0]
    with open(bad["path"], "w", encoding="utf-8") as f:
        f.write("{bozuk")
    before = open(os.path.join(root, "bird_sound_config.json"), encoding="utf-8").read()
    with pytest.raises(ValueError):
        on_hazirlik.restore_backup(root, bad["path"])
    assert open(os.path.join(root, "bird_sound_config.json"), encoding="utf-8").read() == before
    with pytest.raises(ValueError):
        on_hazirlik.restore_backup(root, str(tmp_path / "bird_sound_config.json"))


# ---------------------------------------------------------------- model durumu / veri
def test_model_status_has_three_models_and_combined_needs_both(monkeypatch):
    monkeypatch.setattr(on_hazirlik, "birdnet_status", lambda: (True, "ok"))
    monkeypatch.setattr(on_hazirlik, "perch_status", lambda: (False, "yok"))
    st = on_hazirlik.model_status()
    assert [s[0] for s in st] == ["BirdNET 2.4", "Perch 2.0", "BirdNET + Perch"]
    assert [s[1] for s in st] == [True, False, False]
    monkeypatch.setattr(on_hazirlik, "perch_status", lambda: (True, "ok"))
    assert on_hazirlik.model_status()[2][1] is True


def test_data_overview_counts(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("a;b\r\n1;2\r\n3;4\r\n", encoding="utf-8-sig")
    seg = tmp_path / "segments"
    seg.mkdir()
    (seg / "x.wav").write_bytes(b"0")
    (seg / "y.WAV").write_bytes(b"0")
    (seg / "not.txt").write_text("")
    ov = on_hazirlik.data_overview(str(csv), [str(tmp_path / "yok"), str(seg)])
    assert ov["csv_rows"] == 2 and ov["segment_files"] == 2 and ov["segments_dir"] == str(seg)
    ov2 = on_hazirlik.data_overview(str(tmp_path / "yok.csv"), [str(tmp_path / "yok")])
    assert ov2["csv_rows"] == 0 and not ov2["csv_exists"] and not ov2["segments_exists"]


# ---------------------------------------------------------------- arayüz
@pytest.fixture(scope="module")
def win():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from src.ui.main_window import BirdSoundApp
    w = BirdSoundApp()
    yield w
    w.close()
    del app


def test_prep_tab_has_cards_and_sections(win):
    from PyQt5.QtWidgets import QGroupBox
    assert win.btn_restore_backup.isEnabled() and win.btn_catalog_manager.isEnabled()
    titles = {g.title() for g in win._prep_group.findChildren(QGroupBox)}
    assert {"Varsayılan Etiket Bilgileri", "Veri Klasörleri", "Modeller"} <= titles
    assert win.prep_panel.btn_refresh_models.isEnabled()


def test_defaults_apply_to_form(win, tmp_path, monkeypatch):
    import src.ui.main_window as mw
    monkeypatch.setattr(mw, "src_root_dir", lambda: str(tmp_path))
    on_hazirlik.save_label_defaults(str(tmp_path), {"Recordist": "Deneme Kişi", "Recorder": "Zoom H4n Pro"})
    win.apply_label_defaults()
    assert win.cmb_recordist.currentText() == "Deneme Kişi"
    assert win.cmb_recorder.currentText() == "Zoom H4n Pro"


def test_backup_dialog_restores(tmp_path):
    from PyQt5 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from src.ui.prep_panel import BackupRestoreDialog
    root, _ = _store_with_backups(tmp_path)
    dlg = BackupRestoreDialog(root)
    assert dlg.table.rowCount() == 2 and dlg.btn_restore.isEnabled()
    got = []
    dlg.restored.connect(got.append)
    dlg.table.selectRow(1)  # en eski -> "A"
    assert dlg.restore_selected(confirm=False) == "bird_sound_config.json"
    assert got == ["bird_sound_config.json"]
    assert json.load(open(os.path.join(root, "bird_sound_config.json"), encoding="utf-8"))["project"] == "A"


def test_empty_backup_dialog_disables_button(tmp_path):
    from PyQt5 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from src.ui.prep_panel import BackupRestoreDialog
    dlg = BackupRestoreDialog(str(tmp_path))
    assert dlg.table.rowCount() == 0 and not dlg.btn_restore.isEnabled()
