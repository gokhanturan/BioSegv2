import os
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from src.ui.csv_viewer import CsvViewerDialog, read_csv_text

_app = QApplication.instance() or QApplication([])


def _write(tmp_path, rows):
    p = tmp_path / "d.csv"
    head = "Original_Recording_Name;Scientific_Name;Habitat_ID;Start_Time;Extra"
    p.write_text("\n".join([head] + rows), encoding="utf-8-sig")
    return str(p)


def test_missing_and_empty(tmp_path):
    assert read_csv_text(str(tmp_path / "yok.csv")) is None
    e = tmp_path / "e.csv"
    e.write_text("")
    assert read_csv_text(str(e)) is None
    d = CsvViewerDialog(str(tmp_path / "yok.csv"))
    assert "bulunamadı" in d.info.text()


def test_search_sort_columns(tmp_path):
    p = _write(tmp_path, ["r1;Parus major;H1;10;x", "r1;Turdus merula;H2;2;y", "r2;Parus major;;100;z"])
    d = CsvViewerDialog(p)
    assert d.proxy.rowCount() == 3
    assert "3</b> segment" in d.info.text() and "2</b> tür" in d.info.text()
    d.search.setText("PARUS")
    assert d.proxy.rowCount() == 2
    d.search.setText("")
    d.view.sortByColumn(3, Qt.AscendingOrder)  # sayısal: 2,10,100
    vals = [d.proxy.data(d.proxy.index(i, 3)) for i in range(3)]
    assert vals == ["2", "10", "100"]
    # kompakt: Extra gizli, tümü: görünür
    assert d.view.isColumnHidden(4)
    d.chk_all.setChecked(True)
    assert not d.view.isColumnHidden(4)


def test_reload_picks_up_changes(tmp_path):
    p = _write(tmp_path, ["r1;Parus major;H1;1;x"])
    d = CsvViewerDialog(p)
    assert d.proxy.rowCount() == 1
    with open(p, "a", encoding="utf-8-sig") as f:
        f.write("\nr2;Turdus merula;H2;2;y")
    d.reload()
    assert d.proxy.rowCount() == 2


def test_maximize_flag_and_toggle(tmp_path):
    p = _write(tmp_path, ["r1;Parus major;H1;1;x"])
    d = CsvViewerDialog(p)
    assert d.windowFlags() & Qt.WindowMaximizeButtonHint
    d._toggle_fullscreen()
    assert d.isMaximized()
    d._toggle_fullscreen()
    assert not d.isMaximized()
