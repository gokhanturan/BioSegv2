"""Ana pencere: Analiz sekmesi metinleri doğru mu, tez adı alt çubukta mı (TensorFlow gerekmez)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
_APP = []


@pytest.fixture(scope="module")
def win():
    _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    from src.ui.main_window import BirdSoundApp
    return BirdSoundApp()


def _card_text(btn):
    return " ".join(l.text() for l in btn.findChildren(QtWidgets.QLabel))


def test_footer_carries_every_item_from_the_thesis_cover(win):
    from src.ui import app_info as ai
    box = win.statusBar().findChild(QtWidgets.QWidget, "thesisFooter")
    assert box is not None
    txt = " ".join(l.text() for l in box.findChildren(QtWidgets.QLabel))
    for item in (ai.UNIVERSITY, ai.INSTITUTE, ai.DEPARTMENT, ai.THESIS_TITLE, ai.THESIS_TYPE,
                 ai.AUTHOR, ai.ADVISOR, ai.SECOND_ADVISOR, ai.PLACE_YEAR):
        assert item in txt, item
    # kapak sayfasındaki kesin yazımlar
    assert ai.THESIS_TITLE == "YAPAY ZEKA İLE KUŞ SESLERİNİN AKUSTİK ANALİZİ VE TÜR ZENGİNLİĞİNİN BELİRLENMESİ"
    assert ai.ADVISOR == "Prof. Dr. Ecir UĞUR KÜÇÜKSİLLE" and ai.SECOND_ADVISOR == "Doç. Dr. Halil SÜEL" and ai.PLACE_YEAR == "ISPARTA-2026"


def test_data_quality_card_no_longer_claims_read_only(win):
    t = _card_text(win.btn_data_quality)
    assert "CSV'yi değiştirmez" not in t and "iki onayla" in t


def test_classifier_group_title_differs_from_step_name(win):
    boxes = {b.title() for b in win.findChildren(QtWidgets.QGroupBox)}
    assert "🧠 Tür Sınıflandırma" in boxes and "🧠 Sınıflandırıcı Eğitimi" not in boxes
    assert "2. Tür Sınıflandırma Eğitimi" in _card_text(win.btn_train_classifier)
    assert "çapraz doğrulama" in _card_text(win.btn_train_classifier)


def test_footer_layout_left_two_lines_logo_center_right_two_lines(win, tmp_path, monkeypatch):
    from PyQt5.QtGui import QPixmap, QColor
    from src.ui import app_info as ai, main_window as mw
    pm = QPixmap(40, 40)
    pm.fill(QColor("red"))
    pm.save(str(tmp_path / ai.LOGO_FILE))
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    from src.ui.main_window import BirdSoundApp
    w = BirdSoundApp()
    box = w.statusBar().findChild(QtWidgets.QWidget, "thesisFooter")
    items = [box.layout().itemAt(i).widget() for i in range(box.layout().count())]
    assert len(items) == 3
    left, logo, right = items
    assert ai.UNIVERSITY in left.text() and ai.THESIS_TITLE in left.text()
    assert ai.AUTHOR in right.text() and ai.SECOND_ADVISOR in right.text()
    assert ai.AUTHOR not in left.text() and ai.UNIVERSITY not in right.text()
    assert logo.pixmap() is not None and not logo.pixmap().isNull()


def test_footer_works_without_logo_file(win, tmp_path, monkeypatch):
    from src.ui import main_window as mw
    from src.ui import app_info as ai
    monkeypatch.setattr(mw, "csv_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(mw, "src_root_dir", lambda: str(tmp_path))
    from src.ui.main_window import BirdSoundApp
    box = BirdSoundApp().statusBar().findChild(QtWidgets.QWidget, "thesisFooter")
    assert box.layout().count() == 2 and ai.find_logo([str(tmp_path)]) is None


@pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 2.0])
def test_logo_is_scaled_to_real_device_pixels(win, dpr):
    from PyQt5.QtGui import QPixmap, QColor
    from src.ui.main_window import logo_pixmap
    import tempfile
    big = QPixmap(1300, 1300)
    big.fill(QColor("red"))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "l.png")
        big.save(path)
        pm = logo_pixmap(path, 72, dpr)
    assert pm.width() == round(72 * dpr) and pm.devicePixelRatio() == dpr   # çizimde yeniden ölçeklenmez
    assert abs(pm.width() / pm.devicePixelRatio() - 72) < 1


def test_logo_pixmap_returns_none_for_missing_file(win):
    from src.ui.main_window import logo_pixmap
    assert logo_pixmap("/yok/yok.png", 72) is None


def test_footer_hidden_only_on_labeling_and_detection_tabs(win):
    tabs = win.main_tabs
    win.show()
    hidden = win._footer_hidden_tabs
    assert win.label_tab in hidden and len(hidden) == 2
    for i in range(tabs.count()):
        tabs.setCurrentIndex(i)
        assert win.statusBar().isVisible() == (tabs.widget(i) not in hidden)


def test_labeling_form_is_never_squeezed_below_its_minimum_height(win):
    from PyQt5.QtWidgets import QApplication
    win.resize(1400, 760)
    win.show()
    win.main_tabs.setCurrentWidget(win.label_tab)
    QApplication.processEvents()
    form = win.label_tab.widget()
    assert form.height() >= form.minimumSizeHint().height()
    assert win.label_tab.verticalScrollBar().maximum() > 0
