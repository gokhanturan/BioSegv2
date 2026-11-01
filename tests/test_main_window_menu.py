"""Ana pencere menüleri: iş akışıyla uyum, çakışmasız kısayollar, kısayol listesi menüden üretilir, Hakkında güncel."""
import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
_APP = []


@pytest.fixture(scope="module")
def win():
    _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    from src.ui.main_window import BirdSoundApp
    return BirdSoundApp()


def _walk(menu, path=()):
    for a in menu.actions():
        if a.isSeparator():
            continue
        if a.menu() is not None:
            yield from _walk(a.menu(), path + (a.text(),))
        else:
            yield path, a


def _all(win):
    out = []
    for top in win.menuBar().actions():
        out += list(_walk(top.menu(), (top.text(),)))
    return out


def test_every_screen_is_reachable_from_menus(win):
    texts = " | ".join(a.text() for _p, a in _all(win))
    for name in ("Veri Seti Özeti", "Habitat Bilgileri", "Tür Zenginliği Belirleme", "Konum Haritası",
                 "Veri Kalitesi Denetimi", "Sınıflandırma Klasörleri", "Tür Sınıflandırma Eğitimi",
                 "BirdNET 2.4 Eğitimi", "Perch 2.0 Eğitimi", "BirdNET + Perch (birleşik) Eğitimi",
                 "Model Karşılaştırma", "Başarı Sınırı Analizi", "CSV Görüntüle", "Kuş Ötüşlerini Tespit Et"):
        assert name in texts, name


def test_menu_follows_the_analysis_tab_workflow(win):
    paths = {a.text(): p for p, a in _all(win)}
    sub = lambda name: next(p for t, p in paths.items() if name in t)
    # veri seti bakımı (denetim, tanım verileri) "Veri Seti" menüsünde; analizler "Analiz > Veri Seti Analizi" altında
    assert "Veri Seti" in sub("Veri Kalitesi Denetimi")[0] and "Veri Seti" in sub("Tanım Verileri Yöneticisi")[0]
    assert "Analiz" in sub("Veri Seti Özeti")[0] and "Veri Seti Analizi" in sub("Konum Haritası")[-1]
    assert "Tür Sınıflandırma" in sub("Başarı Sınırı Analizi")[-1]
    assert "Tür Sınıflandırma" in sub("Sınıflandırma Klasörleri")[-1]
    # sınıflandırma klasörleri Veri Seti menüsünden taşındı
    assert "Veri Seti" not in sub("Sınıflandırma Klasörleri")[0]
    # iş akışı sırası: 1 klasörler, 2 eğitim, 3 başarı sınırı
    order = [a.text() for p, a in _all(win) if p and "Tür Sınıflandırma" in p[-1]]
    i1 = next(i for i, t in enumerate(order) if "1." in t)
    i2 = next(i for i, t in enumerate(order) if "2." in t)
    i3 = next(i for i, t in enumerate(order) if "3." in t)
    assert i1 < i2 < i3


def test_no_duplicate_shortcuts_and_all_three_models_have_entries(win):
    seqs = [a.shortcut().toString() for _p, a in _all(win) if not a.shortcut().isEmpty()]
    assert len(seqs) == len(set(seqs)), [s for s in seqs if seqs.count(s) > 1]
    for key in ("Ctrl+T", "Ctrl+K", "Ctrl+Shift+K", "Ctrl+Shift+D", "Ctrl+Shift+V", "F1", "F2"):
        assert key in seqs, key


def test_shortcut_help_is_generated_from_real_menu_actions(win):
    html = win.shortcuts_html()
    for _p, a in _all(win):
        if not a.shortcut().isEmpty():
            assert win._shortcut_label(a.shortcut()) in html
            assert win._action_label(a.text()) in html
    assert "Ctrl+Shift+K" in html and "Model Karşılaştırma" in html
    assert "Boşluk" in html and "Ctrl+←" in html and "<td><b>Space" not in html
    assert not re.search(r"<td>[^<]*\.\.\.<", html)       # '...' temizlenmiş


def test_about_mentions_all_three_models_and_current_features(win, monkeypatch):
    got = {}
    monkeypatch.setattr(QtWidgets.QMessageBox, "about", staticmethod(lambda p, t, text: got.setdefault("t", text)))
    win.show_about()
    t = got["t"]
    for must in ("BirdNET 2.4", "Perch 2.0", "ikisinin birleşimi", "veri kalitesi denetimi", "başarı sınırı",
                 "518 türlük", "standart örnekleme koşullarında"):
        assert must in t, must
    assert "yalnızca BirdNET" not in t and "spektrogram hesaplama" not in t
    # segmentasyon/tespit yalnızca BirdNET'tir; Perch yalnızca sınıflandırmada
    assert "BirdNET Destekli Biyoakustik Segmentasyon" in t and "Perch Destekli" not in t
    assert "Arka plan tür bilgileri" in t and "Fon" not in t and "fon " not in t


def test_analysis_cards_do_not_promise_removed_cv_button(win):
    texts = " ".join(l.text() for b in (win.btn_classification_diagnosis, win.btn_train_classifier)
                     for l in b.findChildren(QtWidgets.QLabel))
    assert "buradan çalıştırabilirsiniz" not in texts
    assert "BirdNET + Perch" in texts and "eğitim yapmaz" in texts


def test_shortcut_list_follows_menu_order_and_has_no_bird_emoji(win):
    """F2 listesi menü sırasını izler (Ctrl+B en üstte); menü simgelerinde Windows'ta bozuk görünen 🐦 yok."""
    rows = dict(win.shortcut_rows())
    analiz = next(v for k, v in rows.items() if "Analiz" in k)
    keys = [k for k, _ in analiz]
    assert keys[0] == "Ctrl+B" and keys.index("Ctrl+B") < keys.index("Ctrl+U") < keys.index("Ctrl+K")
    assert "🐦" not in "".join(a.text() for a in win.findChildren(QtWidgets.QAction))


def test_every_menu_action_has_a_shortcut(win):
    missing = [a.text() for _p, a in _all(win) if a.shortcut().isEmpty()]
    assert not missing, missing


def test_new_shortcuts_and_ribbon_groups(win):
    keys = {a.text(): a.shortcut().toString() for _p, a in _all(win)}
    pick = lambda name: next(v for t, v in keys.items() if name in t)
    assert pick("Veri Kalitesi Denetimi") == "Ctrl+Shift+C" and pick("Tanım Verileri Yöneticisi") == "Ctrl+Y"
    assert [pick(n) for n in ("BirdNET 2.4 Eğitimi", "Perch 2.0 Eğitimi", "BirdNET + Perch (birleşik) Eğitimi")] == ["F5", "F6", "F7"]
    boxes = {b.title(): b for b in win.main_tabs.widget(win.main_tabs.count() - 1).findChildren(QtWidgets.QGroupBox)}
    manage = next(b for t, b in boxes.items() if "Veri Seti Yönetimi" in t)
    analysis = next(b for t, b in boxes.items() if "Veri Seti Analizi" in t)
    for btn in (win.btn_csv_viewer, win.btn_data_quality):
        assert manage.isAncestorOf(btn)
    assert not manage.isAncestorOf(win.btn_catalog_manager)
    # Tanım Verileri Yöneticisi: ilk sekme "Ön Hazırlık"ta; uygulama yine Ses Dosyası sekmesinde açılır
    tabs = win.main_tabs
    assert "Ön Hazırlık" in tabs.tabText(0) and "Ses Dosyası" in tabs.tabText(1)
    assert tabs.widget(0).isAncestorOf(win.btn_catalog_manager)
    assert "Ses Dosyası" in tabs.tabText(tabs.currentIndex())
    for btn in (win.btn_dataset_summary, win.btn_habitat_info, win.btn_species_richness, win.btn_location_map):
        assert analysis.isAncestorOf(btn)


def test_project_name_is_read_only_and_has_no_save_button(win):
    assert win.txt_project.isReadOnly() and not hasattr(win, "btn_save_project")
    assert not hasattr(win, "update_project_info")
