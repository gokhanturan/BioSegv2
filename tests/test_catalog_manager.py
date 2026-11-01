"""Tanım verileri yöneticisi: doğrulama, CRUD, atomik kayıt + yedek, CSV kullanım sayıları, pencere, canlı yenileme."""
import json
import os
import shutil

import pytest

from src.data.catalog_manager import CatalogStore, CatalogError, KEEP_BACKUPS

SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
FILES = ("bird_species_data.json", "habitat_data.json", "bird_sound_config.json")


@pytest.fixture
def root(tmp_path):
    for f in FILES:
        shutil.copy(os.path.join(SRC, f), tmp_path / f)
    return str(tmp_path)


def _sp(**kw):
    d = dict(order="Passeriformes", family="Turdidae", scientific_name="Turdus testus", turkish_name="Deneme Ardıcı")
    d.update(kw)
    return d


def test_loads_real_files(root):
    s = CatalogStore(root)
    assert len(s.species) > 400 and len(s.habitats) > 100 and s.list_keys()
    assert not s.dirty


def test_species_validation_and_crud(root):
    s = CatalogStore(root)
    n = len(s.species)
    for bad in (_sp(scientific_name=""), _sp(turkish_name=""), _sp(scientific_name="turdus Testus"),
                _sp(scientific_name="Turdus"), _sp(order=""), _sp(scientific_name="Turdus merula")):
        with pytest.raises(CatalogError):
            s.add_species(bad)
    assert len(s.species) == n and not s.dirty
    i = s.add_species(_sp(scientific_name="  Turdus   testus "))
    assert s.species[i]["scientific_name"] == "Turdus testus" and s.dirty
    s.update_species(i, _sp(turkish_name="Başka Ad"))
    assert s.species[i]["turkish_name"] == "Başka Ad"
    s.delete_species(i)
    assert len(s.species) == n


def test_same_turkish_name_is_warning_not_error(root):
    s = CatalogStore(root)
    errs, warns = s.validate_species(_sp(turkish_name=s.species[0]["turkish_name"]))
    assert not errs and warns


def test_update_keeps_own_name_valid(root):
    s = CatalogStore(root)
    errs, _ = s.validate_species(dict(s.species[3]), 3)
    assert not errs


def test_habitat_validation_and_crud(root):
    s = CatalogStore(root)
    n = len(s.habitats)
    h = dict(habitat_ID=s.next_habitat_id(), habitat_tipi="Makilik", kodu="yeni1", utm_zone="36N",
             utm_easting="300000", utm_northing="4100000", lat="37,5", lon="30.5", konum="Isparta")
    for key, val in (("lat", "91"), ("lon", "abc"), ("kodu", ""), ("habitat_ID", "x"), ("utm_easting", "q")):
        with pytest.raises(CatalogError):
            s.add_habitat({**h, key: val})
    with pytest.raises(CatalogError):
        s.add_habitat({**h, "habitat_ID": s.habitats[0]["habitat_ID"]})
    with pytest.raises(CatalogError):
        s.add_habitat({**h, "kodu": s.habitats[0]["kodu"]})
    i = s.add_habitat(h)
    rec = s.habitats[i]
    assert rec["lat"] == 37.5 and rec["google_maps"].endswith("?q=37.5,30.5") and isinstance(rec["habitat_ID"], int)
    s.update_habitat(i, {**h, "konum": "Burdur"})
    assert s.habitats[i]["konum"] == "Burdur"
    s.delete_habitat(i)
    assert len(s.habitats) == n


def test_empty_utm_allowed(root):
    s = CatalogStore(root)
    h = dict(habitat_ID=s.next_habitat_id(), habitat_tipi="Taşlık", kodu="yeni2", utm_zone="", utm_easting="",
             utm_northing="", lat="37", lon="30", konum="Burdur")
    assert s.habitats[s.add_habitat(h)]["utm_easting"] is None


def test_list_crud_move_and_duplicates(root):
    s = CatalogStore(root)
    k = "call_types"
    n = len(s.get_list(k))
    with pytest.raises(CatalogError):
        s.add_list_item(k, "  ")
    with pytest.raises(CatalogError):
        s.add_list_item(k, s.get_list(k)[0].upper())
    s.add_list_item(k, " Yeni  Ötüş ")
    assert s.get_list(k)[-1] == "Yeni Ötüş"
    assert s.move_list_item(k, n, -1) == n - 1 and s.get_list(k)[n - 1] == "Yeni Ötüş"
    assert s.move_list_item(k, 0, -1) == 0
    s.update_list_item(k, n - 1, "Yeni Ötüş 2")
    s.delete_list_item(k, n - 1)
    assert len(s.get_list(k)) == n
    with pytest.raises(CatalogError):
        s.set_text("project", " ")


def test_save_is_atomic_backed_up_and_preserves_other_keys(root):
    s = CatalogStore(root)
    before = json.load(open(os.path.join(root, "bird_sound_config.json"), encoding="utf-8"))
    s.add_list_item("call_types", "Sinyal")
    assert s.save() == ["bird_sound_config.json"] and not s.dirty
    after = json.load(open(os.path.join(root, "bird_sound_config.json"), encoding="utf-8"))
    assert after["call_types"][-1] == "Sinyal"
    assert list(after) == list(before)  # anahtar sırası ve bilinmeyen anahtarlar korunur
    assert {k: v for k, v in after.items() if k != "call_types"} == {k: v for k, v in before.items() if k != "call_types"}
    bk = os.listdir(os.path.join(root, "yedek_json"))
    assert len(bk) == 1 and bk[0].startswith("bird_sound_config_")
    assert not [f for f in os.listdir(root) if f.endswith(".tmp")]
    assert s.save() == []  # değişiklik yoksa yazmaz


def test_unchanged_files_are_not_rewritten(root):
    p = os.path.join(root, "habitat_data.json")
    m0 = open(p, "rb").read()
    s = CatalogStore(root)
    s.add_species(_sp())
    s.save()
    assert open(p, "rb").read() == m0


def test_backup_rotation(root):
    s = CatalogStore(root)
    for i in range(KEEP_BACKUPS + 4):
        s.add_list_item("call_types", f"x{i}")
        s.save()
    assert len(os.listdir(os.path.join(root, "yedek_json"))) == KEEP_BACKUPS


def test_revert(root):
    s = CatalogStore(root)
    n = len(s.species)
    s.add_species(_sp())
    s.revert()
    assert len(s.species) == n and not s.dirty


def test_usage_counts(root):
    pd = pytest.importorskip("pandas")
    s = CatalogStore(root)
    sp = s.species[0]
    df = pd.DataFrame({"Scientific_Name": [sp["scientific_name"], "Baska ad", sp["scientific_name"]],
                       "Turkish_Name": ["x", "y", "z"], "Habitat_ID": ["1", "2", "1"],
                       "Call_Type": ["Ötüş", "Çağrı", "ötüş"]})
    assert s.species_usage(df, sp) == 2 and s.species_usage(None, sp) == 0
    assert s.habitat_usage(df, {"habitat_ID": 1}) == 2
    assert s.list_item_usage(df, "call_types", "Ötüş") == 2
    assert s.list_item_usage(df, "yok_liste", "x") is None


def test_problem_report_finds_known_duplicates(root):
    p = " ".join(CatalogStore(root).species_problems())
    assert "Falco peregrinus" in p and "Jynx torquilla" in p


# ------------------------------------------------------------------ pencere ve canlı yenileme
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_dialog_add_update_delete_save(qapp, root, monkeypatch):
    from src.ui import catalog_dialog as cd
    monkeypatch.setattr(cd.QMessageBox, "question", lambda *a, **k: cd.QMessageBox.Yes)
    monkeypatch.setattr(cd.QMessageBox, "warning", lambda *a, **k: None)
    store = CatalogStore(root)
    n = len(store.species)
    d = cd.CatalogDialog(csv_path=None, store=store)
    assert d.sp_table.rowCount() == n and not d.btn_save.isEnabled()
    for k, v in _sp().items():
        d.sp_edits[k].setText(v)
    d.sp_add.click()
    assert len(store.species) == n + 1 and d.btn_save.isEnabled()
    d.sp_add.click()  # aynı bilimsel ad -> reddedilir
    assert len(store.species) == n + 1
    got = []
    d.saved.connect(lambda: got.append(1))
    d.btn_save.click()
    assert got == [1] and not d.btn_save.isEnabled()
    assert any(s["scientific_name"] == "Turdus testus"
               for s in json.load(open(os.path.join(root, "bird_species_data.json"), encoding="utf-8")))
    d.sp_search.setText("Deneme Ardıcı")
    visible = [r for r in range(d.sp_table.rowCount()) if not d.sp_table.isRowHidden(r)]
    assert len(visible) == 1
    d.sp_table.selectRow(visible[0])
    d.sp_edits["turkish_name"].setText("Deneme Ardıcı 2")
    d.sp_upd.click()
    d.sp_del.click()
    assert len(store.species) == n
    d.btn_save.click()
    d.close()


def test_dialog_habitat_new_uses_next_id_and_list_tab(qapp, root, monkeypatch):
    from src.ui import catalog_dialog as cd
    monkeypatch.setattr(cd.QMessageBox, "question", lambda *a, **k: cd.QMessageBox.Yes)
    store = CatalogStore(root)
    d = cd.CatalogDialog(store=store)
    d.hb_new.click()
    assert d.hb_edits["habitat_ID"].text() == str(store.next_habitat_id())
    k = d.ls_combo.itemData(0)
    n = d.ls_list.count()
    d.ls_edit.setText("Zzz Yeni")
    d.ls_add.click()
    assert d.ls_list.count() == n + 1
    d.ls_list.setCurrentRow(n)
    d.ls_del.click()
    assert d.ls_list.count() == n
    store.revert()
    d.close()


def test_main_window_menu_and_live_reload(qapp, tmp_path, monkeypatch):
    from src.ui.main_window import BirdSoundApp
    import src.data.config as cfgmod
    for f in FILES:
        shutil.copy(os.path.join(SRC, f), tmp_path / f)
    monkeypatch.setattr(cfgmod, "get_project_root", lambda: str(tmp_path))
    win = BirdSoundApp()
    texts = [a.text() for top in win.menuBar().actions() for a in top.menu().actions()] + \
            [b.text() for top in win.menuBar().actions() for a in top.menu().actions() if a.menu() for b in a.menu().actions()]
    assert any("Tanım Verileri Yöneticisi" in t for t in texts)
    s = CatalogStore(str(tmp_path))
    s.add_species(_sp(turkish_name="Canlı Yenileme Kuşu", scientific_name="Zzzus liveus"))
    s.add_list_item("call_types", "Canlı Ötüş")
    s.save()
    assert win.cmb_turkish_name.findText("Canlı Yenileme Kuşu") < 0
    win.cmb_call_type.setCurrentIndex(1)
    keep = win.cmb_call_type.currentText()
    win.reload_catalog_data()
    assert win.cmb_turkish_name.findText("Canlı Yenileme Kuşu") >= 0
    assert win.cmb_background_turkish_name.findText("Canlı Yenileme Kuşu") >= 0
    assert win.cmb_call_type.findText("Canlı Ötüş") >= 0 and win.cmb_call_type.currentText() == keep
    win.cmb_turkish_name.setCurrentText("Canlı Yenileme Kuşu")
    assert win.txt_scientific_name.text() == "Zzzus liveus"


# ------------------------------------------------------------------ sade yapılandırma (ENG anahtarları kalktı)
LEGACY = {
    "call_typesENG": ["Kur", "Çağrı"], "call_types": ["Song", "Call"],
    "background_noiseENG": ["Motor Gürültüsü"], "background_noise": ["Engine"],
    "locationsENG": ["Burdur"], "habitatsENG": ["Forest"], "bird_species": [{"order": "x"}],
    "recordistsENG": ["Gülşah YILMAZ"], "recordists": ["Gökhan TURAN"],
    "projectENG": "Proje TR", "project": "Project EN",
}


def test_shipped_config_is_simple_turkish():
    cfg = json.load(open(os.path.join(SRC, "bird_sound_config.json"), encoding="utf-8"))
    from src.data.config import CONFIG_KEYS
    assert tuple(cfg) == CONFIG_KEYS
    assert not [k for k in cfg if k.endswith("ENG")]
    assert cfg["call_types"][:3] == ["Kur", "Çağrı", "Ötüş"] and "Song" not in cfg["call_types"]


def test_legacy_config_migrates_using_eng_values_and_drops_unused():
    from src.data.config import migrate_legacy_config
    out = migrate_legacy_config(LEGACY)
    assert out == {"call_types": ["Kur", "Çağrı"], "background_noise": ["Motor Gürültüsü"],
                   "recordists": ["Gülşah YILMAZ"], "project": "Proje TR"}
    assert migrate_legacy_config(out) == out  # yeni biçim değişmez (idempotent)


def test_load_config_reads_legacy_and_new_files(tmp_path, monkeypatch):
    import src.data.config as c
    monkeypatch.setattr(c, "get_project_root", lambda: str(tmp_path))
    (tmp_path / "bird_sound_config.json").write_text(json.dumps(LEGACY, ensure_ascii=False), encoding="utf-8")
    cfg = c.load_config()
    assert cfg["call_types"] == ["Kur", "Çağrı"] and cfg["project"] == "Proje TR"
    assert cfg["microphones"] == c.DEFAULT_CONFIG["microphones"]  # eksik anahtar varsayılandan gelir
    assert not [k for k in cfg if k.endswith("ENG")]


def test_catalog_store_saves_simplified_form_from_legacy_file(tmp_path):
    (tmp_path / "bird_sound_config.json").write_text(json.dumps(LEGACY, ensure_ascii=False), encoding="utf-8")
    s = CatalogStore(str(tmp_path))
    s.add_list_item("call_types", "Yeni")
    s.save()
    saved = json.load(open(tmp_path / "bird_sound_config.json", encoding="utf-8"))
    assert saved["call_types"] == ["Kur", "Çağrı", "Yeni"] and not [k for k in saved if k.endswith("ENG")]
    assert "bird_species" not in saved
    assert os.listdir(tmp_path / "yedek_json")  # eski biçimin yedeği alındı


def test_habitat_table_shows_every_form_field_and_sorts_ids_numerically(qapp, root):
    from PyQt5.QtCore import Qt
    from src.ui import catalog_dialog as cd
    d = cd.CatalogDialog(store=CatalogStore(root))
    heads = [d.hb_table.horizontalHeaderItem(c).text() for c in range(d.hb_table.columnCount())]
    for key in ("utm_easting", "utm_northing", "google_maps", "utm_zone", "lat", "lon"):
        assert cd._HABITAT_LABELS[key] in heads
    assert d.hb_table.columnCount() == len(cd.HABITAT_FIELDS)  # form ile tablo aynı alan sayısı
    d.hb_table.sortItems(0, Qt.AscendingOrder)
    ids = [int(d.hb_table.item(r, 0).text()) for r in range(d.hb_table.rowCount())]
    assert ids == sorted(ids)
    d.close()
