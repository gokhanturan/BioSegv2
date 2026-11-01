"""Tanım Verileri Yöneticisi: tür veri tabanı, habitat noktaları ve açılır listeler (ekle/güncelle/sil/listele).

Değişiklikler 'Kaydet' ile diske yazılır (önce yedek alınır) ve ana pencere canlı yenilenir (saved sinyali).
CSV'ye dokunulmaz; silme/değiştirme öncesi etkilenecek CSV satır sayısı gösterilir.
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QTableWidget,
                             QTableWidgetItem, QAbstractItemView, QLineEdit, QPushButton, QLabel,
                             QFormLayout, QMessageBox, QComboBox, QListWidget, QListWidgetItem,
                             QHeaderView, QGroupBox, QPlainTextEdit)

from ..data.catalog_manager import (CatalogStore, CatalogError, SPECIES_FIELDS, HABITAT_FIELDS,
                                    CONFIG_TEXT_FIELDS)
from . import theme

_SPECIES_LABELS = {"order": "Takım (order)", "family": "Familya (family)",
                   "scientific_name": "Bilimsel ad", "turkish_name": "Türkçe ad"}
_HABITAT_LABELS = {"habitat_ID": "Habitat ID", "habitat_tipi": "Habitat tipi", "kodu": "Kod",
                   "utm_zone": "UTM bölgesi", "utm_easting": "UTM doğu", "utm_northing": "UTM kuzey",
                   "lat": "Enlem (lat)", "lon": "Boylam (lon)", "google_maps": "Google Maps bağlantısı",
                   "konum": "Konum (il)"}


class _SortItem(QTableWidgetItem):
    """Sayı gibi görünen hücreleri sayısal sıralar (137, 138, 139, 14 yerine 14 ... 137 ...)."""

    def __lt__(self, other):
        try:
            return float(self.text().replace(",", ".")) < float(other.text().replace(",", "."))
        except (ValueError, AttributeError):
            return self.text().casefold() < other.text().casefold()


class CatalogDialog(QDialog):
    saved = pyqtSignal()  # diske yazıldı; ana pencere verileri yeniden yüklemeli

    def __init__(self, csv_path=None, parent=None, store=None):
        super().__init__(parent)
        self.store = store or CatalogStore()
        self.csv_path = csv_path
        self._df = None
        self._df_loaded = False
        self.setWindowTitle("🗂 Tanım Verileri Yöneticisi")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.resize(1280, 760)
        lay = QVBoxLayout(self)
        theme.add_hero(lay, self.windowTitle(),
                       "Tür veri tabanı, habitat noktaları ve açılır listeler. Kaydetmeden önce değişiklikler yalnızca bellektedir.")
        self.tabs = QTabWidget()
        # Genel stil kalın/büyük yazı kullanır; Qt bunu sekme genişliğine katmadığı için metin kesiliyordu.
        # Ana penceredeki çözümle aynı: sekme yazı tipini elle ver, kısaltmayı kapat, en küçük genişlik tanımla.
        from PyQt5.QtGui import QFont
        bar_font = QFont(self.font())
        bar_font.setPixelSize(15)
        bar_font.setBold(True)
        bar = self.tabs.tabBar()
        bar.setFont(bar_font)
        bar.setElideMode(Qt.ElideNone)
        bar.setUsesScrollButtons(False)
        bar.setExpanding(False)
        self.tabs.setStyleSheet("QTabBar::tab { min-width: 140px; padding: 8px 24px; }")
        lay.addWidget(self.tabs, 1)
        self._build_species_tab()
        self._build_habitat_tab()
        self._build_lists_tab()
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)
        row = QHBoxLayout()
        self.btn_save = QPushButton("💾 Kaydet")
        self.btn_save.setToolTip("Değişen JSON dosyalarını yazar; önce yedek_json/ klasörüne tarih damgalı yedek alır.")
        self.btn_revert = QPushButton("↩ Değişiklikleri Geri Al")
        self.btn_close = QPushButton("Kapat")
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_revert)
        row.addStretch()
        row.addWidget(self.btn_close)
        lay.addLayout(row)
        self.btn_save.clicked.connect(self.save)
        self.btn_revert.clicked.connect(self.revert)
        self.btn_close.clicked.connect(self.close)
        self.refresh_all()

    # ------------------------------------------------------------------ ortak
    def df(self):
        if not self._df_loaded:
            self._df_loaded = True
            try:
                from ..data import data_quality
                self._df = data_quality.read_text(self.csv_path) if self.csv_path else None
            except Exception:
                self._df = None
        return self._df

    def _set_status(self, text, ok=True):
        self.lbl_status.setStyleSheet("color:#1b6e3c;" if ok else "color:#b3261e;")
        self.lbl_status.setText(text)

    def _update_dirty_ui(self):
        d = self.store.dirty_kinds()
        names = {"species": "türler", "habitat": "habitat noktaları", "config": "listeler"}
        self.btn_save.setEnabled(bool(d))
        self.btn_revert.setEnabled(bool(d))
        self.btn_save.setText("💾 Kaydet" + (f" ({', '.join(names[k] for k in d)})" if d else ""))

    def _make_table(self, headers):
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSortingEnabled(True)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        t.horizontalHeader().setStretchLastSection(True)
        return t

    @staticmethod
    def _selected_index(table):
        rows = table.selectionModel().selectedRows()
        if not rows:
            return None
        it = table.item(rows[0].row(), 0)
        return it.data(Qt.UserRole) if it is not None else None

    @staticmethod
    def _filter(table, text):
        t = text.casefold().strip()
        for r in range(table.rowCount()):
            hit = not t or any(t in (table.item(r, c).text().casefold() if table.item(r, c) else "")
                               for c in range(table.columnCount()))
            table.setRowHidden(r, not hit)

    def _fill_table(self, table, rows, keep_index=None):
        table.blockSignals(True)
        table.clearSelection()
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        sel_row = None
        for r, (idx, values) in enumerate(rows):
            for c, v in enumerate(values):
                item = _SortItem("" if v is None else str(v))
                if c == 0:
                    item.setData(Qt.UserRole, idx)
                table.setItem(r, c, item)
            if idx == keep_index:
                sel_row = r
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()
        for c in range(table.columnCount() - 1):
            table.setColumnWidth(c, min(max(table.columnWidth(c), 70), 280))
        table.blockSignals(False)
        if sel_row is not None:
            table.selectRow(sel_row)
        else:
            table.itemSelectionChanged.emit()

    def _confirm(self, title, text):
        return QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) == QMessageBox.Yes

    def _warn(self, text):
        QMessageBox.warning(self, "Kayıt yapılamadı", text)

    # ------------------------------------------------------------------ türler
    def _build_species_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        top = QHBoxLayout()
        self.sp_search = QLineEdit()
        self.sp_search.setPlaceholderText("Ara (Türkçe ad, bilimsel ad, familya, takım)…")
        self.sp_count = QLabel("")
        self.sp_problems = QPushButton("⚠ Tutarsızlıkları göster")
        top.addWidget(self.sp_search, 1)
        top.addWidget(self.sp_count)
        top.addWidget(self.sp_problems)
        v.addLayout(top)
        self.sp_table = self._make_table([_SPECIES_LABELS[k] for k in SPECIES_FIELDS])
        v.addWidget(self.sp_table, 1)
        box = QGroupBox("Tür kaydı")
        form = QFormLayout(box)
        self.sp_edits = {}
        for k in SPECIES_FIELDS:
            e = QLineEdit()
            self.sp_edits[k] = e
            form.addRow(_SPECIES_LABELS[k] + ":", e)
        self.sp_edits["scientific_name"].setPlaceholderText("Turdus merula")
        v.addWidget(box)
        btns = QHBoxLayout()
        self.sp_new = QPushButton("🆕 Yeni (formu temizle)")
        self.sp_add = QPushButton("➕ Ekle")
        self.sp_upd = QPushButton("✏ Seçiliyi Güncelle")
        self.sp_del = QPushButton("🗑 Seçiliyi Sil")
        for b in (self.sp_new, self.sp_add, self.sp_upd, self.sp_del):
            btns.addWidget(b)
        btns.addStretch()
        v.addLayout(btns)
        self.tabs.addTab(w, "🐦 Kuş Türleri")
        self.sp_search.textChanged.connect(lambda t: self._filter(self.sp_table, t))
        self.sp_table.itemSelectionChanged.connect(self._species_selected)
        self.sp_new.clicked.connect(self._species_new)
        self.sp_add.clicked.connect(self._species_add)
        self.sp_upd.clicked.connect(self._species_update)
        self.sp_del.clicked.connect(self._species_delete)
        self.sp_problems.clicked.connect(self._species_show_problems)

    def _species_form(self):
        return {k: e.text() for k, e in self.sp_edits.items()}

    def refresh_species(self, keep=None):
        rows = [(i, [s.get(k, "") for k in SPECIES_FIELDS]) for i, s in enumerate(self.store.species)]
        self._fill_table(self.sp_table, rows, keep)
        self.sp_count.setText(f"{len(rows)} tür")
        self._filter(self.sp_table, self.sp_search.text())
        self._update_dirty_ui()

    def _species_selected(self):
        i = self._selected_index(self.sp_table)
        has = i is not None and i < len(self.store.species)
        self.sp_upd.setEnabled(has)
        self.sp_del.setEnabled(has)
        if has:
            for k, e in self.sp_edits.items():
                e.setText(str(self.store.species[i].get(k, "")))

    def _species_new(self):
        self.sp_table.clearSelection()
        for e in self.sp_edits.values():
            e.clear()
        self.sp_edits["turkish_name"].setFocus()

    def _species_warn_confirm(self, rec, index=None):
        errs, warns = self.store.validate_species(rec, index)
        if errs:
            self._warn("\n".join(errs))
            return False
        if warns:
            return self._confirm("Uyarı", "\n\n".join(warns) + "\n\nYine de kaydedilsin mi?")
        return True

    def _species_add(self):
        rec = self._species_form()
        if not self._species_warn_confirm(rec):
            return
        try:
            i = self.store.add_species(rec)
        except CatalogError as e:
            return self._warn(str(e))
        self.refresh_species(i)
        self._set_status(f"Eklendi: {rec['scientific_name']}. Kalıcı olması için 💾 Kaydet'e basın.")

    def _species_update(self):
        i = self._selected_index(self.sp_table)
        if i is None:
            return
        old, rec = dict(self.store.species[i]), self._species_form()
        if not self._species_warn_confirm(rec, i):
            return
        n = self.store.species_usage(self.df(), old)
        renamed = (old.get("scientific_name") != rec["scientific_name"].strip()
                   or old.get("turkish_name") != rec["turkish_name"].strip())
        if n and renamed and not self._confirm(
                "Ad değişiyor",
                f"Bu türün CSV'de {n} segmenti var. Ad değişikliği CSV'deki eski etiketleri DEĞİŞTİRMEZ; "
                "bu segmentler eski adla kalır (ad tutarsızlığı doğar).\n\nDevam edilsin mi?"):
            return
        try:
            self.store.update_species(i, rec)
        except CatalogError as e:
            return self._warn(str(e))
        self.refresh_species(i)
        self._set_status(f"Güncellendi: {rec['scientific_name']}. Kalıcı olması için 💾 Kaydet'e basın.")

    def _species_delete(self):
        i = self._selected_index(self.sp_table)
        if i is None:
            return
        rec = self.store.species[i]
        n = self.store.species_usage(self.df(), rec)
        msg = f"'{rec.get('turkish_name')} ({rec.get('scientific_name')})' silinsin mi?"
        if n:
            msg += (f"\n\nBu türe ait CSV'de {n} segment var. Segmentler silinmez, ancak tür artık "
                    "açılır listede çıkmaz ve etiketlemede seçilemez.")
        if not self._confirm("Tür sil", msg):
            return
        self.store.delete_species(i)
        self._species_new()
        self.refresh_species()
        self._set_status("Silindi. Kalıcı olması için 💾 Kaydet'e basın (yedek alınır).")

    def _species_show_problems(self):
        p = self.store.species_problems()
        QMessageBox.information(self, "Tür veri tabanı denetimi",
                                "\n".join(p) if p else "Yinelenen bilimsel/Türkçe ad veya şüpheli biçim yok.")

    # ------------------------------------------------------------------ habitat
    def _build_habitat_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        top = QHBoxLayout()
        self.hb_search = QLineEdit()
        self.hb_search.setPlaceholderText("Ara (ID, kod, habitat tipi, il)…")
        self.hb_count = QLabel("")
        top.addWidget(self.hb_search, 1)
        top.addWidget(self.hb_count)
        v.addLayout(top)
        show = ("habitat_ID", "kodu", "habitat_tipi", "konum", "utm_zone", "utm_easting", "utm_northing",
                "lat", "lon", "google_maps")
        self._hb_cols = show
        self.hb_table = self._make_table([_HABITAT_LABELS[k] for k in show])
        self.hb_table.setMinimumHeight(240)
        v.addWidget(self.hb_table, 1)
        box = QGroupBox("Habitat noktası")
        cols = QHBoxLayout(box)
        left, right = QFormLayout(), QFormLayout()
        cols.addLayout(left, 1)
        cols.addLayout(right, 1)
        self.hb_edits = {}
        keys = [k for k in HABITAT_FIELDS if k != "google_maps"]
        for n, k in enumerate(keys):
            e = QLineEdit()
            self.hb_edits[k] = e
            (left if n < 5 else right).addRow(_HABITAT_LABELS[k] + ":", e)
        self.hb_types = QComboBox()
        self.hb_types.setToolTip("Var olan habitat tiplerinden birini seçerek yazım birliğini koruyun.")
        self.hb_types.activated.connect(lambda _i: self.hb_edits["habitat_tipi"].setText(self.hb_types.currentText()))
        right.addRow("Mevcut tipler:", self.hb_types)
        self.hb_maps = QLabel("Google Maps bağlantısı enlem/boylamdan otomatik üretilir.")
        right.addRow("", self.hb_maps)
        v.addWidget(box)
        btns = QHBoxLayout()
        self.hb_new = QPushButton("🆕 Yeni (sonraki ID)")
        self.hb_add = QPushButton("➕ Ekle")
        self.hb_upd = QPushButton("✏ Seçiliyi Güncelle")
        self.hb_del = QPushButton("🗑 Seçiliyi Sil")
        for b in (self.hb_new, self.hb_add, self.hb_upd, self.hb_del):
            btns.addWidget(b)
        btns.addStretch()
        v.addLayout(btns)
        self.tabs.addTab(w, "📍 Habitat Noktaları")
        self.hb_search.textChanged.connect(lambda t: self._filter(self.hb_table, t))
        self.hb_table.itemSelectionChanged.connect(self._habitat_selected)
        self.hb_new.clicked.connect(self._habitat_new)
        self.hb_add.clicked.connect(self._habitat_add)
        self.hb_upd.clicked.connect(self._habitat_update)
        self.hb_del.clicked.connect(self._habitat_delete)

    def _habitat_form(self):
        return {k: e.text() for k, e in self.hb_edits.items()}

    def refresh_habitats(self, keep=None):
        rows = [(i, [h.get(k, "") for k in self._hb_cols]) for i, h in enumerate(self.store.habitats)]
        self._fill_table(self.hb_table, rows, keep)
        self.hb_count.setText(f"{len(rows)} nokta")
        self.hb_types.clear()
        self.hb_types.addItems(self.store.habitat_types())
        self._filter(self.hb_table, self.hb_search.text())
        self._update_dirty_ui()

    def _habitat_selected(self):
        i = self._selected_index(self.hb_table)
        has = i is not None and i < len(self.store.habitats)
        self.hb_upd.setEnabled(has)
        self.hb_del.setEnabled(has)
        if has:
            h = self.store.habitats[i]
            for k, e in self.hb_edits.items():
                v = h.get(k, "")
                e.setText("" if v is None else str(v))

    def _habitat_new(self):
        self.hb_table.clearSelection()
        for e in self.hb_edits.values():
            e.clear()
        self.hb_edits["habitat_ID"].setText(str(self.store.next_habitat_id()))
        self.hb_edits["kodu"].setFocus()

    def _habitat_add(self):
        rec = self._habitat_form()
        errs, _ = self.store.validate_habitat(rec)
        if errs:
            return self._warn("\n".join(errs))
        i = self.store.add_habitat(rec)
        self.refresh_habitats(i)
        self._set_status(f"Eklendi: habitat {rec['habitat_ID']} ({rec['kodu']}). Kalıcı olması için 💾 Kaydet'e basın.")

    def _habitat_update(self):
        i = self._selected_index(self.hb_table)
        if i is None:
            return
        old, rec = self.store.habitats[i], self._habitat_form()
        errs, _ = self.store.validate_habitat(rec, i)
        if errs:
            return self._warn("\n".join(errs))
        n = self.store.habitat_usage(self.df(), old)
        if n and str(old.get("habitat_ID")) != rec["habitat_ID"].strip() and not self._confirm(
                "Habitat ID değişiyor",
                f"Bu habitat ID'si CSV'de {n} segmentte kullanılıyor. ID değişikliği CSV'deki eski ID'yi "
                "DEĞİŞTİRMEZ; bu segmentlerin habitat bilgisi kopar.\n\nDevam edilsin mi?"):
            return
        self.store.update_habitat(i, rec)
        self.refresh_habitats(i)
        self._set_status(f"Güncellendi: habitat {rec['habitat_ID']}. Kalıcı olması için 💾 Kaydet'e basın.")

    def _habitat_delete(self):
        i = self._selected_index(self.hb_table)
        if i is None:
            return
        h = self.store.habitats[i]
        n = self.store.habitat_usage(self.df(), h)
        msg = f"Habitat {h.get('habitat_ID')} ({h.get('kodu')}, {h.get('konum')}) silinsin mi?"
        if n:
            msg += (f"\n\nCSV'de bu ID ile {n} segment var. Segmentler silinmez; ancak habitat bilgisi, "
                    "konum haritası ve tür zenginliği hesapları bu noktayı bulamaz. Silmek yerine çoğu zaman "
                    "güncellemek daha güvenlidir.")
        if not self._confirm("Habitat sil", msg):
            return
        self.store.delete_habitat(i)
        self._habitat_new()
        self.refresh_habitats()
        self._set_status("Silindi. Kalıcı olması için 💾 Kaydet'e basın (yedek alınır).")

    # ------------------------------------------------------------------ listeler
    def _build_lists_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        sel = QHBoxLayout()
        sel.addWidget(QLabel("Liste:"))
        self.ls_combo = QComboBox()
        self.ls_combo.setMinimumWidth(320)
        sel.addWidget(self.ls_combo)
        self.ls_note = QLabel("")
        self.ls_note.setWordWrap(True)
        sel.addWidget(self.ls_note, 1)
        v.addLayout(sel)
        mid = QHBoxLayout()
        self.ls_list = QListWidget()
        mid.addWidget(self.ls_list, 1)
        side = QVBoxLayout()
        self.ls_up = QPushButton("⬆ Yukarı")
        self.ls_down = QPushButton("⬇ Aşağı")
        side.addWidget(self.ls_up)
        side.addWidget(self.ls_down)
        side.addStretch()
        mid.addLayout(side)
        v.addLayout(mid, 1)
        row = QHBoxLayout()
        self.ls_edit = QLineEdit()
        self.ls_edit.setPlaceholderText("Değer…")
        self.ls_add = QPushButton("➕ Ekle")
        self.ls_upd = QPushButton("✏ Seçiliyi Güncelle")
        self.ls_del = QPushButton("🗑 Seçiliyi Sil")
        row.addWidget(self.ls_edit, 1)
        for b in (self.ls_add, self.ls_upd, self.ls_del):
            row.addWidget(b)
        v.addLayout(row)
        box = QGroupBox("Proje adı")
        f = QFormLayout(box)
        self.ls_text = {}
        for k, label in CONFIG_TEXT_FIELDS.items():
            e = QLineEdit()
            self.ls_text[k] = e
            f.addRow(label + ":", e)
            e.editingFinished.connect(lambda k=k, e=e: self._text_changed(k, e))
        v.addWidget(box)
        self.tabs.addTab(w, "📋 Listeler")
        self.ls_combo.currentIndexChanged.connect(self.refresh_list)
        self.ls_list.currentRowChanged.connect(self._list_selected)
        self.ls_add.clicked.connect(self._list_add)
        self.ls_upd.clicked.connect(self._list_update)
        self.ls_del.clicked.connect(self._list_delete)
        self.ls_up.clicked.connect(lambda: self._list_move(-1))
        self.ls_down.clicked.connect(lambda: self._list_move(1))

    def _cur_key(self):
        return self.ls_combo.currentData()

    def refresh_list_combo(self):
        cur = self._cur_key()
        self.ls_combo.blockSignals(True)
        self.ls_combo.clear()
        for k in self.store.list_keys():
            used = self.store.list_is_used(k)
            self.ls_combo.addItem(f"{self.store.list_title(k)}  [{k}]" + ("" if used else "  — arayüzde kullanılmıyor"), k)
        if cur:
            j = self.ls_combo.findData(cur)
            if j >= 0:
                self.ls_combo.setCurrentIndex(j)
        self.ls_combo.blockSignals(False)
        for k, e in self.ls_text.items():
            e.setText(self.store.get_text(k))

    def refresh_list(self, *_):
        k = self._cur_key()
        self.ls_list.clear()
        if not k:
            return
        for v in self.store.get_list(k):
            self.ls_list.addItem(QListWidgetItem(v))
        if self.store.list_is_used(k):
            self.ls_note.setText("Bu liste etiketleme formundaki açılır listeyi besler.")
        else:
            self.ls_note.setText("Bu liste şu an arayüzde kullanılmıyor (eski/yedek). Düzenlemek ana pencereyi etkilemez.")
        self._update_dirty_ui()

    def _list_selected(self, row):
        k = self._cur_key()
        if k and 0 <= row < len(self.store.get_list(k)):
            self.ls_edit.setText(self.store.get_list(k)[row])

    def _text_changed(self, k, e):
        if _same(e.text(), self.store.get_text(k)):
            return
        try:
            self.store.set_text(k, e.text())
            self._set_status("Proje adı değişti. Kalıcı olması için 💾 Kaydet'e basın.")
        except CatalogError as ex:
            self._warn(str(ex))
            e.setText(self.store.get_text(k))
        self._update_dirty_ui()

    def _list_add(self):
        k = self._cur_key()
        try:
            self.store.add_list_item(k, self.ls_edit.text())
        except CatalogError as e:
            return self._warn(str(e))
        self.refresh_list()
        self.ls_list.setCurrentRow(self.ls_list.count() - 1)
        self._set_status("Eklendi. Kalıcı olması için 💾 Kaydet'e basın.")

    def _list_update(self):
        k, r = self._cur_key(), self.ls_list.currentRow()
        if r < 0:
            return
        old = self.store.get_list(k)[r]
        new = self.ls_edit.text()
        n = self.store.list_item_usage(self.df(), k, old)
        if n and not _same(old, new) and not self._confirm(
                "Değer değişiyor",
                f"'{old}' değeri CSV'de {n} satırda kullanılıyor. Değişiklik CSV'deki eski değeri DEĞİŞTİRMEZ.\n\nDevam edilsin mi?"):
            return
        try:
            self.store.update_list_item(k, r, new)
        except CatalogError as e:
            return self._warn(str(e))
        self.refresh_list()
        self.ls_list.setCurrentRow(r)
        self._set_status("Güncellendi. Kalıcı olması için 💾 Kaydet'e basın.")

    def _list_delete(self):
        k, r = self._cur_key(), self.ls_list.currentRow()
        if r < 0:
            return
        val = self.store.get_list(k)[r]
        n = self.store.list_item_usage(self.df(), k, val)
        msg = f"'{val}' listeden silinsin mi?"
        if n:
            msg += f"\n\nBu değer CSV'de {n} satırda kullanılıyor; CSV değişmez, ama yeni etiketlemede listede çıkmaz."
        if not self._confirm("Değer sil", msg):
            return
        self.store.delete_list_item(k, r)
        self.refresh_list()
        self._set_status("Silindi. Kalıcı olması için 💾 Kaydet'e basın.")

    def _list_move(self, d):
        k, r = self._cur_key(), self.ls_list.currentRow()
        if r < 0:
            return
        j = self.store.move_list_item(k, r, d)
        self.refresh_list()
        self.ls_list.setCurrentRow(j)

    # ------------------------------------------------------------------ genel
    def refresh_all(self):
        self.refresh_species()
        self.refresh_habitats()
        self.refresh_list_combo()
        self.refresh_list()
        self._species_selected()
        self._habitat_selected()
        self._update_dirty_ui()

    def save(self):
        try:
            written = self.store.save()
        except Exception as e:  # disk/izin hatası
            QMessageBox.critical(self, "Kaydedilemedi", f"Dosya yazılamadı:\n{e}")
            return
        if written:
            self._set_status("Kaydedildi: " + ", ".join(written) + "  (yedekler: yedek_json/)")
            self.saved.emit()
        self._update_dirty_ui()

    def revert(self):
        if self.store.dirty and not self._confirm("Geri al", "Kaydedilmemiş tüm değişiklikler atılsın mı?"):
            return
        self.store.revert()
        self.refresh_all()
        self._set_status("Değişiklikler geri alındı.")

    def closeEvent(self, ev):
        if self.store.dirty:
            r = QMessageBox.question(self, "Kaydedilmemiş değişiklik",
                                     "Kaydedilmemiş değişiklikler var. Kaydedilsin mi?",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                     QMessageBox.Save)
            if r == QMessageBox.Cancel:
                return ev.ignore()
            if r == QMessageBox.Save:
                self.save()
                if self.store.dirty:
                    return ev.ignore()
        ev.accept()


def _same(a, b):
    return " ".join(str(a).split()) == " ".join(str(b).split())
