"""Ön Hazırlık sekmesinin alt bölümleri: varsayılan etiket bilgileri, veri klasörleri, model durumu; yedek penceresi."""
import os

from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QHeaderView,
                             QAbstractItemView, QWidget)

from ..data import on_hazirlik

FIELD_LABELS = {"Recordist": "Kaydeden", "Ornithologist": "Ornitolog",
                "Microphone": "Mikrofon", "Recorder": "Kayıt cihazı"}


def _open_folder(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    return QDesktopServices.openUrl(QUrl.fromLocalFile(path))


class PrepPanel(QWidget):
    """Üç bölüm: Varsayılan etiket bilgileri / Veri klasörleri / Modeller."""
    defaults_saved = pyqtSignal()

    def __init__(self, app, config_root, csv_path, segment_dirs, cards=(), parent=None):
        super().__init__(parent)
        self._app = app
        self._root = config_root
        self._csv_path = csv_path
        self._segment_dirs = segment_dirs
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        left = QVBoxLayout()
        card_row = QHBoxLayout()
        card_row.setSpacing(10)
        for c in cards:
            card_row.addWidget(c)
        left.addLayout(card_row)
        left.addWidget(self._build_defaults())
        left.addStretch()
        right = QVBoxLayout()
        right.addWidget(self._build_models())
        right.addWidget(self._build_data())
        right.addStretch()
        lay.addLayout(left, 3)
        lay.addLayout(right, 2)
        self.refresh_all()

    # ------------------------------------------------------------ varsayılan etiket bilgileri
    def _build_defaults(self) -> QGroupBox:
        box = QGroupBox("Varsayılan Etiket Bilgileri")
        grid = QGridLayout(box)
        self.default_combos = {}
        for i, (field, label) in enumerate(FIELD_LABELS.items()):
            grid.addWidget(QLabel(label + ":"), i // 2, (i % 2) * 2)
            cmb = QComboBox()
            cmb.setEditable(True)
            cmb.setMinimumWidth(170)
            self.default_combos[field] = cmb
            grid.addWidget(cmb, i // 2, (i % 2) * 2 + 1)
        row = QHBoxLayout()
        self.btn_save_defaults = QPushButton("Kaydet")
        self.btn_save_defaults.clicked.connect(self.save_defaults)
        self.lbl_defaults = QLabel("Etiketleme formu bu değerlerle açılır.")
        self.lbl_defaults.setStyleSheet("color:#555;")
        row.addWidget(self.btn_save_defaults)
        row.addWidget(self.lbl_defaults, 1)
        grid.addLayout(row, 2, 0, 1, 4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return box

    def refresh_lists(self):
        """Listeler (Tanım Verileri Yöneticisi'nde) değiştiyse seçenekleri yeniler; yazılı değerleri korur."""
        cfg = self._app.config_data
        saved = on_hazirlik.load_label_defaults(self._root)
        for field, key in on_hazirlik.DEFAULT_FIELDS.items():
            cmb = self.default_combos[field]
            cur = cmb.currentText() or saved.get(field, "")
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItem("")
            cmb.addItems(list(cfg.get(key, [])))
            cmb.blockSignals(False)
            idx = cmb.findText(cur)
            if idx >= 0:
                cmb.setCurrentIndex(idx)
            else:
                cmb.setEditText(cur)

    def save_defaults(self):
        values = {f: c.currentText().strip() for f, c in self.default_combos.items()}
        try:
            on_hazirlik.save_label_defaults(self._root, values)
        except OSError as exc:
            QMessageBox.warning(self, "Varsayılan Etiket Bilgileri", f"Kaydedilemedi:\n{exc}")
            return
        self._app.apply_label_defaults()
        self.lbl_defaults.setText("Kaydedildi. Etiketleme formuna uygulandı.")
        self.defaults_saved.emit()

    # ------------------------------------------------------------ veri klasörleri
    def _build_data(self) -> QGroupBox:
        box = QGroupBox("Veri Klasörleri")
        grid = QGridLayout(box)
        self.lbl_csv = QLabel()
        self.lbl_seg = QLabel()
        for lbl in (self.lbl_csv, self.lbl_seg):
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lbl.setWordWrap(True)
        self.btn_open_csv = QPushButton("Klasörü aç")
        self.btn_open_seg = QPushButton("Klasörü aç")
        self.btn_open_csv.clicked.connect(lambda: self._open(os.path.dirname(self._csv_path)))
        self.btn_open_seg.clicked.connect(lambda: self._open(self._overview.get("segments_dir", "")))
        grid.addWidget(QLabel("CSV:"), 0, 0)
        grid.addWidget(self.lbl_csv, 0, 1)
        grid.addWidget(self.btn_open_csv, 0, 2)
        grid.addWidget(QLabel("Segmentler:"), 1, 0)
        grid.addWidget(self.lbl_seg, 1, 1)
        grid.addWidget(self.btn_open_seg, 1, 2)
        grid.setColumnStretch(1, 1)
        self._overview = {}
        return box

    def _open(self, folder):
        if not _open_folder(folder):
            QMessageBox.information(self, "Veri Klasörleri", "Klasör bulunamadı.")

    def refresh_data(self):
        ov = on_hazirlik.data_overview(self._csv_path, self._segment_dirs)
        self._overview = ov
        self.lbl_csv.setText(f"{ov['csv_path']}  —  {ov['csv_rows']} kayıt" if ov["csv_exists"]
                             else f"{ov['csv_path']}  —  dosya henüz yok")
        self.lbl_seg.setText(f"{ov['segments_dir']}  —  {ov['segment_files']} ses dosyası" if ov["segments_exists"]
                             else f"{ov['segments_dir']}  —  klasör henüz yok")
        self.btn_open_csv.setEnabled(os.path.isdir(os.path.dirname(self._csv_path)))
        self.btn_open_seg.setEnabled(ov["segments_exists"])

    # ------------------------------------------------------------ modeller
    def _build_models(self) -> QGroupBox:
        box = QGroupBox("Modeller")
        grid = QGridLayout(box)
        self._model_labels = []
        for i in range(3):
            name = QLabel()
            name.setStyleSheet("font-weight:bold;")
            msg = QLabel()
            msg.setWordWrap(True)
            msg.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(name, i, 0)
            grid.addWidget(msg, i, 1)
            self._model_labels.append((name, msg))
        self.btn_refresh_models = QPushButton("Yenile")
        self.btn_refresh_models.clicked.connect(self.refresh_models)
        self.btn_refresh_models.setToolTip("Model dosyalarına yeniden bakar.")
        grid.addWidget(self.btn_refresh_models, 2, 2)
        grid.setColumnStretch(1, 1)
        return box

    def refresh_models(self):
        for (name, msg), (title, ok, detail) in zip(self._model_labels, on_hazirlik.model_status()):
            name.setText(("✔ " if ok else "✘ ") + title)
            name.setStyleSheet("font-weight:bold; color:%s;" % ("#0F8B6D" if ok else "#B23B3B"))
            msg.setText(detail)

    def refresh_all(self):
        self.refresh_lists()
        self.refresh_data()
        self.refresh_models()


class BackupRestoreDialog(QDialog):
    """Tanım Verileri Yöneticisi yedeklerini listeler; seçileni geri yükler."""
    restored = pyqtSignal(str)

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self._root = root
        self.setWindowTitle("Yedekten Geri Yükle")
        self.setMinimumSize(560, 320)
        self.resize(720, 420)
        self.setSizeGripEnabled(True)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        lay = QVBoxLayout(self)
        note = QLabel("Tanım Verileri Yöneticisi her kayıtta eski dosyanın yedeğini alır. "
                      "Geri yüklemeden önce geçerli dosyanın da yedeği alınır.")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Dosya", "Yedek tarihi", "Boyut"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setStretchLastSection(False)
        self.table.itemSelectionChanged.connect(self._sync)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addStretch()
        self.btn_restore = QPushButton("Geri Yükle")
        self.btn_restore.clicked.connect(self.restore_selected)
        self.btn_close = QPushButton("Kapat")
        self.btn_close.clicked.connect(self.accept)
        row.addWidget(self.btn_restore)
        row.addWidget(self.btn_close)
        lay.addLayout(row)
        self._backups = []
        self.reload()

    def reload(self):
        self._backups = on_hazirlik.list_backups(self._root)
        self.table.setRowCount(len(self._backups))
        for r, b in enumerate(self._backups):
            self.table.setItem(r, 0, QTableWidgetItem(b["target"]))
            self.table.setItem(r, 1, QTableWidgetItem(b["when"].strftime("%d.%m.%Y  %H:%M:%S")))
            self.table.setItem(r, 2, QTableWidgetItem(f"{b['size'] / 1024:.1f} KB"))
        if self._backups:
            self.table.selectRow(0)
        self._sync()

    def _sync(self):
        self.btn_restore.setEnabled(bool(self._backups) and self.table.currentRow() >= 0)

    def restore_selected(self, confirm=True):
        r = self.table.currentRow()
        if not (0 <= r < len(self._backups)):
            return None
        b = self._backups[r]
        if confirm:
            ans = QMessageBox.question(
                self, "Yedekten Geri Yükle",
                f"{b['target']} dosyası {b['when']:%d.%m.%Y %H:%M:%S} tarihli yedekle değiştirilecek.\n"
                "Geçerli dosyanın yedeği de alınır. Devam edilsin mi?")
            if ans != QMessageBox.Yes:
                return None
        try:
            name = on_hazirlik.restore_backup(self._root, b["path"])
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Yedekten Geri Yükle", f"Geri yüklenemedi:\n{exc}")
            return None
        self.reload()
        self.restored.emit(name)
        return name
