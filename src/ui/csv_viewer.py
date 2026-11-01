"""Uygulama içi CSV görüntüleyici: veri setine kaydedilenleri salt-okunur tabloda gösterir."""
import os
from datetime import datetime

import pandas as pd
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QPushButton,
    QTableView, QHeaderView, QAbstractItemView, QShortcut,
)

from . import theme

# Varsayılan (kompakt) görünümde gösterilen sütunlar; dosyada olmayanlar atlanır.
COMPACT_COLUMNS = [
    "Original_Recording_Name", "Filename", "Start_Time", "End_Time", "Scientific_Name",
    "Turkish_Name", "Call_Type", "Location", "Habitat_ID", "Habitat_Type",
    "Verification_Status", "Confidence_Level", "Recordist", "Ornithologist",
]


def read_csv_text(path):
    """CSV'yi tamamı metin olarak okur (`;` ayraç, utf-8-sig). Dosya yok/boşsa None."""
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return None
    return df if len(df.columns) else None


def _as_float(s):
    try:
        return float(str(s).replace(",", "."))
    except ValueError:
        return None


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self._df = df
        self._numeric = []
        for c in df.columns:
            vals = [v for v in df[c] if str(v).strip() != ""]
            self._numeric.append(bool(vals) and all(_as_float(v) is not None for v in vals))

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        v = self._df.iat[index.row(), index.column()]
        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            return str(v)
        if role == Qt.UserRole:  # sıralama anahtarı
            if self._numeric[index.column()]:
                f = _as_float(v)
                return float("-inf") if f is None else f
            return str(v).casefold()
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return str(self._df.columns[section]) if orientation == Qt.Horizontal else str(section + 1)

    def column_names(self):
        return list(self._df.columns)


class _Proxy(QSortFilterProxyModel):
    """Tüm sütunlarda büyük/küçük harf duyarsız arama; sayısal sütunlarda sayısal sıralama."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSortRole(Qt.UserRole)
        self._needle = ""

    def set_needle(self, text):
        self._needle = text.strip().casefold()
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        if not self._needle:
            return True
        m = self.sourceModel()
        return any(self._needle in str(m.data(m.index(row, c))).casefold()
                   for c in range(m.columnCount()))


class CsvViewerDialog(QDialog):
    def __init__(self, csv_path, open_external=None, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self._open_external = open_external
        self.setWindowTitle("Veri Seti (CSV) Görüntüleyici")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        self.setSizeGripEnabled(True)
        self.resize(1250, 720)
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)

        lay = QVBoxLayout(self)
        theme.add_hero(lay, self.windowTitle(), "Veri setini tablo olarak inceleyin, arayın ve sıralayın")
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        lay.addWidget(self.info)

        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Ara (tür, kayıt, konum, habitat… tüm sütunlarda)")
        self.search.setClearButtonEnabled(True)
        self.chk_all = QCheckBox("Tüm sütunlar")
        self.count_lbl = QLabel()
        bar.addWidget(self.search, 1)
        bar.addWidget(self.chk_all)
        bar.addWidget(self.count_lbl)
        lay.addLayout(bar)

        self.view = QTableView()
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setAlternatingRowColors(True)
        self.view.setSortingEnabled(True)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.view, 1)

        btns = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 Yenile")
        self.btn_external = QPushButton("📊 Harici Programda Aç")
        self.btn_close = QPushButton("Kapat")
        btns.addWidget(self.btn_refresh)
        btns.addWidget(self.btn_external)
        btns.addStretch()
        btns.addWidget(self.btn_close)
        lay.addLayout(btns)

        self.proxy = _Proxy(self)
        self.view.setModel(self.proxy)
        self.search.textChanged.connect(self._on_search)
        self.chk_all.toggled.connect(self._apply_columns)
        self.btn_refresh.clicked.connect(self.reload)
        self.btn_external.clicked.connect(self._on_external)
        self.btn_close.clicked.connect(self.close)
        self.btn_external.setEnabled(open_external is not None)
        self.reload()

    # ---- veri ----
    def reload(self):
        df = read_csv_text(self.csv_path)
        self.df = df
        if df is None:
            self.proxy.setSourceModel(None)
            self._model = None
            self.info.setText(f"CSV dosyası bulunamadı veya boş:<br><code>{self.csv_path}</code>")
            self.count_lbl.setText("")
            return
        self._model = DataFrameModel(df, self)
        self.proxy.setSourceModel(self._model)
        self._apply_columns()
        self.view.sortByColumn(-1, Qt.AscendingOrder)
        self._update_info()
        self._update_count()

    def _update_info(self):
        df = self.df
        parts = [f"<b>{len(df)}</b> segment"]
        if "Scientific_Name" in df:
            parts.append(f"<b>{df['Scientific_Name'].replace('', pd.NA).dropna().nunique()}</b> tür")
        if "Original_Recording_Name" in df:
            parts.append(f"<b>{df['Original_Recording_Name'].replace('', pd.NA).dropna().nunique()}</b> kayıt dosyası")
        if "Habitat_ID" in df:
            parts.append(f"<b>{df['Habitat_ID'].replace('', pd.NA).dropna().nunique()}</b> habitat noktası")
        mtime = datetime.fromtimestamp(os.path.getmtime(self.csv_path)).strftime("%d.%m.%Y %H:%M")
        self.info.setText(" · ".join(parts)
                          + f"<br><span style='color:#666'>{self.csv_path} — son değişiklik: {mtime}</span>")

    def _update_count(self):
        if self._model is None:
            return
        shown, total = self.proxy.rowCount(), self._model.rowCount()
        self.count_lbl.setText(f"{shown} / {total} satır")

    def _on_search(self, text):
        self.proxy.set_needle(text)
        self._update_count()

    def _apply_columns(self):
        if self._model is None:
            return
        names = self._model.column_names()
        compact = [c for c in COMPACT_COLUMNS if c in names]
        show_all = self.chk_all.isChecked() or not compact
        for i, n in enumerate(names):
            self.view.setColumnHidden(i, not (show_all or n in compact))
        self.view.resizeColumnsToContents()
        for i in range(len(names)):  # çok uzun sütunları sınırla
            if self.view.columnWidth(i) > 280:
                self.view.setColumnWidth(i, 280)

    def _toggle_fullscreen(self):
        """F11: tam ekran / normal."""
        if self.isMaximized() or self.isFullScreen():
            self.showNormal()
        else:
            self.showMaximized()

    def _on_external(self):
        if self._open_external:
            self._open_external()
