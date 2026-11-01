import os
import csv
import sys
import numpy as np
import tempfile
import warnings
import json
import re
import wave
from datetime import datetime
from scipy.io.wavfile import read as wavread, write as wavwrite, WavFileWarning
import scipy.signal as sps
from scipy import signal
import traceback
from typing import Optional, List
import pandas as pd


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QFileDialog, QComboBox, QTextEdit, QGroupBox,
    QProgressDialog, QMessageBox, QTabWidget, QSplitter, QSizePolicy, QFormLayout, QCheckBox, QMenu,
    QLineEdit, QSpinBox, QDoubleSpinBox, QAction, QTableWidget, QTableWidgetItem, QInputDialog, QTableView,
    QAbstractItemView, QDialog, QHeaderView, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QUrl, QAbstractTableModel, QModelIndex, QThread
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QGuiApplication, QCursor, QDesktopServices, QColor, QFont, QFontMetrics, QKeySequence
from PyQt5.QtWidgets import QTabBar, QTextBrowser

# Matplotlib
import matplotlib.pyplot as plt
try:
    plt.switch_backend('Qt5Agg')
except ImportError:
    pass
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.patches import Rectangle


from ..audio.constants import (
    TARGET_SAMPLE_RATE, MAX_FILE_SIZE_MB, TARGET_LOUDNESS_LUFS,
    NOISE_REDUCTION_N_FFT, NOISE_REDUCTION_WIN_LENGTH,
    DEFAULT_SENSITIVITY, DEFAULT_MIN_DURATION, DEFAULT_MAX_DURATION,
    DEFAULT_MIN_SILENCE, DEFAULT_FRAME_LEN, DEFAULT_HOP_LEN,
    DEFAULT_N_FFT, DEFAULT_HOP, SPEC_MAX_COLS
)
from ..audio.utils import get_audio_file_info, write_wav_safe
from ..audio.loader import FileLoaderThread
from ..audio.processor import AudioProcessThread
from ..detection.detector import SpeciesDetectionThread
from ..detection import turkish_lookup as species_lookup
from ..visualization.spectrogram import SpectrogramComputeThread
from ..visualization import richness_plots
from ..visualization.plot_style import RENK
from ..data.config import HABITAT_DATA, BIRD_SPECIES_DATA, load_config, save_config as config_save_config
from ..data import dataset_summary
from ..data import coverage_report
from ..data import richness_insights
from ..data import location_map
from ..data import habitat_info
from ..data import richness_core
from ..visualization import habitat_plots
from ..data import classification_folders
from ..ml import pipeline as ml_pipeline
from ..ml import classification_diagnosis
from ..ml import last_result_store
from ..ml import model_registry
from ..ml import model_significance
from ..visualization import classification_plots
from ..data.csv_handler import get_project_root as csv_project_root
from ..data.config import get_project_root as src_root_dir
from . import theme


try:
    import librosa
except ImportError:
    librosa = None
try:
    import noisereduce as nr
except ImportError:
    nr = None
try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None

class _BaseSummaryDialog(QDialog):
    """
    Veri Seti Özeti ve Tür Zenginliği pencereleri için ortak altyapı (tam ekran
    geçişi, grafik çizimi, düğme satırı). Bu iki pencere bilerek AYRI tutulmuştur:
    biri genel veri seti istatistiklerini (kayıt sayıları, tür başına sayılar),
    diğeri habitat/bölge bazında tür zenginliğini gösterir. Böylece birine
    yapılacak eklemeler diğerini etkilemez.
    """

    def __init__(self, window_title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(window_title)
        self.resize(900, 750)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self._is_fullscreen = False
        self._window_title_plain = window_title

    HERO_SUBTITLE = ""

    def _add_hero(self, layout):
        theme.add_hero(layout, self._window_title_plain, self.HERO_SUBTITLE, compact=True)

    def _build_shell(self, text_summary: str) -> QVBoxLayout:
        """
        Ortak iskeleti kurar (metin özeti + kaydırılabilir grafik alanı +
        alt düğme satırı). Alt sınıflar kendi grafiklerini döndürülen
        scroll_layout'a ekler.
        """
        outer_layout = QVBoxLayout(self)
        self._add_hero(outer_layout)

        # Metin özeti ve grafikler TEK kaydırma alanında durur: dikey kaydırma çubuğu
        # pencerenin sağ kenarında, metni de grafikleri de birlikte kaydırır.
        text_label = QLabel(text_summary)
        text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        text_label.setStyleSheet(
            "QLabel { background:#eef6ee; padding:10px 14px; border-radius:6px; "
            "font-size: 15px; }"
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_content = QWidget()
        # Grafikler tek tek sola hizalanır (pencere genişleyince yapay biçimde
        # "ortalanmış" görünmesin); metin kutusu tam genişliği kullanır.
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setAlignment(Qt.AlignTop)
        scroll_layout.addWidget(text_label)
        self._scroll_layout = scroll_layout
        scroll.setWidget(scroll_content)
        outer_layout.addWidget(scroll)

        outer_layout.addLayout(self._make_button_row())

        return scroll_layout

    def _make_button_row(self) -> QHBoxLayout:
        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("🔄 Yenile")
        btn_refresh.clicked.connect(self._on_refresh)
        self.btn_fullscreen = QPushButton("🔲 Tam Ekran")
        self.btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        btn_close = QPushButton("Kapat")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(btn_refresh)
        btn_row.addWidget(self.btn_fullscreen)
        btn_row.addWidget(self._make_save_charts_button())
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        return btn_row

    def _add_chart(self, counts, title: str, xlabel: str, highlight_thresholds: bool,
                   bar_color: str = "#4a90d9", max_bars=richness_plots.MAX_BARS):
        canvas = self._make_bar_chart_canvas(counts, title, xlabel, highlight_thresholds, bar_color,
                                             max_bars=max_bars)
        # Sola hizalama, QVBoxLayout'un tuvali tam genişliğe zorla germesini
        # (matplotlib'in yeniden çizip ortalanmış gibi göstermesini) önler;
        # tuval kendi sabit boyutunu korur.
        self._scroll_layout.addWidget(canvas, alignment=Qt.AlignLeft)

    def _add_custom_figure(self, fig):
        """Hazır bir matplotlib Figure'ünü (çubuk grafik olmayan, örn. harita)
        aynı sabit boyut/sola hizalama kurallarıyla ekler."""
        canvas = self._wrap_figure_as_canvas(fig)
        self._scroll_layout.addWidget(canvas, alignment=Qt.AlignLeft)

    # ------------------------------------------------------------------
    # Grafikleri yüksek çözünürlüklü dosya olarak kaydetme
    # ------------------------------------------------------------------
    @staticmethod
    def _slugify_for_filename(text: str) -> str:
        """Başlığı dosya adına uygun, ASCII ve güvenli bir metne çevirir."""
        tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
        text = text.translate(tr_map)
        text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
        return text[:60] or "grafik"

    def _collect_figures(self):
        """Penceredeki (kaydırılabilir alandaki) güncel matplotlib figürlerini toplar."""
        # İç içe yerleşimlerdeki (iki sütunlu satırlar) tuvaller de dahil
        return [c.figure for c in self.findChildren(FigureCanvas)]

    def _make_save_charts_button(self) -> QPushButton:
        btn = QPushButton("💾 Grafikleri Kaydet")
        menu = QMenu(btn)
        menu.addAction("PNG — 300 dpi (tez/yayın için önerilir)",
                       lambda checked=False: self._save_all_charts("png", 300))
        menu.addAction("PNG — 600 dpi (çok yüksek çözünürlük)",
                       lambda checked=False: self._save_all_charts("png", 600))
        menu.addAction("PDF — vektörel (büyütünce bozulmaz)",
                       lambda checked=False: self._save_all_charts("pdf", None))
        btn.setMenu(menu)
        return btn

    def _save_all_charts(self, fmt: str, dpi):
        """Penceredeki tüm grafikleri seçilen klasöre, her biri ayrı dosya olarak kaydeder."""
        figures = self._collect_figures()
        if not figures:
            QMessageBox.information(self, "Grafikleri Kaydet", "Kaydedilecek grafik yok.")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Grafiklerin kaydedileceği klasörü seçin", csv_project_root()
        )
        if not folder:
            return

        prefix = self._slugify_for_filename(re.sub(r"[^\w\s]", "", self.windowTitle()))
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        saved, errors = [], []
        for idx, fig in enumerate(figures, start=1):
            title = fig.axes[0].get_title() if fig.axes else ""
            filename = f"{prefix}_{stamp}_{idx:02d}_{self._slugify_for_filename(title)}.{fmt}"
            path = os.path.join(folder, filename)
            try:
                kwargs = {"bbox_inches": "tight", "facecolor": "white"}
                if dpi:
                    kwargs["dpi"] = dpi
                fig.savefig(path, **kwargs)
                saved.append(path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{filename}: {exc}")

        self._report_saved_files(folder, saved, errors)

    def _report_saved_files(self, folder: str, saved: list, errors: list):
        """Kaydetme sonucunu bildirir ve klasörü açmayı önerir."""
        if errors:
            QMessageBox.warning(
                self, "Grafikleri Kaydet",
                f"{len(saved)} grafik kaydedildi, {len(errors)} grafik kaydedilemedi:\n\n" + "\n".join(errors),
            )
            return
        answer = QMessageBox.question(
            self, "Grafikleri Kaydet",
            f"{len(saved)} grafik kaydedildi:\n{folder}\n\nKlasörü açmak ister misiniz?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _toggle_fullscreen(self):
        if self._is_fullscreen:
            self.showNormal()
            self.btn_fullscreen.setText("🔲 Tam Ekran")
        else:
            self.showFullScreen()
            self.btn_fullscreen.setText("🔳 Normal Görünüm")
        self._is_fullscreen = not self._is_fullscreen

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._is_fullscreen:
            self._toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    def _wrap_figure_as_canvas(self, fig):
        """
        matplotlib Figure'ünü kendi piksel boyutuna sabitlenmiş bir FigureCanvas
        içine alır. Aksi halde QVBoxLayout tuvali kap genişliğine (tam ekranda
        çok daha geniş) uzatır ve grafik yeniden çizilerek yanıltıcı biçimde
        "ortalanmış/gerilmiş" görünür.
        """
        w_inches, h_inches = fig.get_size_inches()
        canvas = FigureCanvas(fig)
        canvas.setFixedSize(int(w_inches * fig.dpi), int(h_inches * fig.dpi))
        # FigureCanvas kendi referansını tuttuğundan figürü pyplot'un genel
        # figür takibinden çıkarırız; aksi halde uzun oturumlarda (çok sayıda
        # pencere açılıp kapandığında) figür listesi büyür, bellek kullanımı
        # artar ve kapanışta/çöp toplamada nadiren hata çıkabilir.
        plt.close(fig)
        return canvas

    def _make_bar_chart_canvas(self, counts, title: str, xlabel: str, highlight_thresholds: bool,
                                bar_color: str = None, max_bars=richness_plots.MAX_BARS):
        colors = None
        if highlight_thresholds:
            def colors(values):
                return [
                    RENK["kirmizi"] if v < dataset_summary.LOW_SAMPLE_THRESHOLD
                    else RENK["yesil"] if v >= dataset_summary.GOOD_SAMPLE_THRESHOLD
                    else RENK["turuncu"]
                    for v in values
                ]
        fig = richness_plots.bar_chart(counts, title, xlabel, colors=colors, bar_color=bar_color,
                                       max_bars=max_bars)
        return self._wrap_figure_as_canvas(fig)


class DatasetSummaryDialog(_BaseSummaryDialog):
    """
    Genel veri seti istatistiklerini (toplam kayıt, tür/bölge/habitat/doğrulama
    durumu dağılımları) gösteren pencere. Tür zenginliği burada DEĞİL,
    ayrı SpeciesRichnessDialog penceresindedir.
    """

    HERO_SUBTITLE = "Tür, bölge, habitat ve doğrulama durumuna göre kayıt sayıları"

    def __init__(self, summary: "dataset_summary.DatasetSummary", parent=None):
        super().__init__("📈 Veri Seti Özeti", parent)
        self.summary = summary
        self._build_ui()

    def _build_ui(self):
        self._build_shell(self.summary.format_text_summary())

        self._add_chart(
            counts=self.summary.species_counts,
            title="Tür Başına Segment Sayısı",
            xlabel="Segment sayısı",
            highlight_thresholds=True,
            max_bars=None,   # tüm türler (1 segmentlikler dahil) gösterilir
        )
        self._add_chart(
            counts=self.summary.location_counts,
            title="Bölge Başına Segment Sayısı",
            xlabel="Segment sayısı",
            highlight_thresholds=False,
        )
        self._add_chart(
            counts=self.summary.habitat_counts,
            title="Habitat Tipi Başına Segment Sayısı",
            xlabel="Segment sayısı",
            highlight_thresholds=False,
        )
        self._add_chart(
            counts=self.summary.verification_counts,
            title="Doğrulama Durumu Başına Segment Sayısı",
            xlabel="Segment sayısı",
            highlight_thresholds=False,
            bar_color=RENK["gri"],
        )

    def _on_refresh(self):
        parent = self.parent()
        if parent is None:
            return
        csv_full_path = os.path.join(csv_project_root(), parent.csv_file) \
            if not os.path.isabs(parent.csv_file) else parent.csv_file
        df = dataset_summary.load_dataset(csv_full_path)
        if df is None:
            QMessageBox.information(self, "Veri Seti Özeti", "Henüz veri seti (CSV) bulunamadı ya da dosya boş.")
            return
        self.summary = dataset_summary.compute_summary(df, config=coverage_report.load_project_config())
        self.close()
        new_dialog = DatasetSummaryDialog(self.summary, parent=parent)
        parent._dataset_summary_dialog = new_dialog
        new_dialog.show()


class _FitTabBar(QTabBar):
    """Sekme genişliğini kendi yazı ölçüsünden hesaplar. Qt, stil sayfasındaki kalın/büyük yazıyı (özellikle ana
    pencerenin genel stili devralındığında) boyut hesabına katmayıp sekmeleri yazıdan dar bırakabiliyor; bu
    sınıf yazıyı her zaman KALIN ve iki punto büyük varsayarak ölçer, bu yüzden hiçbir koşulda kırpılmaz."""

    PAD = 14

    def tabSizeHint(self, index):
        base = super().tabSizeHint(index)
        f = QFont(self.font())
        f.setBold(True)
        if f.pointSizeF() > 0:
            f.setPointSizeF(f.pointSizeF() + 2)
        elif f.pixelSize() > 0:
            f.setPixelSize(f.pixelSize() + 3)
        w = QFontMetrics(f).horizontalAdvance(self.tabText(index)) + 2 * self.PAD
        base.setWidth(max(base.width(), w))
        return base


class _FitTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabBar(_FitTabBar(self))


class _FitWidthTable(QTableWidget):
    """İçeriğe göre boyutlanan sütunlar; tablo içerikten genişse artan genişlik sütunlara eşit paylaştırılır, böylece
    başlık şeridinin sağında boş alan kalmaz. İçerik tablodan genişse sütunlar içerik genişliğinde kalır (yatay kaydırma)."""

    def refit(self):
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.resizeColumnsToContents()
        base = [self.columnWidth(c) for c in range(self.columnCount())]
        extra = self.viewport().width() - sum(base)
        if base and extra > 0:
            hdr.setSectionResizeMode(QHeaderView.Interactive)
            share = extra // len(base)
            for c, w in enumerate(base):
                self.setColumnWidth(c, w + share)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.refit()

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, self.refit)


class SpeciesRichnessDialog(_BaseSummaryDialog):
    """
    Tür Zenginliği Belirleme panosu (DatasetSummaryDialog'dan bilerek ayrı):
    başlık bandı, özet kartları, otomatik bulgular ve sekmelere ayrılmış, her biri
    "ne gösterir / nasıl okunur / dikkat" açıklaması taşıyan grafik kartları.
    Yalnızca belirlenen türler ve seyreltilmiş karşılaştırma değerleri gösterilir.
    """

    _CSS = """
    QDialog { background:#f3f5f9; }
    QLabel { background:transparent; }
    QFrame#card { background:#ffffff; border:1px solid #e1e6ee; border-radius:12px; }
    QLabel#sec_title { font-size:15px; font-weight:700; color:#1f2a37; }
    QLabel#cap { font-size:12px; color:#4b5563; }
    QTabWidget::pane { border:none; }
    QTabBar::tab { padding:9px 12px; min-width:0px; min-height:20px; margin-right:3px; background:#e3e9f2;
                   color:#374151; border-top-left-radius:8px; border-top-right-radius:8px; font-size:12px;
                   font-weight:600; border:none; }
    QTabBar::tab:selected { background:#ffffff; color:#0b5394; }
    QScrollArea { background:transparent; border:none; }
    QScrollArea > QWidget > QWidget { background:transparent; }
    """

    def __init__(self, summary: "dataset_summary.DatasetSummary", parent=None):
        super().__init__("🧮 Tür Zenginliği Belirleme", parent)
        self.summary = summary
        self.resize(1560, 920)
        self.setMinimumWidth(900)
        self.setStyleSheet(self._CSS)
        self._build_ui()

    def showEvent(self, e):
        super().showEvent(e)
        tabs = getattr(self, "_tabs", None)
        if tabs is not None:       # stil sayfası uygulandıktan sonra sekme boyutlarını yeniden hesapla
            QTimer.singleShot(0, lambda: (tabs.tabBar().updateGeometry(), tabs.tabBar().update()))

    # ------------------------------------------------------------------ yapı taşları
    def _hero(self) -> QFrame:
        return theme.make_hero("Tür Zenginliği Belirleme",
                               "Etiketli noktalarda belirlenen türlerin nokta, habitat ve bölge bazında özeti")

    @staticmethod
    def _kpi(value: str, title: str, sub: str, accent: str) -> QFrame:
        f = QFrame()
        f.setObjectName("card")
        f.setStyleSheet(f"QFrame#card {{ background:#ffffff; border:1px solid #e1e6ee; "
                        f"border-top:4px solid {accent}; border-radius:12px; }}")
        v = QVBoxLayout(f)
        v.setContentsMargins(16, 10, 16, 10)
        v.setSpacing(0)
        a = QLabel(value)
        a.setStyleSheet(f"font-size:32px; font-weight:800; color:{accent};")
        b = QLabel(title)
        b.setStyleSheet("font-size:13px; font-weight:700; color:#1f2a37;")
        c = QLabel(sub)
        c.setWordWrap(True)
        c.setStyleSheet("font-size:11px; color:#6b7280;")
        v.addWidget(a)
        v.addWidget(b)
        v.addWidget(c)
        return f

    def _kpi_row(self) -> QWidget:
        s = self.summary
        eff = s.point_effort
        has_eff = eff is not None and len(eff) > 0
        n_seg = int(eff["segment"].sum()) if has_eff else s.total_records
        med = f"nokta başına ortanca {eff['segment'].median():.0f}" if has_eff else "etiketli kayıt"
        inc = s.point_incidence
        tek_txt = "—"
        if inc is not None and getattr(inc, "size", 0) > 0:
            freq = np.asarray(inc).astype(bool).sum(axis=0)
            freq = freq[freq > 0]
            if len(freq):
                tek = int((freq == 1).sum())
                tek_txt = f"{tek} tanesi (%{100 * tek / len(freq):.0f}) yalnızca 1 noktada"
        loc = s.location_rarefaction
        n_loc = len(loc) if loc is not None else s.unique_locations
        cards = [
            (str(s.n_species_observed), "Belirlenen tür", tek_txt, RENK["mavi"]),
            (str(s.n_points), "Etiketli nokta", "Habitat_ID birimi", RENK["yesil"]),
            (f"{n_seg:,}".replace(",", "."), "Etiketli segment", med, RENK["mor"]),
            (str(s.n_recordings) if s.n_recordings else "—", "Kayıt", "farklı uzun kayıt dosyası", RENK["turuncu"]),
            (str(len(s.type_richness)), "Habitat tipi", "ekolojik grup", RENK["acik_mavi"]),
            (str(n_loc), "Bölge", "etiketli noktaların bulunduğu bölge", RENK["kirmizi"]),
        ]
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)
        for c in cards:
            h.addWidget(self._kpi(*c), 1)
        return row

    @staticmethod
    def _card() -> tuple:
        card = QFrame()
        card.setObjectName("card")
        v = QVBoxLayout(card)
        v.setContentsMargins(18, 14, 18, 16)
        v.setSpacing(6)
        return card, v

    @staticmethod
    def _caption(what=None, how=None, caution=None, max_width=None) -> list:
        labels = []
        for head, text, color in (("Ne gösterir: ", what, "#1f2a37"), ("Nasıl okunur: ", how, "#1f2a37"),
                                  ("Dikkat: ", caution, "#b45309")):
            if not text:
                continue
            lab = QLabel(f"<span style='color:{color}'><b>{head}</b></span>{text}")
            lab.setObjectName("cap")
            lab.setTextFormat(Qt.RichText)
            lab.setWordWrap(True)
            if max_width:
                lab.setMaximumWidth(max_width)
            labels.append(lab)
        return labels

    def _chart_card(self, fig, title, what, how, caution=None, max_px=760) -> QFrame:
        card, v = self._card()
        w_in, h_in = fig.get_size_inches()
        if w_in * fig.dpi > max_px:   # iki grafik yan yana sığsın: oranı koruyarak küçült, yerleşimi yenile
            k = max_px / (w_in * fig.dpi)
            fig.set_size_inches(w_in * k, h_in * k)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)    # gridspec'li şekiller tight_layout uyarısı verir
                    fig.tight_layout()
            except Exception:  # noqa: BLE001
                pass
        canvas = self._wrap_figure_as_canvas(fig)
        t = QLabel(title)
        t.setObjectName("sec_title")
        t.setWordWrap(True)
        t.setMaximumWidth(canvas.width())
        v.addWidget(t)
        for lab in self._caption(what, how, caution, canvas.width()):
            v.addWidget(lab)
        v.addSpacing(4)
        v.addWidget(canvas, alignment=Qt.AlignLeft)
        v.addStretch()
        return card

    def _note_card(self, title, text) -> QFrame:
        card, v = self._card()
        t = QLabel(title)
        t.setObjectName("sec_title")
        v.addWidget(t)
        for lab in self._caption(text):
            lab.setText(text)
            v.addWidget(lab)
        return card

    @staticmethod
    def _row(widgets) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(14)
        for x in widgets:
            if x is not None:
                h.addWidget(x, alignment=Qt.AlignTop)
        h.addStretch()
        return w

    @staticmethod
    def _page(widgets) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 12, 0, 8)
        v.setSpacing(14)
        v.setAlignment(Qt.AlignTop)
        for w in widgets:
            if w is not None:
                v.addWidget(w)
        sa.setWidget(inner)
        return sa

    @staticmethod
    def _on_accumulation_click(canvas, event):
        """Birikim eğrisinde çizgiye (özellikle uçtaki yuvarlağa) tıklayınca 'N nokta → T tür' yazar."""
        if event.x is None or event.y is None:
            return
        richness_plots.accumulation_label_at(canvas.figure, event.x, event.y)
        canvas.draw_idle()

    def _on_incidence_click(self, event):
        """Tür-sıklığı grafiğinde tıklanan değer için (x = nokta sayısı) türleri ve noktalarını gösterir."""
        if event.inaxes is None or event.xdata is None:
            return
        self._show_species_at_frequency(int(round(event.xdata)))

    def _show_species_at_frequency(self, k: int):
        inc = getattr(self.summary, "point_incidence", None)
        if k < 1 or inc is None or getattr(inc, "empty", True):
            return
        rows = richness_core.species_at_frequency(inc, k)
        if not rows:
            QMessageBox.information(
                self, "Bu değerde tür yok",
                f"Tam {k} noktada belirlenen hiçbir tür yok.\n\n"
                "Bu bir hata değildir: grafikteki boşluk, o nokta sayısına sahip türün bulunmadığını gösterir.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Tam {k} noktada belirlenen türler")
        dlg.resize(760, 420)
        v = QVBoxLayout(dlg)
        head = QLabel(f"<b>Tam {k} noktada belirlenen {len(rows)} tür</b> (nokta = Habitat_ID)")
        head.setTextFormat(Qt.RichText)
        v.addWidget(head)
        tbl = QTableWidget(len(rows), 3)
        tbl.setHorizontalHeaderLabels(["Tür", "Nokta sayısı", "Noktalar (Habitat_ID)"])
        for i, (sp, pts) in enumerate(rows):
            tbl.setItem(i, 0, QTableWidgetItem(sp))
            it = QTableWidgetItem()
            it.setData(Qt.DisplayRole, len(pts))
            it.setTextAlignment(Qt.AlignCenter)
            tbl.setItem(i, 1, it)
            tbl.setItem(i, 2, QTableWidgetItem(", ".join(pts)))
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.verticalHeader().setVisible(False)
        tbl.setTextElideMode(Qt.ElideNone)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.resizeColumnsToContents()
        tbl.setAlternatingRowColors(True)
        v.addWidget(tbl, 1)
        btn = QPushButton("Kapat")
        btn.clicked.connect(dlg.close)
        v.addWidget(btn, alignment=Qt.AlignRight)
        self._freq_dialog = dlg
        dlg.show()

    def _habitat_limits_card(self) -> QFrame:
        """Habitat/bölge karşılaştırmasının dayanağı: tip ve bölge başına nokta sayısı ve yorum sınırı."""
        s = self.summary
        pt = getattr(s, "point_types", None)
        parts, weak = [], 0
        if pt is not None and len(pt):
            vc = pd.Series(pt).value_counts()
            weak = int((vc < 3).sum())
            parts.append("Habitat tipi başına nokta: " + ", ".join(f"{k} ({v})" for k, v in vc.items()) + ".")
        loc = s.location_rarefaction
        if loc is not None and len(loc):
            parts.append("Bölge başına nokta: " + ", ".join(f"{r.grup} ({int(r.m)})" for r in loc.itertuples()) + ".")
        verdict = ("Tip başına 3'ten az nokta olan grup(lar) var; bu sekmedeki habitat ve bölge karşılaştırmaları "
                   "<b>betimleyicidir, tezde bir fark iddiasının dayanağı olarak kullanılmamalıdır</b>."
                   if weak else
                   "Karşılaştırmalar yine betimleyicidir; fark iddiası için nokta sayısının yeterliliği ayrıca gösterilmelidir.")
        return self._note_card("Bu sekmenin yorum sınırı", " ".join(parts + [verdict]))

    def _ai_page(self) -> QScrollArea:
        """Yapay zekâ (BirdNET / Perch / birleşik) ile belirlenen tür zenginliğinin uzman etiketleriyle uyumu."""
        from ..ml import ai_richness
        from ..ml import richness_validation as _rv
        try:
            stored = model_registry.stored_results(csv_project_root())
        except Exception:  # noqa: BLE001
            stored = []
        res = ai_richness.build_rows(stored)
        widgets = []
        note, nv = self._card()
        title = QLabel("Yapay zekâ ile belirlenen tür zenginliği")
        title.setObjectName("sec_title")
        nv.addWidget(title)
        lab = QLabel(ai_richness.SCOPE_NOTE)
        lab.setObjectName("cap")
        lab.setTextFormat(Qt.RichText)
        lab.setWordWrap(True)
        nv.addWidget(lab)
        cov = ai_richness.coverage_text(res["modeller"], self.summary.n_species_observed)
        if cov:
            c = QLabel(f"<span style='color:#b45309'><b>Dikkat: </b></span>{cov} Kapsam dışı türler bu tabloda "
                       "değerlendirilemez.")
            c.setObjectName("cap")
            c.setTextFormat(Qt.RichText)
            c.setWordWrap(True)
            nv.addWidget(c)
        widgets.append(note)
        if not res["satirlar"]:
            msg = "Kayıtlı bir model sonucu yok. Önce modellerin eğitim penceresinde çapraz doğrulamalı eğitim çalıştırın."
            widgets.append(self._note_card("Veri yok", msg))
        else:
            tcard, tv = self._card()
            tt = QLabel("Modeller ve güven eşikleri: uzman ve model tür zenginliği")
            tt.setObjectName("sec_title")
            tv.addWidget(tt)
            tbl = _FitWidthTable(len(res["satirlar"]), len(ai_richness.COLUMNS))
            tbl.setHorizontalHeaderLabels(list(ai_richness.COLUMNS))
            for j, tip in enumerate(ai_richness.COLUMN_HELP):
                tbl.horizontalHeaderItem(j).setToolTip(tip)
            group, last = -1, None
            for i, r in enumerate(res["satirlar"]):
                if r["model_id"] != last:                      # model değişince zemin rengi değişir: gruplar ayırt edilir
                    group, last = group + 1, r["model_id"]
                bg = QColor("#f1f5fb") if group % 2 else QColor("#ffffff")
                for j, v in enumerate(r["hucreler"]):
                    it = QTableWidgetItem(str(v))
                    it.setBackground(bg)
                    it.setTextAlignment(Qt.AlignVCenter | (Qt.AlignLeft if j == 0 else Qt.AlignCenter))
                    if j == 0:
                        f_ = it.font()
                        f_.setBold(True)
                        it.setFont(f_)
                    tbl.setItem(i, j, it)
            tbl.setEditTriggers(QTableWidget.NoEditTriggers)
            tbl.verticalHeader().setVisible(False)
            tbl.setTextElideMode(Qt.ElideNone)
            tbl.setWordWrap(False)
            hdr = tbl.horizontalHeader()
            hdr.setMinimumHeight(34)
            tbl.refit()
            tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            tbl.setFixedHeight(max(hdr.height(), 34) + sum(tbl.rowHeight(i) for i in range(tbl.rowCount())) + 24)
            from PyQt5.QtWidgets import QSizePolicy as _SP
            tbl.setSizePolicy(_SP.Ignored, _SP.Fixed)      # kart genişliğini tablo belirlemesin; tablo kendi içinde kayar
            tv.addWidget(tbl)
            for cap in self._caption(
                    "Her model için, uzman etiketleriyle aynı noktalarda belirlenen tür sayıları ve (nokta, tür) çiftlerinde uyum.",
                    "Jaccard 1'e, nokta başı |fark| 0'a yaklaştıkça model uzmanın tür listesini daha iyi yeniden üretir. "
                    "Güven eşiği yükseldikçe sahte tür azalır, kaçırılan tür artar; ön tanımlı değer seçilmez, hepsi raporlanır.",
                    "Modeller arası farklar küçükse ve ± değerleri örtüşüyorsa üstünlük iddia edilmez."):
                tv.addWidget(cap)
            widgets.append(tcard)
            figs = []
            for m in res["modeller"]:
                if m["nokta"]:
                    figs.append(self._chart_card(
                        _rv.make_scatter_figure(m["nokta"], m["ad"], compact=True),
                        f"Nokta başına tür sayısı — {m['ad']}",
                        "Her nokta için uzman ve modelin belirlediği tür sayısı.",
                        "Köşegen = aynı sayıda tür (aynı türler olması gerekmez). Üstü sahte tür, altı kaçırılan tür.",
                        "Güven eşiği yok, ilk tekrar.", max_px=700))
            for f_ in figs:        # alt alta: sayfa yatayda taşmaz, metin kartları kırpılmaz
                widgets.append(self._row([f_]))
        for w_ in res["uyarilar"]:
            widgets.append(self._note_card("Uyarı", w_))
        return self._page(widgets)

    def _insights_card(self) -> QFrame:
        card, v = self._card()
        t = QLabel("Başlıca bulgular")
        t.setObjectName("sec_title")
        v.addWidget(t)
        sub = QLabel("Sayılar veri setinden otomatik üretilir; her cümle yalnızca belirlenen türleri ve seyreltilmiş değerleri anlatır.")
        sub.setObjectName("cap")
        sub.setWordWrap(True)
        v.addWidget(sub)
        colors = {"ok": RENK["yesil"], "info": RENK["mavi"], "warn": RENK["turuncu"]}
        for ins in richness_insights.compute_insights(self.summary):
            r = QHBoxLayout()
            r.setSpacing(10)
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{colors.get(ins.kind, RENK['mavi'])}; font-size:16px;")
            dot.setAlignment(Qt.AlignTop)
            dot.setFixedWidth(18)
            lab = QLabel(f"<b>{ins.title}</b><br><span style='color:#4b5563'>{ins.text}</span>")
            lab.setTextFormat(Qt.RichText)
            lab.setWordWrap(True)
            lab.setStyleSheet("font-size:13px;")
            r.addWidget(dot)
            r.addWidget(lab, 1)
            v.addLayout(r)
        return card

    def _method_card(self) -> QFrame:
        card, v = self._card()
        t = QLabel("Yöntem")
        t.setObjectName("sec_title")
        v.addWidget(t)
        for head, text in (
            ("Birim:", "Habitat_ID noktası. Bir tür, bir noktada en az bir segmentte belirlenmişse o noktada vardır."),
            ("Karşılaştırma:", "Gruplar karşılaştırma için aynı sayıda noktaya seyreltilir "
                               "(Coleman 1982; Colwell, Mao ve Chang 2004). Seyreltme, noktaların rastgele sırayla "
                               "eklenmesiyle beklenen tür sayısını hesaplar."),
            ("±1 s.s.:", "Yalnızca eldeki noktaların farklı sıralanmasından kaynaklanan değişkenliği gösterir; "
                         "örnekleme belirsizliği (hangi noktaların etiketlendiği) bu aralığa girmez."),
            ("Tür birikim eğrisi:", "Etiketlenen nokta sayısı arttıkça belirlenen tür sayısı. Eğri son noktada biter; "
                                    "ötesi için tahmin üretilmez."),
            ("Kapsam:", "Yalnızca etiketlenmiş noktalarda belirlenen türler gösterilir; etiketlenmemiş noktalar hakkında "
                        "bilgi verilmez. Sonuçlar standart örnekleme koşullarında akustik olarak tespit edilen tür "
                        "zenginliği içindir."),
        ):
            lab = QLabel(f"<b>{head}</b> {text}")
            lab.setTextFormat(Qt.RichText)
            lab.setWordWrap(True)
            lab.setStyleSheet("font-size:13px;")
            v.addWidget(lab)
        return card

    def _report_card(self) -> QFrame:
        card, v = self._card()
        t = QLabel("Ayrıntılı rapor (metin)")
        t.setObjectName("sec_title")
        v.addWidget(t)
        lab = QLabel(self.summary.format_richness_summary())
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lab.setStyleSheet("font-size:13px; color:#1f2a37;")
        v.addWidget(lab)
        return card

    # ------------------------------------------------------------------ ana düzen
    def _build_ui(self):
        s = self.summary
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 12)
        outer.setSpacing(12)
        outer.addWidget(self._hero())

        if not s.n_points:
            outer.addWidget(self._note_card("Veri yok", "Habitat_ID ile etiketlenmiş nokta bulunamadı."))
            outer.addStretch()
            outer.addLayout(self._make_button_row())
            return

        outer.addWidget(self._kpi_row())

        eff = s.point_effort
        has_eff = eff is not None and len(eff) > 0
        inc = s.point_incidence
        has_inc = inc is not None and getattr(inc, "size", 0) > 0
        loc = s.location_rarefaction
        pc = s.pooled_curve
        has_curve = pc is not None and len(pc) > 0

        acc = self._chart_card(
            richness_plots.accumulation(pc),
            "Yeni nokta eklendikçe kaç yeni tür belirleniyor?",
            "Noktalar rastgele sırayla eklendiğinde belirlenen toplam tür sayısı.",
            "Çizgi hâlâ yukarı çıkıyorsa yeni noktalar yeni tür getirmeye devam ediyor, yani tür envanteri tamamlanmamıştır. "
            "Çizgi yataylaşırsa yeni noktalar artık az sayıda yeni tür getiriyor demektir. Sağ alttaki sayı, son noktalarda "
            "nokta başına gelen yeni tür sayısıdır.",
            "Betimleyicidir: eğri son noktada biter, ötesi için bir sayı üretilmez. "
            "Çizginin üzerine (özellikle uçtaki yuvarlağa) tıklayınca o noktadaki nokta ve tür sayısı yazılır.") if has_curve else None
        if acc is not None:
            ac = acc.findChild(FigureCanvas)
            if ac is not None:
                ac.setCursor(QCursor(Qt.PointingHandCursor))
                ac.mpl_connect("button_press_event", lambda ev, c=ac: self._on_accumulation_click(c, ev))
        hist = self._chart_card(
            richness_plots.species_incidence_histogram(inc),
            "Türler kaç noktada belirlendi?",
            "Her çubuk bir nokta sayısını temsil eder; yüksekliği, o kadar noktada belirlenen tür sayısıdır. "
            "Örnek: x=1 çubuğu 31 ise, 31 tür yalnızca tek bir noktada belirlenmiştir.",
            "Sol çubuk yüksekse türlerin çoğu tek bir noktada belirlenmiştir (turuncu). Sağa uzanan çubuklar birçok noktada "
            "görülen yaygın türlerdir. Çubuğu olmayan bir değerde (örn. 9) o kadar noktada belirlenen tür yoktur.",
            "Bir çubuğa (ya da boş bir değere) tıklayınca o gruptaki türler ve noktaları listelenir.") if has_inc else None
        if hist is not None:
            hc = hist.findChild(FigureCanvas)
            if hc is not None:
                hc.setCursor(QCursor(Qt.PointingHandCursor))
                hc.mpl_connect("button_press_event", self._on_incidence_click)
        tab1 = self._page([self._insights_card(), self._row([acc, hist])])

        pts = self._chart_card(
            richness_plots.point_richness_chart(s.habitat_richness, all_types=[str(v) for v in s.point_types.values]
                                                if getattr(s, "point_types", None) is not None else None),
            "Hangi noktada kaç tür belirlendi?",
            "Her noktada belirlenen farklı tür sayısını; çubuk rengi habitat tipidir.",
            "Çubuğun yanındaki parantez habitat tipidir. En uzun çubuk en çok tür belirlenen noktadır.",
            "Noktalar ham tür sayısıyla sıralanmaz; noktalara düşen segment sayıları farklıdır.")
        tab2 = self._page([self._row([pts])])

        hab = self._chart_card(
            richness_plots.habitat_type_richness(s.type_richness, inc, s.point_types),
            "Habitat tipleri arasında tür zenginliği farklı mı?",
            "Her habitat tipi aynı sayıda noktaya indirildiğinde beklenen tür sayısını (nokta) ve ±1 s.s. (çubuk).",
            "Noktalar ve çubukları yatayda birbirinden ayrışıyorsa fark vardır; çubuklar üst üste biniyorsa tipler ayırt edilemez.",
            "Tip başına nokta sayısı az olduğunda sonuçlar kesin değildir.") if s.type_richness else None
        reg = self._chart_card(
            richness_plots.location_rarefaction(loc),
            "Bölgeler arasında tür zenginliği farklı mı?",
            "Bölgeler aynı sayıda noktaya indirildiğinde beklenen tür sayısını.",
            "Hata çubuğu yoksa o bölgenin tüm noktaları kullanılmıştır (seyreltme belirsizliği yoktur).",
            "Az noktalı bölge (örn. 2 nokta) için karşılaştırma sınırlıdır.") if loc is not None and len(loc) > 1 \
            else self._note_card("Bölge karşılaştırması", "Karşılaştırılacak ikinci bir bölge yok.")
        tab3 = self._page([self._habitat_limits_card(), self._row([hab, reg])])

        heat = self._chart_card(
            richness_plots.presence_heatmap(inc, s.point_types),
            "Hangi tür hangi noktada belirlendi?",
            "Tüm türler (tek noktada görülenler dahil) satırlarda, noktalar sütunlarda; mavi kare türün o noktada belirlendiğini gösterir.",
            "Satırlar yaygınlığa, sütunlar nokta başına tür sayısına göre sıralıdır; sütun üstündeki renkli şerit habitat tipidir (şeritteki numara, üstteki listedeki habitat tipinin numarasıdır).",
            None, max_px=3000) if has_inc else None
        tab4 = self._page([self._row([heat])])

        tab5 = self._page([self._method_card(), self._report_card()])
        tab_ai = self._ai_page()

        tabs = _FitTabWidget()
        self._tabs = tabs
        tabs.setDocumentMode(True)
        tabs.tabBar().setElideMode(Qt.ElideNone)
        tabs.tabBar().setUsesScrollButtons(True)      # dar pencerede sekmeler kırpılmaz, kaydırma okları çıkar
        tabs.tabBar().setExpanding(False)
        for page, name in ((tab1, "Genel Bakış"), (tab2, "Noktalar"), (tab3, "Habitat / Bölge"),
                           (tab4, "Tür Yapısı"), (tab_ai, "Yapay Zekâ"),
                           (tab5, "Yöntem / Rapor")):
            tabs.addTab(page, name)
        outer.addWidget(tabs, 1)
        outer.addLayout(self._make_button_row())

    def _on_refresh(self):
        parent = self.parent()
        if parent is None:
            return
        csv_full_path = os.path.join(csv_project_root(), parent.csv_file) \
            if not os.path.isabs(parent.csv_file) else parent.csv_file
        df = dataset_summary.load_dataset(csv_full_path)
        if df is None:
            QMessageBox.information(self, "Tür Zenginliği Belirleme", "Henüz veri seti (CSV) bulunamadı ya da dosya boş.")
            return
        self.summary = dataset_summary.compute_summary(df, config=coverage_report.load_project_config())
        self.close()
        new_dialog = SpeciesRichnessDialog(self.summary, parent=parent)
        parent._species_richness_dialog = new_dialog
        new_dialog.show()


class HabitatInfoDialog(SpeciesRichnessDialog):
    """
    Habitat Bilgileri: habitat_data.json'daki tüm tanımlı noktaların (habitat tipi, bölge, koordinat) envanteri ve
    bunların etiketleme kapsamı. SpeciesRichnessDialog'un kart/sayfa yardımcılarını kullanır ama kendi verisiyle çalışır.
    Bir ENVANTER ve KAPSAM ekranıdır; habitatlar arası tür zenginliği karşılaştırması burada yapılmaz.
    """

    HERO_SUBTITLE = "Tanımlı noktalar, habitat tipleri ve bölgeler; etiketleme kapsamı"
    _DURUM = ("Tümü", "Etiketli", "Etiketsiz")

    def __init__(self, result: dict, parent=None):
        _BaseSummaryDialog.__init__(self, "🌿 Habitat Bilgileri", parent)   # SpeciesRichnessDialog.__init__ atlanır (özet gerektirir)
        self.result = result
        self.resize(1400, 900)
        self.setMinimumWidth(900)
        self.setStyleSheet(self._CSS)
        self._build_ui()

    # ------------------------------------------------------------------ yardımcılar
    def _hero(self) -> QFrame:
        return theme.make_hero("Habitat Bilgileri", self.HERO_SUBTITLE)

    @staticmethod
    def _num_item(value, fmt=None):
        it = QTableWidgetItem()
        if value is None or (isinstance(value, float) and value != value):
            it.setData(Qt.DisplayRole, "—")
        elif isinstance(value, (int, np.integer)):
            it.setData(Qt.DisplayRole, int(value))
        elif isinstance(value, (float, np.floating)):
            it.setData(Qt.DisplayRole, float(round(value, 1)) if fmt is None else fmt.format(value))
        else:
            it.setData(Qt.DisplayRole, str(value))
        it.setTextAlignment(Qt.AlignCenter if not isinstance(value, str) else Qt.AlignLeft | Qt.AlignVCenter)
        return it

    def _make_table(self, headers, rows, tips=None, max_rows_visible=None) -> QTableWidget:
        tbl = QTableWidget(len(rows), len(headers))
        tbl.setHorizontalHeaderLabels(list(headers))
        for j, tip in enumerate(tips or []):
            tbl.horizontalHeaderItem(j).setToolTip(tip)
        tbl.setSortingEnabled(False)
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                tbl.setItem(i, j, self._num_item(v))
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setSelectionBehavior(QTableWidget.SelectRows)
        tbl.setAlternatingRowColors(True)
        tbl.verticalHeader().setVisible(False)
        tbl.setTextElideMode(Qt.ElideNone)
        tbl.setWordWrap(False)
        hdr = tbl.horizontalHeader()
        hdr.setMinimumHeight(34)
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        tbl.resizeColumnsToContents()
        hdr.setSortIndicator(-1, Qt.AscendingOrder)       # veri sırası korunur; başlığa tıklayınca sıralanır
        tbl.setSortingEnabled(True)
        tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        n = len(rows) if max_rows_visible is None else min(len(rows), max_rows_visible)
        tbl.setFixedHeight(max(hdr.height(), 34) + sum(tbl.rowHeight(i) for i in range(n)) + 24)
        tbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)     # kart genişliğini tablo belirlemesin
        return tbl

    def _table_card(self, title, tbl, caps=()) -> QFrame:
        card, v = self._card()
        t = QLabel(title)
        t.setObjectName("sec_title")
        v.addWidget(t)
        v.addWidget(tbl)
        for c in caps:
            v.addWidget(c)
        return card

    def _kpis(self) -> QWidget:
        o = self.result.get("ozet") or {}
        cards = [
            (str(o.get("n_tanimli", 0)), "Tanımlı nokta", "habitat_data.json", RENK["acik_mavi"]),
            (str(o.get("n_etiketli", 0)), "Etiketli nokta", f"tanımlı noktaların %{o.get('kapsam', 0):.0f}'i", RENK["mavi"]),
            (f"{o.get('n_segment', 0):,}".replace(",", "."), "Etiketli segment",
             f"nokta başına ortanca {o.get('seg_ortanca', float('nan')):.0f}" if o.get("n_etiketli") else "—", RENK["mor"]),
            (f"{o.get('n_tip_etiketli', 0)} / {o.get('n_tip', 0)}", "Habitat tipi", "etiketli / tanımlı", RENK["yesil"]),
            (f"{o.get('n_bolge_etiketli', 0)} / {o.get('n_bolge', 0)}", "Bölge", "etiketli / tanımlı", RENK["turuncu"]),
        ]
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)
        for c in cards:
            h.addWidget(self._kpi(*c), 1)
        return row

    def _insights_card_habitat(self) -> QFrame:
        card, v = self._card()
        t = QLabel("Başlıca bulgular")
        t.setObjectName("sec_title")
        v.addWidget(t)
        sub = QLabel("Cümleler veri setinden otomatik üretilir; veri güncellendikçe değişir.")
        sub.setObjectName("cap")
        v.addWidget(sub)
        colors = {"ok": RENK["yesil"], "info": RENK["mavi"], "warn": RENK["turuncu"]}
        for kind, title, text in habitat_info.insights(self.result):
            r = QHBoxLayout()
            r.setSpacing(10)
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{colors.get(kind, RENK['mavi'])}; font-size:16px;")
            dot.setAlignment(Qt.AlignTop)
            dot.setFixedWidth(18)
            lab = QLabel(f"<b>{title}</b><br><span style='color:#4b5563'>{text}</span>")
            lab.setTextFormat(Qt.RichText)
            lab.setWordWrap(True)
            lab.setStyleSheet("font-size:13px;")
            r.addWidget(dot)
            r.addWidget(lab, 1)
            v.addLayout(r)
        return card

    # ------------------------------------------------------------------ sayfalar
    def _types_page(self) -> QScrollArea:
        tp = self.result["tipler"]
        heads = ("Habitat tipi", "Tanımlı nokta", "Etiketli nokta", "Kapsam (%)", "Segment", "Tür", "Segment/nokta (ortanca)",
                 "En az", "En çok", f"Düşük çaba (<{habitat_info.LOW_EFFORT_SEGMENTS})")
        tips = ("Habitat tipi.", "habitat_data.json'da bu tipe ait nokta sayısı.", "CSV'de en az bir segmenti olan nokta sayısı.",
                "Etiketli / tanımlı nokta.", "Bu tipteki etiketli segment sayısı.", "Bu tipte belirlenen farklı tür sayısı.",
                "Etiketli noktalarda segment sayısının ortancası.", "Etiketli noktalar içinde en az segment.",
                "Etiketli noktalar içinde en çok segment.",
                "Segment sayısı eşikten az olan etiketli nokta sayısı: bu noktalar doğrudan karşılaştırılamaz.")
        rows = [[r.habitat_tipi, int(r.tanimli), int(r.etiketli), float(r.kapsam), int(r.segment), int(r.tur),
                 float(r.seg_ortanca), int(r.seg_min), int(r.seg_maks), int(r.dusuk_cabali)] for r in tp.itertuples()]
        tbl = self._make_table(heads, rows, tips)
        caps = self._caption(
            "Her habitat tipinin tanımlı nokta sayısı ve ne kadarının etiketlendiği.",
            "Kapsam ve nokta başı segment dengesizse, tipler arası sayısal karşılaştırma örnekleme çabasını yansıtır, tür zenginliğini değil.",
            "Tür sütunu, tipteki segment sayısına bağlıdır; farklı çabalı tipler bu sütunla karşılaştırılmamalıdır.")
        cov = self._chart_card(habitat_plots.coverage_by_type(tp), "Tanımlı ve etiketli nokta sayısı",
                               "Her tip için tanımlı (gri) ve etiketli (mavi) nokta sayısı.",
                               "Mavi çubuk gri çubuğa ne kadar yakınsa tip o kadar çok noktada örneklenmiştir.",
                               "Etiketsiz nokta 'tür yok' demek değildir.", max_px=760)
        seg = self._chart_card(habitat_plots.segments_per_point(self.result["noktalar"]),
                               "Etiketli noktalarda segment sayısı",
                               "Her etiketli nokta bir daire; yatay konum o noktadaki segment sayısıdır.",
                               "Daireler tip içinde dağınıksa noktalar arası çaba eşit değildir. Turuncu daireler düşük çabalı noktalardır.",
                               None, max_px=760)
        return self._page([self._table_card("Habitat tiplerine göre envanter", tbl, caps),
                           self._row([cov]), self._row([seg])])      # alt alta: sayfa yatayda taşmaz

    def _regions_page(self) -> QScrollArea:
        bl = self.result["bolgeler"]
        heads = ("Bölge", "Tanımlı nokta", "Etiketli nokta", "Kapsam (%)", "Segment", "Tür", "Segment/nokta (ortanca)",
                 "En az", "En çok", f"Düşük çaba (<{habitat_info.LOW_EFFORT_SEGMENTS})")
        rows = [[r.konum, int(r.tanimli), int(r.etiketli), float(r.kapsam), int(r.segment), int(r.tur),
                 float(r.seg_ortanca), int(r.seg_min), int(r.seg_maks), int(r.dusuk_cabali)] for r in bl.itertuples()]
        tbl = self._make_table(heads, rows)
        caps = self._caption(
            "Her bölgenin tanımlı nokta sayısı ve etiketleme kapsamı.",
            "Etiketli noktası olmayan bölge için hiçbir sonuç üretilemez; az noktalı bölgede sonuçlar kesin değildir.",
            "Bölgeler arası fark yorumu için nokta sayısının yeterli olduğu ayrıca gösterilmelidir.")
        mat = self._chart_card(habitat_plots.region_type_matrix(self.result["matris_tanimli"], self.result["matris_etiketli"]),
                               "Bölge × habitat tipi: etiketli / tanımlı nokta",
                               "Her hücrede etiketli nokta sayısı / tanımlı nokta sayısı; renk etiketli oranıdır.",
                               "Koyu hücreler iyi örneklenmiştir; açık hücreler boşluğu gösterir. '—' o bölgede o tipin tanımlı olmadığını belirtir.",
                               "Bölge ve tip aynı anda karşılaştırılacaksa her hücrede yeterli nokta bulunmalıdır.", max_px=1100)
        return self._page([self._table_card("Bölgelere göre envanter", tbl, caps), self._row([mat])])

    def _points_page(self) -> QScrollArea:
        pts = self.result["noktalar"]
        card, v = self._card()
        t = QLabel("Tüm tanımlı noktalar")
        t.setObjectName("sec_title")
        v.addWidget(t)
        frow = QHBoxLayout()
        frow.setSpacing(10)
        self._cmb_durum = QComboBox()
        self._cmb_durum.addItems(self._DURUM)
        self._cmb_tip = QComboBox()
        self._cmb_tip.addItems(["Tüm habitat tipleri"] + sorted(pts["habitat_tipi"].unique().tolist()))
        self._cmb_konum = QComboBox()
        self._cmb_konum.addItems(["Tüm bölgeler"] + sorted(pts["konum"].unique().tolist()))
        for lbl, w in (("Durum:", self._cmb_durum), ("Habitat tipi:", self._cmb_tip), ("Bölge:", self._cmb_konum)):
            frow.addWidget(QLabel(lbl))
            frow.addWidget(w)
        self._lbl_count = QLabel("")
        self._lbl_count.setObjectName("cap")
        frow.addStretch()
        frow.addWidget(self._lbl_count)
        v.addLayout(frow)
        self._pts_tbl = QTableWidget()
        self._pts_tbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._pts_tbl.setFixedHeight(560)
        self._pts_tbl.cellDoubleClicked.connect(self._open_map_row)
        v.addWidget(self._pts_tbl)
        for c in self._caption("Her tanımlı noktanın habitat tipi, bölgesi, koordinatı ve etiketleme durumu.",
                               "Bir satıra çift tıklayınca nokta Google Haritalar'da açılır. Başlığa tıklayarak sıralayabilirsiniz.",
                               "Segment, tür ve kayıt sütunları yalnızca etiketli noktalar içindir; etiketsiz nokta 0 gösterir."):
            v.addWidget(c)
        for w in (self._cmb_durum, self._cmb_tip, self._cmb_konum):
            w.currentIndexChanged.connect(self._fill_points)
        self._fill_points()
        return self._page([card])

    def _fill_points(self, *_):
        pts = self.result["noktalar"]
        d = self._cmb_durum.currentText()
        if d == "Etiketli":
            pts = pts[pts["etiketli"]]
        elif d == "Etiketsiz":
            pts = pts[~pts["etiketli"]]
        if self._cmb_tip.currentIndex() > 0:
            pts = pts[pts["habitat_tipi"] == self._cmb_tip.currentText()]
        if self._cmb_konum.currentIndex() > 0:
            pts = pts[pts["konum"] == self._cmb_konum.currentText()]
        heads = ["Nokta", "Habitat tipi", "Kod", "Bölge", "Durum", "Segment", "Tür", "Kayıt", "Enlem", "Boylam"]
        tbl = self._pts_tbl
        tbl.setSortingEnabled(False)
        tbl.clear()
        tbl.setColumnCount(len(heads))
        tbl.setHorizontalHeaderLabels(heads)
        tbl.setRowCount(len(pts))
        self._urls = {}
        for i, r in enumerate(pts.itertuples()):
            vals = [int(r.nokta), r.habitat_tipi, r.kodu, r.konum, "Etiketli" if r.etiketli else "Etiketsiz",
                    int(r.segment), int(r.tur), int(r.kayit),
                    None if r.lat is None else f"{r.lat:.5f}", None if r.lon is None else f"{r.lon:.5f}"]
            for j, val in enumerate(vals):
                it = self._num_item(val)
                if j == 4 and not r.etiketli:
                    it.setForeground(QColor("#9ca3af"))
                if j == 0:
                    it.setData(Qt.UserRole, r.google_maps)
                tbl.setItem(i, j, it)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setSelectionBehavior(QTableWidget.SelectRows)
        tbl.setAlternatingRowColors(True)
        tbl.verticalHeader().setVisible(False)
        tbl.setTextElideMode(Qt.ElideNone)
        hdr = tbl.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setMinimumHeight(34)
        tbl.resizeColumnsToContents()
        hdr.setSortIndicator(-1, Qt.AscendingOrder)
        tbl.setSortingEnabled(True)
        self._lbl_count.setText(f"{len(pts)} / {len(self.result['noktalar'])} nokta")

    def _open_map_row(self, row, _col):
        it = self._pts_tbl.item(row, 0)
        url = it.data(Qt.UserRole) if it is not None else None
        if url:
            QDesktopServices.openUrl(QUrl(str(url)))

    def _method_card_habitat(self) -> QFrame:
        card, v = self._card()
        t = QLabel("Veri kaynağı ve yorum sınırı")
        t.setObjectName("sec_title")
        v.addWidget(t)
        for head, text in (
            ("Tanımlı nokta:", "habitat_data.json dosyasındaki her kayıt (habitat tipi, kod, bölge, koordinat)."),
            ("Etiketli nokta:", "bird_sound_dataset.csv'de Habitat_ID ile eşleşen en az bir segmenti olan nokta."),
            ("Etiketsiz nokta:", "Henüz etiketlenmemiştir; 'tür yok' anlamına gelmez."),
            ("Kapsam:", "Bu ekran envanter ve kapsam bilgisidir. Habitat tipleri veya bölgeler arasında tür zenginliği farkı "
                        "için Tür Zenginliği Belirleme ekranındaki seyreltilmiş karşılaştırma kullanılmalıdır."),
            ("Düşük çaba:", f"{habitat_info.LOW_EFFORT_SEGMENTS}'ten az segmenti olan etiketli nokta; Konum Haritası'ndaki işaretle aynı eşik."),
        ):
            lab = QLabel(f"<b>{head}</b> {text}")
            lab.setTextFormat(Qt.RichText)
            lab.setWordWrap(True)
            lab.setStyleSheet("font-size:13px;")
            v.addWidget(lab)
        return card

    def _build_ui(self):
        r = self.result
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 12)
        outer.setSpacing(12)
        outer.addWidget(self._hero())
        if r["noktalar"].empty:
            outer.addWidget(self._note_card("Veri yok", "habitat_data.json bulunamadı ya da boş."))
            outer.addStretch()
            outer.addLayout(self._make_button_row())
            return
        outer.addWidget(self._kpis())
        warn = [self._note_card("Uyarı", w) for w in r["uyarilar"]]
        tab1 = self._page([self._insights_card_habitat()] + warn + [self._method_card_habitat()])
        tabs = _FitTabWidget()
        self._tabs = tabs
        tabs.setDocumentMode(True)
        tabs.tabBar().setElideMode(Qt.ElideNone)
        tabs.tabBar().setUsesScrollButtons(True)
        tabs.tabBar().setExpanding(False)
        for page, name in ((tab1, "Genel Bakış"), (self._types_page(), "Habitat Tipleri"),
                           (self._regions_page(), "Bölgeler"), (self._points_page(), "Noktalar")):
            tabs.addTab(page, name)
        outer.addWidget(tabs, 1)
        outer.addLayout(self._make_button_row())

    def _on_refresh(self):
        parent = self.parent()
        if parent is None:
            return
        new_dialog = HabitatInfoDialog(parent._compute_habitat_info(), parent=parent)
        self.close()
        parent._habitat_info_dialog = new_dialog
        new_dialog.show()


class ClassificationDiagnosisDialog(_BaseSummaryDialog):
    """
    Başarı sınırı analizi: başarının sınırı model mi veri mi? Veri yeterliliği,
    yöntem karşılaştırması ve tür bazlı başarı (çapraz doğrulama dosyaları
    varsa) tek pencerede; sayılar veri setinden/dosyalardan hesaplanır.
    """

    HERO_SUBTITLE = "Başarının sınırı: model mi, veri mi?"

    def __init__(self, diagnosis: "classification_diagnosis.Diagnosis", parent=None):
        super().__init__("🔬 Başarı Sınırı Analizi", parent)
        self.diagnosis = diagnosis
        self._build_ui()

    def _build_ui(self):
        d = self.diagnosis
        self._build_shell(d.format())
        # Sonuçlar, her modelin kendi eğitim penceresindeki kayıtlı çapraz doğrulamadan okunur (burada eğitim yapılmaz).
        self.lbl_source = QLabel(
            "Sonuçlar BirdNET 2.4, Perch 2.0 ve BirdNET + Perch modellerinin <b>kayıtlı çapraz doğrulamasından</b> "
            "okunur. Güncellemek için ilgili modelin eğitim penceresinde eğitimi çalıştırın, sonra bu pencerede "
            "Yenile'ye basın.")
        self.lbl_source.setWordWrap(True)
        self.lbl_source.setTextFormat(Qt.RichText)
        self.lbl_source.setStyleSheet("QLabel { background:#eef2fb; padding:8px 12px; border-radius:6px; font-size:12px; }")
        self.layout().insertWidget(1, self.lbl_source)
        if len(d.species_table):
            self._add_custom_figure(classification_plots.data_sufficiency(
                d.species_table, d.min_segment, d.min_kayit))
        if d.has_cv and d.methods:
            self._add_custom_figure(classification_plots.methods_accuracy_f1(d.methods))
        if d.has_cv and "f1" in d.species_table.columns and d.species_table["f1"].notna().sum() >= 3:
            self._add_custom_figure(classification_plots.support_vs_f1(d.species_table, d.rho, p=d.rho_p))

    def _on_refresh(self):
        parent = self.parent()
        if parent is None:
            return
        new = parent._compute_classification_diagnosis()
        if new is None:
            return
        self.close()
        parent._classification_diagnosis_dialog = ClassificationDiagnosisDialog(new, parent=parent)
        parent._classification_diagnosis_dialog.show()


class LocationMapDialog(_BaseSummaryDialog):
    """
    `Lat`/`Lon` sütunlarındaki kayıt noktalarını (Burdur/Isparta/Antalya gibi
    bölgeler) gösteren ayrı pencere. Nokta boyutu o noktadaki kayıt sayısına,
    rengi tür zenginliğine göre ölçeklenir. `contextily` kuruluysa gerçek harita
    karoları eklenir; kurulu değilse (veya internet yoksa) sessizce düz bir
    koordinat serpme grafiğine geçilir.
    """

    HERO_SUBTITLE = "Kayıt noktalarının etkileşimli haritası"

    def __init__(self, points: list, parent=None):
        super().__init__("🗺️ Konum Haritası", parent)
        self.points = points
        self._build_ui()

    def _build_ui(self):
        # Diğer pencerelerin (Veri Seti Özeti, Tür Zenginliği) kullandığı
        # _build_shell/_add_custom_figure yolu bilerek kullanılmaz: o yol,
        # kaydırılabilir listede SABİT boyutlu birden çok çubuk grafiği alt alta
        # göstermek içindir. Harita ise tek bir etkileşimli görsel olduğundan
        # (fare tekerleğiyle yakınlaştırma, sürükleme) pencere boyutuna
        # uyan ayrı bir yerleşim kurulur.
        outer_layout = QVBoxLayout(self)
        self._add_hero(outer_layout)

        n_points = len(self.points)
        total_records = sum(p.record_count for p in self.points)

        self._try_build_webengine_map(outer_layout, n_points, total_records)

        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("🔄 Yenile")
        btn_refresh.clicked.connect(self._on_refresh)
        self.btn_fullscreen = QPushButton("🔲 Tam Ekran")
        self.btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        btn_close = QPushButton("Kapat")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(btn_refresh)
        btn_row.addWidget(self.btn_fullscreen)
        btn_row.addWidget(self._make_save_charts_button())
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        outer_layout.addLayout(btn_row)

    def _try_build_webengine_map(self, outer_layout, n_points: int, total_records: int):
        """
        Önce GERÇEK, dünya çapında etkileşimli bir harita kurmayı dener
        (folium + QWebEngineView — Leaflet.js tabanlı, karoları dinamik yükler).
        `PyQtWebEngine` kurulu değilse sessizce eski statik matplotlib+contextily
        yaklaşımına geçer (fare tekerleğiyle yakınlaştırılabilir, ancak tüm
        dünyada kaydırılamaz).
        """
        self._using_webengine = False
        try:
            from PyQt5.QtWebEngineWidgets import QWebEngineView
            from PyQt5.QtCore import QUrl

            if n_points > 0:
                summary_text = (
                    f"{n_points} farklı kayıt noktası (Habitat_ID/Location'a göre gruplu), "
                    f"toplam {total_records} kayıt. Gerçek, dünya çapında etkileşimli bir harita "
                    "— fare tekerleğiyle yakınlaştırıp uzaklaştırabilir, sürükleyerek dünyanın her yerine gidebilirsiniz. "
                    "Sağ üstten Harita, Arazi veya Uydu görünümünü seçebilirsiniz. "
                    "Ayrıntı için bir noktaya tıklayın. Daire boyutu etiketli segment sayısını (çaba), "
                    "rengi belirlenen tür sayısını gösterir; 5'ten az segmentli noktalar kesikli kenarlıdır."
                )
            else:
                summary_text = (
                    "Henüz Enlem/Boylam (Lat/Lon) değeri girilmiş kayıt yok.\n"
                    "Segmentleri etiketlerken 'Konum ve Habitat' sekmesindeki Enlem/Boylam alanlarını doldurun; "
                    "kayıtlar burada görünecektir."
                )
            self._add_text_label(outer_layout, summary_text)

            html = location_map.build_interactive_map_html(self.points)
            tmp_path = os.path.join(tempfile.gettempdir(), f"bioseg_location_map_{id(self)}.html")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(html)
            self._map_html_path = tmp_path

            self._map_view = QWebEngineView()
            self._map_view.load(QUrl.fromLocalFile(tmp_path))
            outer_layout.addWidget(self._map_view, stretch=1)
            self._using_webengine = True

        except ImportError:
            note = (
                "\n\n(Not: Gerçek dünya haritası için 'PyQtWebEngine' kurulu değil — şimdilik "
                "yalnızca verilen bölgeyi kapsayan statik bir harita gösteriliyor; dünya genelinde "
                "kaydırma yok. Kurmak için: pip install folium PyQtWebEngine)"
            )
            summary_text = (
                (
                    f"{n_points} farklı kayıt noktası (Habitat_ID/Location'a göre gruplu), "
                    f"toplam {total_records} kayıt.\n"
                    "Nokta boyutu: o noktadaki kayıt sayısı. Renk: o noktadaki tür zenginliği "
                    "(farklı tür sayısı). Fare tekerleği: yakınlaştır/uzaklaştır."
                    if n_points > 0 else
                    "Henüz Enlem/Boylam (Lat/Lon) değeri girilmiş kayıt yok."
                ) + note
            )
            self._add_text_label(outer_layout, summary_text)

            self._map_fig = location_map.build_map_figure(self.points, use_basemap=True)
            self._map_canvas = FigureCanvas(self._map_fig)
            self._map_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            toolbar = NavigationToolbar2QT(self._map_canvas, self)
            outer_layout.addWidget(toolbar)
            outer_layout.addWidget(self._map_canvas, stretch=1)
            self._map_canvas.mpl_connect('scroll_event', self._on_scroll_zoom)

    def _make_save_charts_button(self) -> QPushButton:
        """Harita penceresi için kaydetme menüsü (taban sınıftaki grafik menüsünün yerine geçer)."""
        btn = QPushButton("💾 Haritayı Kaydet")
        menu = QMenu(btn)
        if getattr(self, "_using_webengine", False):
            menu.addAction("Görünen haritayı PNG olarak kaydet (ekran çözünürlüğü)",
                           lambda checked=False: self._save_visible_map())
        menu.addAction("Statik harita — PNG 300 dpi (tez/yayın için)",
                       lambda checked=False: self._save_static_map("png", 300))
        menu.addAction("Statik harita — PDF (vektörel)",
                       lambda checked=False: self._save_static_map("pdf", None))
        btn.setMenu(menu)
        return btn

    def _ask_map_save_path(self, extension: str) -> str:
        default_name = f"konum_haritasi_{datetime.now().strftime('%Y%m%d_%H%M')}.{extension}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Haritayı kaydet", os.path.join(csv_project_root(), default_name),
            f"{extension.upper()} dosyası (*.{extension})",
        )
        return path

    def _save_visible_map(self):
        """İnteraktif haritanın ekranda görünen halini PNG olarak kaydeder (ekran çözünürlüğünde)."""
        path = self._ask_map_save_path("png")
        if not path:
            return
        pixmap = self._map_view.grab()
        if pixmap.save(path, "PNG"):
            self._report_saved_files(os.path.dirname(path), [path], [])
        else:
            QMessageBox.warning(self, "Haritayı Kaydet", f"Harita kaydedilemedi:\n{path}")

    def _save_static_map(self, fmt: str, dpi):
        """
        Haritayı statik (matplotlib) olarak yüksek çözünürlükte kaydeder. Harita karoları
        için `contextily` ve internet gerekir; yoksa koordinat grafiği kaydedilir.
        """
        path = self._ask_map_save_path(fmt)
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            fig = getattr(self, "_map_fig", None)
            created_here = fig is None
            if created_here:
                fig = location_map.build_map_figure(self.points, use_basemap=True)
            kwargs = {"bbox_inches": "tight", "facecolor": "white"}
            if dpi:
                kwargs["dpi"] = dpi
            fig.savefig(path, **kwargs)
            if created_here:
                plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Haritayı Kaydet", f"Harita kaydedilemedi:\n{exc}")
            return
        QApplication.restoreOverrideCursor()
        self._report_saved_files(os.path.dirname(path), [path], [])

    def _add_text_label(self, outer_layout, text: str):
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setStyleSheet(
            "QLabel { background:#eef6ee; padding:10px 14px; border-radius:6px; font-size: 15px; }"
        )
        outer_layout.addWidget(text_label)

    def _on_scroll_zoom(self, event):
        """Fare tekerleğiyle, imleç konumu merkez alınarak yakınlaştırma/uzaklaştırma
        (standart matplotlib tekniği)."""
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return

        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self._map_canvas.draw_idle()

    def _on_refresh(self):
        parent = self.parent()
        if parent is None:
            return
        csv_full_path = os.path.join(csv_project_root(), parent.csv_file) \
            if not os.path.isabs(parent.csv_file) else parent.csv_file
        df = dataset_summary.load_dataset(csv_full_path)
        if df is None:
            QMessageBox.information(self, "Konum Haritası", "Henüz veri seti (CSV) bulunamadı ya da dosya boş.")
            return
        new_points = location_map.extract_location_points(df)
        self.close()
        new_dialog = LocationMapDialog(new_points, parent=parent)
        parent._location_map_dialog = new_dialog
        new_dialog.show()


class ClassificationFoldersDialog(_BaseSummaryDialog):
    """
    `segments/` klasöründeki ses dosyalarını, veri setindeki Scientific_Name alanına
    göre `segments/classification/<tür>/` altında tür bazlı klasör yapısına
    KOPYALAYAN (orijinal `segments/` dosyalarına hiç dokunmayan) ayrı pencere.
    Klasörleri bir sonraki aşamaya (sınıflandırıcı eğitimi) hazırlamak için
    tasarlanmıştır.
    """

    HERO_SUBTITLE = "Segmentleri türlere göre klasörlere hazırlar"

    def __init__(self, parent=None):
        super().__init__("🗂️ Sınıflandırma Klasörleri", parent)
        self.result_data = None
        self._build_ui()

    def _build_ui(self):
        outer_layout = QVBoxLayout(self)
        self._add_hero(outer_layout)

        info_label = QLabel(
            "Bu araç, her tür için 'segments/classification/&lt;Scientific_Name&gt;/' altında bir klasör oluşturur "
            "ve veri setinizdeki (bird_sound_dataset.csv) her segmentin ses dosyasını bilimsel adına göre "
            "ilgili klasöre <b>KOPYALAR</b> — "
            "'segments/' klasöründeki orijinal dosyalar <b>asla silinmez veya taşınmaz</b>. "
            "Bir sonraki aşama olan sınıflandırıcı eğitimi için klasörleri hazırlamakta kullanılır.<br><br>"
            "Tekrar çalıştırmak güvenlidir — daha önce kopyalanmış ve değişmemiş dosyalar atlanır, "
            "yalnızca yeni eklenen segmentler kopyalanır."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            "QLabel { background:#eef6ee; padding:10px 14px; border-radius:6px; font-size: 15px; }"
        )
        outer_layout.addWidget(info_label)

        self.btn_run = QPushButton("▶ Klasörleri Oluştur / Güncelle")
        self.btn_run.clicked.connect(self._on_run)
        outer_layout.addWidget(self.btn_run)

        self.result_label = QLabel("Henüz çalıştırılmadı.")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("QLabel { padding: 8px 4px; font-size: 14.5px; }")
        outer_layout.addWidget(self.result_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(scroll_content)
        self._scroll_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        scroll.setWidget(scroll_content)
        outer_layout.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_fullscreen = QPushButton("🔲 Tam Ekran")
        self.btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        btn_close = QPushButton("Kapat")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_fullscreen)
        btn_row.addWidget(self._make_save_charts_button())
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        outer_layout.addLayout(btn_row)

    def _on_run(self):
        parent = self.parent()
        csv_full_path = None
        if parent is not None:
            csv_full_path = os.path.join(csv_project_root(), parent.csv_file) \
                if not os.path.isabs(parent.csv_file) else parent.csv_file

        try:
            result = classification_folders.build_classification_folders(csv_path=csv_full_path)
        except FileNotFoundError as exc:
            QMessageBox.information(self, "Sınıflandırma Klasörleri", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Sınıflandırma Klasörleri", f"Beklenmeyen hata: {exc}")
            return

        self.result_data = result
        self.result_label.setText(result.format_summary())

        for i in reversed(range(self._scroll_layout.count())):
            item = self._scroll_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)

        if result.species_counts:
            counts = pd.Series(result.species_counts).sort_values(ascending=False)
            canvas = self._make_bar_chart_canvas(
                counts, "Tür Klasörü Başına Dosya Sayısı", "Dosya sayısı",
                highlight_thresholds=False, bar_color="#2980b9",
            )
            self._scroll_layout.addWidget(canvas, alignment=Qt.AlignLeft)


class ModelPickerDialog(QDialog):
    """Tür sınıflandırma eğitimi için model seçici: her model kendi pencerede açılır; ayrıca karşılaştırma."""

    COMPARE = "karsilastirma"

    def __init__(self, parent=None, root=None):
        super().__init__(parent)
        self.setWindowTitle("Tür Sınıflandırma Eğitimi — model seçin")
        self.choice = None
        self.buttons = {}
        lay = QVBoxLayout(self)
        info = QLabel("Hangi modelin gömmeleri üzerinde sınıflandırıcı eğitilecek? Her model kendi pencerede ve kendi "
                      "sonuç klasöründe çalışır; birbirinin sonucunu değiştirmez.")
        info.setWordWrap(True)
        lay.addWidget(info)
        status = {}
        try:
            for spec, st in model_registry.stored_results(root or csv_project_root()):
                status[spec.id] = f"Kayıtlı sonuç: {last_result_store.format_time(st.saved_at)}"
        except Exception:  # noqa: BLE001
            pass
        style = ("QPushButton { text-align:left; padding:12px 16px; font-size:15px; border:1px solid #D7E0DE; "
                 "border-radius:8px; background:#ffffff; color:#1f2a37; } "
                 "QPushButton:hover { background:#E3F3EE; border-color:#0F8B6D; color:#1f2a37; } "
                 "QPushButton:pressed { background:#CFE6E2; color:#1f2a37; }")
        for mid, spec in model_registry.MODELS.items():
            btn = QPushButton(f"🧠  {spec.ad}\n{status.get(mid, 'Henüz eğitilmedi')}")
            btn.setStyleSheet(style)
            btn.clicked.connect(lambda _c=False, m=mid: self._pick(m))
            lay.addWidget(btn)
            self.buttons[mid] = btn
        cmp_btn = QPushButton("📊  Model Karşılaştırma\nKayıtlı sonuçları yan yana göster")
        cmp_btn.setStyleSheet(style)
        cmp_btn.clicked.connect(lambda _c=False: self._pick(self.COMPARE))
        lay.addWidget(cmp_btn)
        self.buttons[self.COMPARE] = cmp_btn
        self.setMinimumWidth(440)

    def _pick(self, choice):
        self.choice = choice
        self.accept()


class ModelComparisonDialog(_BaseSummaryDialog):
    """Kayıtlı model sonuçlarını (her model kendi pencerede eğitilir) tek tabloda yan yana gösterir.
    Yalnızca kayıtlı sonuçları okur; eğitim yapmaz, hiçbir dosyayı değiştirmez."""

    HERO_SUBTITLE = "Kayıtlı model sonuçlarının yan yana karşılaştırması"
    COLUMNS = ("Model", "Doğruluk", "Makro F1", "Ağırlıklı F1", "Makro PR-AUC", "Havuzlanmış makro F1",
               "Segment", "Kayıt", "Tür", "Değerlendirme", "Eğitim zamanı")

    def __init__(self, parent=None):
        super().__init__("📊 Model Karşılaştırma", parent)
        self.resize(1180, 760)
        outer = QVBoxLayout(self)
        self._add_hero(outer)
        self.note_label = QLabel()
        self.note_label.setWordWrap(True)
        self.note_label.setTextFormat(Qt.RichText)
        self.note_label.setStyleSheet("QLabel { background:#eef2fb; padding:10px 14px; border-radius:6px; font-size:13px; }")
        outer.addWidget(self.note_label)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        outer.addWidget(self.table)
        sig_title = QLabel("<b>Eşleştirilmiş anlamlılık testi</b> (makro F1; Nadeau–Bengio düzeltmeli, aile içi Holm)")
        outer.addWidget(sig_title)
        self.sig_table = QTableWidget(0, len(model_significance.TABLE_COLUMNS))
        self.sig_table.setHorizontalHeaderLabels(list(model_significance.TABLE_COLUMNS))
        for j, tip in enumerate(model_significance.TABLE_COLUMN_HELP):
            self.sig_table.horizontalHeaderItem(j).setToolTip(tip)
        sig_title.setToolTip(model_significance.FAMILY_HELP)
        self.sig_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sig_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sig_table.horizontalHeader().setStretchLastSection(True)
        self.sig_table.verticalHeader().setVisible(False)
        outer.addWidget(self.sig_table)
        outer.addStretch(1)
        self._sig_result = {"satirlar": [], "uyarilar": []}
        self.warn_label = QLabel()
        self.warn_label.setWordWrap(True)
        self.warn_label.setTextFormat(Qt.RichText)
        self.warn_label.setStyleSheet("QLabel { background:#fff6e5; padding:10px 14px; border-radius:6px; font-size:13px; }")
        outer.addWidget(self.warn_label)
        row = QHBoxLayout()
        btn_refresh = QPushButton("Yenile")
        btn_refresh.clicked.connect(self.refresh)
        btn_csv = QPushButton("Anlamlılık tablosunu CSV kaydet")
        btn_csv.clicked.connect(self._save_sig_csv)
        btn_close = QPushButton("Kapat")
        btn_close.clicked.connect(self.close)
        row.addWidget(btn_refresh)
        row.addWidget(btn_csv)
        row.addStretch(1)
        row.addWidget(btn_close)
        outer.addLayout(row)
        self.refresh()

    def _save_sig_csv(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Anlamlılık tablosunu kaydet", "model_anlamlilik.csv",
                                              "CSV (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                f.write(model_significance.to_csv(self._sig_result))

    @staticmethod
    def _pm(v, pct=False):
        """(ortalama, s.s.) -> metin; yoksa '—'."""
        if not v:
            return "—"
        m, sd = float(v[0]), float(v[1])
        return f"%{m * 100:.1f} ± {sd * 100:.1f}" if pct else f"{m:.3f} ± {sd:.3f}"

    @classmethod
    def build_rows(cls, stored_list):
        """[(ModelSpec, StoredResult)] -> (satır listesi, uyarı listesi). Arayüzden bağımsız, test edilebilir."""
        rows, warns, hashes = [], [], {}
        for spec, st in stored_list:
            tr = st.result.training_result
            cv = tr.cv_summary if isinstance(tr.cv_summary, dict) and "dogruluk" in tr.cv_summary else None
            if cv:
                havuz = cv.get("havuzlanmis") or {}
                vals = (cls._pm(cv.get("dogruluk"), pct=True), cls._pm(cv.get("makro_f1")),
                        cls._pm(cv.get("agirlikli_f1")), cls._pm(cv.get("makro_pr_auc")),
                        f"{havuz['makro_f1']:.3f}" if "makro_f1" in havuz else "—")
                degerlendirme = f"çapraz doğrulama ({cv.get('n_kat')}×{cv.get('n_tekrar')})"
            else:
                vals = (f"%{tr.test_accuracy * 100:.1f}", f"{tr.test_macro_f1:.3f}", "—", "—", "—")
                degerlendirme = "tek bölme (kararsız)"
            n_seg = tr.n_train + tr.n_val + tr.n_test
            n_rec = tr.n_train_recordings + tr.n_val_recordings + tr.n_test_recordings
            rows.append((spec.ad,) + vals + (str(n_seg), str(n_rec), str(len(tr.class_names)), degerlendirme,
                                              last_result_store.format_time(st.saved_at)))
            fp = st.fingerprint or {}
            if fp.get("hash"):
                hashes[spec.ad] = fp["hash"]
        if len(set(hashes.values())) > 1:
            warns.append("Modeller <b>farklı veri durumunda</b> eğitilmiş (veri izi eşleşmiyor): sonuçlar doğrudan "
                         "karşılaştırılamaz. Hepsini aynı veriyle yeniden eğitin.")
        evals = {r[9] for r in rows}
        if len(evals) > 1:
            warns.append("Modellerin değerlendirme yöntemi farklı (çapraz doğrulama / tek bölme): "
                         "yalnızca aynı yöntemle elde edilen değerler karşılaştırılabilir.")
        names = {"min_samples": "en az segment", "min_recordings": "en az kayıt", "test_size": "test oranı",
                 "val_size": "doğrulama oranı", "max_epochs": "en çok epoch", "bird_only": "yalnızca kuş türleri",
                 "class_weights": "sınıf ağırlığı", "cv": "çapraz doğrulama"}
        diff = [ad for key, ad in names.items()
                if len({repr((st.settings or {}).get(key)) for _spec, st in stored_list}) > 1]
        if diff:
            warns.append("Modellerin eğitim ayarları farklı (" + ", ".join(diff) + "): karşılaştırma için hepsini "
                         "aynı ayarlarla yeniden eğitin.")
        return rows, warns

    @staticmethod
    def _fit_height(tbl):
        """Tabloyu satır sayısına göre boyutlandırır (en az 2 satır yüksekliği, en çok 10)."""
        n = max(2, min(tbl.rowCount(), 10))
        h = tbl.horizontalHeader().height() + sum(tbl.rowHeight(i) for i in range(min(tbl.rowCount(), 10)))
        h += max(0, n - tbl.rowCount()) * 30 + 6
        tbl.setFixedHeight(h)

    def refresh(self):
        try:
            stored = model_registry.stored_results(csv_project_root())
        except Exception:  # noqa: BLE001
            stored = []
        rows, warns = self.build_rows(stored)
        self._sig_result = model_significance.compare_stored(stored)
        warns = warns + self._sig_result["uyarilar"]
        self.note_label.setText(
            "<b>Nasıl okunur:</b> ± değeri çapraz doğrulama testleri arasındaki standart sapmadır (güven aralığı "
            "değil). Üst tablo yalnızca özet değerleri gösterir; modeller arası farkın anlamlılığı alttaki "
            "<b>eşleştirilmiş testle</b> (aynı katlarda makro F1 farkı) okunur. Test, bu veri kümesinin yeniden "
            "örneklenen bölmelerinde algoritmaların farkını sınar; yeni nokta/alan için genelleme iddiası değildir. "
            "Segmentler ve etiketler BirdNET tespit ve önerisine dayandığından karşılaştırma BirdNET lehine yanlı "
            "olabilir. Perch 2.0 3 sn segmentlerde sıfır dolgu ile çalışır.")
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                self.table.setItem(i, j, QTableWidgetItem(str(v)))
        self.table.resizeColumnsToContents()
        self._fit_height(self.table)
        sig = self._sig_result["satirlar"]
        self.sig_table.setRowCount(len(sig))
        for i, r in enumerate(sig):
            for j, v in enumerate(model_significance.table_cells(r)):
                self.sig_table.setItem(i, j, QTableWidgetItem(str(v)))
        self.sig_table.resizeColumnsToContents()
        self._fit_height(self.sig_table)
        if not rows:
            warns.append("Kayıtlı model sonucu yok. Önce ilgili modelin eğitim penceresinde bir eğitim çalıştırın.")
        self.warn_label.setVisible(bool(warns))
        self.warn_label.setText("<br>".join("⚠ " + w for w in warns))


class ClassifierTrainingThread(QThread):
    """
    `segments/classification/` altında türlere göre düzenlenmiş veriden BirdNET 2.4
    gömmelerini (embedding) çıkaran ve bunlar üzerinde küçük bir MLP sınıflandırıcı
    eğiten arka plan iş parçacığı. SpeciesDetectionThread gibi ayrı bir iş
    parçacığında çalışır; böylece ana pencere (arayüz) donmaz.
    """

    progress = pyqtSignal(str, int, int)  # aşama, mevcut, toplam
    finished = pyqtSignal(object)  # ml_pipeline.PipelineResult
    error = pyqtSignal(str)

    def __init__(self, min_samples_per_class, min_recordings_per_class, test_size, val_size,
                 max_epochs, restrict_to_known_bird_species, parent=None, use_class_weights=False, cv_folds=0,
                 model_id=model_registry.DEFAULT_MODEL_ID, learning_curve=False):
        super().__init__(parent)
        self.model_id = model_id
        self.learning_curve = learning_curve
        self.use_class_weights = use_class_weights
        self.cv_folds = cv_folds
        self.min_samples_per_class = min_samples_per_class
        self.min_recordings_per_class = min_recordings_per_class
        self.test_size = test_size
        self.val_size = val_size
        self.max_epochs = max_epochs
        self.restrict_to_known_bird_species = restrict_to_known_bird_species

    def run(self):
        try:
            def on_progress(stage, current, total):
                self.progress.emit(stage, current, total)

            spec = model_registry.get_model(self.model_id)
            root = csv_project_root()
            result = ml_pipeline.run_full_training_pipeline(
                backend=spec.backend,
                save_dir=model_registry.model_save_dir(self.model_id, root),
                embedding_cache_path=model_registry.embedding_cache_path(self.model_id, root),
                embedding_read_caches=model_registry.embedding_read_caches(self.model_id, root),
                component_caches=model_registry.component_cache_paths(self.model_id, root),
                trainer=spec.egitici,
                min_samples_per_class=self.min_samples_per_class,
                min_recordings_per_class=self.min_recordings_per_class,
                test_size=self.test_size,
                val_size=self.val_size,
                max_epochs=self.max_epochs,
                restrict_to_known_bird_species=self.restrict_to_known_bird_species,
                use_class_weights=self.use_class_weights,
                cv_folds=self.cv_folds,
                learning_curve=self.learning_curve,
                progress_callback=on_progress,
            )
            self.finished.emit(result)
        except ml_pipeline.InsufficientDataError as exc:
            self.error.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Eğitim sırasında beklenmeyen hata: {exc}\n{traceback.format_exc()}")


class ClassifierTrainingDialog(_BaseSummaryDialog):
    """
    `segments/classification/` içindeki tür bazlı etiketli veriyi kullanarak,
    BirdNET 2.4'ün çıkardığı akustik gömmeler üzerinde özel bir
    tür sınıflandırıcısı eğiten pencere.
    """

    HERO_SUBTITLE = "BirdNET 2.4 gömmeleri üzerinde sınıflandırıcı eğitimi"

    def __init__(self, parent=None, model_id=model_registry.DEFAULT_MODEL_ID):
        self.model_id = model_id
        self.model_spec = model_registry.get_model(model_id)
        title = "🧠 Tür Sınıflandırma Eğitimi"
        if model_id != model_registry.DEFAULT_MODEL_ID:
            title += f" — {self.model_spec.ad}"
        super().__init__(title, parent)
        self.HERO_SUBTITLE = self.model_spec.altyazi or f"{self.model_spec.ad} gömmeleri üzerinde sınıflandırıcı eğitimi"
        self.result_dir = model_registry.result_dir(model_id, csv_project_root())
        self.training_thread: Optional[ClassifierTrainingThread] = None
        self.last_result = None
        self._build_ui()
        self._restore_last_result()

    def _build_ui(self):
        from PyQt5.QtWidgets import QProgressBar, QTabWidget, QPlainTextEdit
        outer_layout = QVBoxLayout(self)
        self._add_hero(outer_layout)

        # Dört sekme: her biri pencerenin tüm yüksekliğini kullanır (eskiden her şey tek sütundaydı ve
        # grafiklere çok az yer kalıyordu).
        self.tabs = QTabWidget()
        # Uygulama stil sayfası sekme yazısını kalın/büyük yapıyor; Qt sekme genişliğini sekme çubuğunun KENDİ
        # yazı tipinden hesapladığı için bu ayrı pencerede metin kırpılıyordu. Aynı yazı tipini çubuğa veriyoruz,
        # emoji kullanmıyoruz ve kırpmayı kapatıyoruz.
        from PyQt5.QtGui import QFont
        tab_font = QFont(self.font())
        tab_font.setPixelSize(16)
        tab_font.setBold(True)
        bar = self.tabs.tabBar()
        bar.setFont(tab_font)
        bar.setElideMode(Qt.ElideNone)
        bar.setUsesScrollButtons(False)
        bar.setExpanding(False)
        self.tabs.setStyleSheet("QTabBar::tab { min-width: 120px; padding: 8px 24px; }")
        outer_layout.addWidget(self.tabs, stretch=1)

        # ---- Sekme 1: Ayarlar
        settings = QWidget()
        sv = QVBoxLayout(settings)
        sv.setContentsMargins(8, 10, 8, 8)
        info_label = QLabel(
            f"{self.model_spec.girdi_notu} Üzerine küçük bir "
            "sinir ağı (MLP) sınıflandırıcı eğitilir. Eğitim / doğrulama / test ayrımı <b>kayıt bazlıdır</b>: aynı "
            "kayıttan gelen segmentler asla farklı kümelere düşmez."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            "QLabel { background:#eef6ee; padding:10px 14px; border-radius:6px; font-size: 14px; }"
        )
        sv.addWidget(info_label)

        param_form = QFormLayout()

        self.spn_min_samples = QSpinBox()
        self.spn_min_samples.setRange(2, 1000)
        self.spn_min_samples.setValue(dataset_summary.MIN_SEGMENT_FOR_TRAINING)
        param_form.addRow(QLabel("Min. örnek (segment) sayısı/tür:"), self.spn_min_samples)

        self.spn_min_recordings = QSpinBox()
        self.spn_min_recordings.setRange(2, 100)
        self.spn_min_recordings.setValue(2)
        self.spn_min_recordings.setToolTip(
            "Bir türün kaynaklanması gereken en az FARKLI orijinal kayıt sayısı — "
            "en az 2 (biri eğitime, biri teste ayrılabilsin diye). Bu, segment sayısı "
            "şartından AYRI bir şarttır."
        )
        param_form.addRow(QLabel("Min. FARKLI kayıt sayısı/tür:"), self.spn_min_recordings)

        self.spn_test_size = QDoubleSpinBox()
        self.spn_test_size.setRange(0.05, 0.5)
        self.spn_test_size.setSingleStep(0.05)
        self.spn_test_size.setValue(0.2)
        param_form.addRow(QLabel("Test oranı:"), self.spn_test_size)

        self.spn_val_size = QDoubleSpinBox()
        self.spn_val_size.setRange(0.05, 0.5)
        self.spn_val_size.setSingleStep(0.05)
        self.spn_val_size.setValue(0.15)
        param_form.addRow(QLabel("Doğrulama oranı:"), self.spn_val_size)

        self.spn_max_epochs = QSpinBox()
        self.spn_max_epochs.setRange(10, 1000)
        self.spn_max_epochs.setValue(100)
        param_form.addRow(QLabel("Maks. epoch (erken durdurma açık):"), self.spn_max_epochs)

        self.chk_bird_only = QCheckBox("Yalnızca bilinen kuş türleriyle sınırla (518 türlük veri tabanı)")
        self.chk_bird_only.setChecked(True)
        self.chk_bird_only.setToolTip(
            "Açıkken, projenin bird_species_data.json veri tabanında BULUNMAYAN türler "
            "(örn. yanlışlıkla etiketlenmiş bir kurbağa/böcek türü) eğitim kümesinden "
            "otomatik olarak çıkarılır."
        )
        param_form.addRow(QLabel(""), self.chk_bird_only)

        self.chk_class_weights = QCheckBox("Sınıf ağırlıklarını kullan (nadir türlere daha fazla ağırlık)")
        self.chk_class_weights.setChecked(True)
        self.chk_class_weights.setToolTip(
            "Açıkken kayıp fonksiyonu tür başına 'dengeli' ağırlık (N / (K·n_tür)) kullanır; "
            "az örnekli türler daha çok önemsenir. Sonuç bu veri setine özgüdür."
        )
        param_form.addRow(QLabel(""), self.chk_class_weights)

        self.chk_cv = QCheckBox("Çapraz doğrulama ile güvenilir başarı ölçümü (önerilir; 5 kat × 2 tekrar)")
        self.chk_cv.setChecked(True)
        self.chk_cv.setToolTip(
            "Tek bir test kümesi küçük veride kararsızdır (tek bir kayıt sonucu belirleyebilir). Açıkken, "
            "son model eğitilmeden önce kayıt bazlı tekrarlı çapraz doğrulama yapılır ve ana sonuç olarak "
            "gösterilir. Eğitim süresi uzar.")
        param_form.addRow(QLabel(""), self.chk_cv)

        self.chk_curve = QCheckBox("Öğrenme eğrisini de çiz (eğitim kayıtlarının %25/50/75/100'ü; süre uzar)")
        self.chk_curve.setChecked(True)
        self.chk_curve.setToolTip(
            "Çapraz doğrulama açıkken, aynı bölmelerle eğitim kayıtlarının %25, %50 ve %75'i de denenir "
            "(%100 sonucu çapraz doğrulamadan alınır). Test kümesi sabit kalır. Sonuç Grafikler sekmesinde "
            "görünür ve Sonuçları Temizle'ye kadar saklanır.")
        self.chk_cv.toggled.connect(self.chk_curve.setEnabled)
        param_form.addRow(QLabel(""), self.chk_curve)

        sv.addLayout(param_form)
        sv.addStretch()
        self.tabs.addTab(settings, "Ayarlar")

        # ---- Sekme 2: Sonuç (özet kartları + yorum + uyarılar)
        self._result_scroll = QScrollArea()
        self._result_scroll.setWidgetResizable(True)
        self._result_scroll.setFrameShape(QScrollArea.NoFrame)
        result_inner = QWidget()
        self._result_layout = QVBoxLayout(result_inner)
        self._result_layout.setContentsMargins(8, 12, 8, 8)
        self._result_layout.setSpacing(12)
        self._result_layout.setAlignment(Qt.AlignTop)
        self._result_scroll.setWidget(result_inner)
        self._set_result_placeholder()
        self.tabs.addTab(self._result_scroll, "Sonuç")

        # ---- Sekme 3: Grafikler (tüm yükseklik)
        charts_scroll = QScrollArea()
        charts_scroll.setWidgetResizable(True)
        charts_scroll.setFrameShape(QScrollArea.NoFrame)
        charts_inner = QWidget()
        self._scroll_layout = QVBoxLayout(charts_inner)
        self._scroll_layout.setContentsMargins(8, 12, 8, 8)
        self._scroll_layout.setSpacing(14)
        self._scroll_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        charts_scroll.setWidget(charts_inner)
        self.tabs.addTab(charts_scroll, "Grafikler")

        # ---- Sekme 4: Ayrıntılar (ham metin özeti; kopyalanabilir)
        self.details_view = QPlainTextEdit()
        self.details_view.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(10)
        self.details_view.setFont(mono)
        self.details_view.setPlaceholderText("Eğitim bitince ayrıntılı metin özeti burada görünür.")
        self.tabs.addTab(self.details_view, "Ayrıntılar")

        # ---- Alt şerit: her sekmede görünür (ilerleme + düğmeler)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        outer_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Hazır.")
        self.status_label.setWordWrap(True)
        outer_layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶ Eğitimi Başlat")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_fullscreen = QPushButton("🔲 Tam Ekran")
        self.btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        self.btn_clear = QPushButton("🗑 Sonuçları Temizle")
        self.btn_clear.setToolTip("Son eğitimin kayıtlı sonucunu ve rapor dosyalarını siler. "
                                  "Eğitilmiş model ve veri setiniz silinmez.")
        self.btn_clear.clicked.connect(self._on_clear_results)
        btn_close = QPushButton("Kapat")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_fullscreen)
        btn_row.addWidget(self._make_save_charts_button())
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        outer_layout.addLayout(btn_row)

    def _on_clear_results(self):
        if self.training_thread is not None and self.training_thread.isRunning():
            QMessageBox.information(self, "Eğitim sürüyor", "Eğitim bitmeden sonuçlar temizlenemez.")
            return
        ans = QMessageBox.question(
            self, "Sonuçları temizle",
            "Son eğitimin kayıtlı sonucu ve rapor dosyaları (son_siniflandirma_sonucu.json, son_egitim_sonucu.json, "
            "test_segment_tahminleri.csv, test_kayit_bazli.csv, test_tur_varligi.csv, ogrenme_egrisi*.csv) silinecek.\n\n"
            "Eğitilmiş model, gömme önbelleği ve veri setiniz silinmeyecek. Bu işlem geri alınamaz.\n\n"
            "Devam edilsin mi?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        try:
            removed = last_result_store.clear_saved_results(self.result_dir)
        except OSError as exc:
            QMessageBox.warning(self, "Temizlenemedi", f"Dosyalar silinemedi: {exc}")
            return
        self.last_result = None
        self._clear_layout(self._scroll_layout)
        self.details_view.clear()
        self._set_result_placeholder()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Kayıtlı sonuçlar temizlendi ({len(removed)} dosya silindi)." if removed
                                  else "Silinecek kayıtlı sonuç yoktu.")
        self.tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------ sonuç sekmesi yardımcıları
    @staticmethod
    def _metric_card(value: str, title: str, sub: str, accent: str) -> QFrame:
        f = QFrame()
        f.setObjectName("card")
        f.setStyleSheet(f"QFrame#card {{ background:#ffffff; border:1px solid #e1e6ee; "
                        f"border-top:4px solid {accent}; border-radius:12px; }}")
        v = QVBoxLayout(f)
        v.setContentsMargins(18, 12, 18, 12)
        v.setSpacing(0)
        a = QLabel(value)
        a.setStyleSheet(f"font-size:38px; font-weight:800; color:{accent};")
        b = QLabel(title)
        b.setStyleSheet("font-size:14px; font-weight:700; color:#1f2a37;")
        c = QLabel(sub)
        c.setWordWrap(True)
        c.setStyleSheet("font-size:12px; color:#6b7280;")
        v.addWidget(a)
        v.addWidget(b)
        v.addWidget(c)
        return f

    @staticmethod
    def _note_label(text: str, color: str = "#1f2a37", bg: str = "#f3f6fa") -> QLabel:
        lab = QLabel(text)
        lab.setTextFormat(Qt.RichText)
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lab.setStyleSheet(f"QLabel {{ background:{bg}; color:{color}; padding:10px 14px; "
                          f"border-radius:8px; font-size:14px; }}")
        return lab

    @staticmethod
    def _clear_layout(layout):
        for i in reversed(range(layout.count())):
            item = layout.takeAt(i)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                ClassifierTrainingDialog._clear_layout(item.layout())

    def _set_result_placeholder(self):
        self._clear_layout(self._result_layout)
        self._result_layout.addWidget(self._note_label(
            "Henüz eğitim yapılmadı. <b>Ayarlar</b> sekmesinden ayarları kontrol edip "
            "<b>▶ Eğitimi Başlat</b>'a basın; sonuçlar burada özetlenir."))

    def _populate_results(self, result, saved_at=None, restored=False, stale_note=None):
        """Sonuç sekmesi: eğitim zamanı, üç özet kartı, tek cümlelik yorum, en zayıf türler ve (varsa) kısa uyarılar."""
        tr = result.training_result
        cv = tr.cv_summary
        cv_ok = bool(cv) and "hata" not in cv
        self._clear_layout(self._result_layout)

        if saved_at is not None:
            when = last_result_store.format_time(saved_at)
            if restored:
                self._result_layout.addWidget(self._note_label(
                    f"<b>Kayıtlı sonuç:</b> en son eğitim <b>{when}</b> tarihinde yapıldı. Pencere yeniden açıldığı için "
                    "bu sonuç diskten yüklendi; yeni bir ölçüm için <b>▶ Eğitimi Başlat</b>'a basın.",
                    color="#1e3a5f", bg="#e8f1fb"))
            else:
                self._result_layout.addWidget(self._note_label(
                    f"<b>Eğitim zamanı:</b> {when}. Bu sonuç kaydedildi; pencereyi sonra açtığınızda yeniden "
                    "eğitmeden gösterilir.", color="#1e3a5f", bg="#e8f1fb"))
            if stale_note:
                self._result_layout.addWidget(self._note_label("⚠ " + stale_note, color="#92400e", bg="#fff4e0"))

        n_seg = tr.n_train + tr.n_val + tr.n_test
        n_rec = tr.n_train_recordings + tr.n_val_recordings + tr.n_test_recordings
        cards = QHBoxLayout()
        cards.setSpacing(14)
        if cv_ok:
            (a, asd), (f, fsd) = cv["dogruluk"], cv["makro_f1"]
            cards.addWidget(self._metric_card(
                f"%{a * 100:.1f}", "Doğruluk",
                f"± {asd * 100:.1f} · çapraz doğrulama ({cv['n_kat']}×{cv['n_tekrar']})", RENK["yesil"]), 1)
            cards.addWidget(self._metric_card(
                f"{f:.2f}", "Makro F1",
                f"± {fsd:.3f} · her tür eşit ağırlıkla", RENK["mavi"]), 1)
        else:
            cards.addWidget(self._metric_card(
                f"%{tr.test_accuracy * 100:.1f}", "Doğruluk",
                "tek test kümesi · kararsız olabilir", RENK["turuncu"]), 1)
            cards.addWidget(self._metric_card(
                f"{tr.test_macro_f1:.2f}", "Makro F1",
                "tek test kümesi · her tür eşit ağırlıkla", RENK["turuncu"]), 1)
        cards.addWidget(self._metric_card(
            str(len(tr.class_names)), "Sınıflandırılan tür", f"{n_seg} segment · {n_rec} kayıt", RENK["mor"]), 1)
        self._result_layout.addLayout(cards)

        if cv_ok:
            extra = QHBoxLayout()
            extra.setSpacing(14)
            if "agirlikli_f1" in cv:
                w, wsd = cv["agirlikli_f1"]
                extra.addWidget(self._metric_card(
                    f"{w:.2f}", "Ağırlıklı F1",
                    f"± {wsd:.3f} · her türün F1 değeri, o türün segment sayısıyla ağırlıklandırıldı; "
                    "çok segmentli türler sonucu daha çok etkiler", RENK["turuncu"]), 1)
            if "makro_pr_auc" in cv:
                p, psd = cv["makro_pr_auc"]
                extra.addWidget(self._metric_card(
                    f"{p:.2f}", "Makro PR-AUC",
                    f"± {psd:.3f} · modelin doğru türe ne kadar yüksek olasılık verdiği (1,00 en iyi); "
                    "her tür eşit ağırlıklı", RENK["mor"]), 1)
            if "havuzlanmis" in cv:
                hp = cv["havuzlanmis"]["makro_f1"]
                extra.addWidget(self._metric_card(
                    f"{hp:.2f}", "Havuzlanmış makro F1",
                    f"{cv['n_deger']} testin tüm tahminleri birleştirilip tek seferde hesaplandı; "
                    "Grafikler sekmesindeki tür F1 değerleri bu hesaptandır", RENK["mavi"]), 1)
            if extra.count():
                self._result_layout.addWidget(QLabel("<b>Ek ölçütler</b>"))
                self._result_layout.addLayout(extra)
            else:
                self._result_layout.addWidget(self._note_label(
                    "<b>Ek ölçütler</b> (Ağırlıklı F1, Makro PR-AUC, Havuzlanmış makro F1) bu kayıtlı sonuçta yok; "
                    "<b>▶ Eğitimi Başlat</b> ile yeniden eğitince hesaplanır."))

        if cv_ok:
            lo, hi = cv["dogruluk_aralik"]
            self._result_layout.addWidget(self._note_label(
                f"<b>Nasıl okunur:</b> Kayıtlar ayrı tutularak model {cv['n_deger']} kez test edildi; kartlardaki "
                f"değer bu testlerin ortalamasıdır. Tek tek testlerde doğruluk %{lo * 100:.0f}–%{hi * 100:.0f} "
                f"arasında değişti; yeni kayıtlarda beklenen başarı da bu aralıkta dalgalanır."))
        else:
            reason = f" ({cv['hata']})" if cv and "hata" in cv else ""
            self._result_layout.addWidget(self._note_label(
                f"<b>Dikkat:</b> Çapraz doğrulama çalışmadı{reason}. Sonuç yalnızca tek bir test kümesinden "
                f"geliyor; küçük veride bir kayıt sonucu belirleyebilir.", color="#92400e", bg="#fff4e0"))

        per_f1 = (cv or {}).get("tur_f1") if cv_ok else None
        if not per_f1:
            per_f1 = {c: i["f1-score"] for c, i in tr.per_class_report.items()
                      if c in tr.class_names and i.get("support", 1) > 0}
        if per_f1:
            weakest = sorted(per_f1.items(), key=lambda kv: kv[1])[:3]
            self._result_layout.addWidget(self._note_label(
                "<b>En zayıf türler (F1):</b> " + " · ".join(f"{k} ({v:.2f})" for k, v in weakest)
                + "<br><span style='color:#6b7280'>Tür bazlı tüm değerler <b>Grafikler</b> sekmesinde.</span>"))

        zu = tr.zenginlik_uyumu
        if zu and "esikler" in zu:
            head = ("<tr><th align='left'>Güven eşiği</th><th>Tespit oranı</th><th>Tür (uzman / model)</th>"
                    "<th>Jaccard</th><th>Kesinlik</th><th>Duyarlılık</th><th>Sahte tür</th><th>Kaçırılan</th></tr>")
            body = "".join(
                f"<tr><td>≥ {o['guven_esigi']:.1f}</td><td align='center'>{o['tespit_orani_ort']:.2f}</td>"
                f"<td align='center'>{o['tur_uzman_ort']:.0f} / {o['tur_model_ort']:.1f}</td>"
                f"<td align='center'>{o['var_yok_jaccard_ort']:.2f}</td><td align='center'>{o['var_yok_kesinlik_ort']:.2f}</td>"
                f"<td align='center'>{o['var_yok_duyarlilik_ort']:.2f}</td>"
                f"<td align='center'>{o['sahte_tur_nokta_cifti_ort']:.1f}</td>"
                f"<td align='center'>{o['kacirilan_tur_nokta_cifti_ort']:.1f}</td></tr>" for o in zu["esikler"])
            self._result_layout.addWidget(self._note_label(
                f"<b>Tür zenginliği uyumu</b> ({zu['n_nokta']} nokta): modelin çapraz doğrulama tahminlerinden çıkan tür listesi, "
                "uzmanın listesiyle karşılaştırıldı. Sahte tür = modelin noktada olmayan bir türü bulması; kaçırılan = uzmanın "
                "bulduğu türün model listesinde olmaması (nokta×tür çifti sayısı).<br>"
                f"<table cellpadding='4' style='margin-top:6px'>{head}{body}</table>"
                "<span style='color:#6b7280'>Yalnızca sınıflandırma hatasının etkisi ölçülür; uzmanın seçtiği segmentler girdi "
                "olduğundan tespit aşaması hariçtir. Segmentler BirdNET önerisiyle seçildiği için BirdNET lehine yanlılık olabilir. "
                "Tüm eşikler raporlanır, biri seçilmez.</span>"))
        elif zu and "hata" in zu:
            self._result_layout.addWidget(self._note_label("⚠ Tür zenginliği uyumu hesaplanamadı: " + zu["hata"],
                                                           color="#92400e", bg="#fff4e0"))

        warns = []
        if not cv_ok and tr.test_predictions:
            from ..ml.recording_level import dominance_warning
            w = dominance_warning(tr.test_predictions)
            if w:
                warns.append(w.replace("⚠ ", ""))
        if tr.excluded_species:
            warns.append(f"{len(tr.excluded_species)} tür yetersiz örnek nedeniyle eğitime alınmadı "
                         f"(liste: Ayrıntılar sekmesi).")
        if tr.n_val == 0:
            warns.append("Ayrı bir doğrulama kümesi için yeterli farklı kayıt yoktu; yalnızca erken durdurma "
                         "etkilenir, sonuçlar hâlâ kayıt-izoledir.")
        for w in warns:
            self._result_layout.addWidget(self._note_label("⚠ " + w, color="#92400e", bg="#fff4e0"))

        foot = QLabel(f"Model kaydedildi: {result.save_dir}")
        foot.setStyleSheet("color:#6b7280; font-size:12px; padding:2px 4px;")
        foot.setTextInteractionFlags(Qt.TextSelectableByMouse)
        foot.setWordWrap(True)
        self._result_layout.addWidget(foot)

    def _add_chart_with_caption(self, canvas, caption: str):
        lab = QLabel(caption)
        lab.setWordWrap(True)
        lab.setStyleSheet("color:#4b5563; font-size:13px;")
        self._scroll_layout.addWidget(lab)            # hizasız: tam genişlikte, metin kesilmeden sarılır
        self._scroll_layout.addWidget(canvas, alignment=Qt.AlignLeft)

    # ------------------------------------------------------------------ çalıştırma
    def _on_run(self):
        if self.training_thread is not None and self.training_thread.isRunning():
            return

        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Veri seti taranıyor...")

        self.training_thread = ClassifierTrainingThread(
            min_samples_per_class=self.spn_min_samples.value(),
            min_recordings_per_class=self.spn_min_recordings.value(),
            test_size=self.spn_test_size.value(),
            val_size=self.spn_val_size.value(),
            max_epochs=self.spn_max_epochs.value(),
            restrict_to_known_bird_species=self.chk_bird_only.isChecked(),
            use_class_weights=self.chk_class_weights.isChecked(),
            cv_folds=5 if self.chk_cv.isChecked() else 0,
            model_id=self.model_id,
            learning_curve=self.chk_cv.isChecked() and self.chk_curve.isChecked(),
        )
        self.training_thread.progress.connect(self._on_progress)
        self.training_thread.finished.connect(self._on_finished)
        self.training_thread.error.connect(self._on_error)
        self.training_thread.start()

    def _model_download_info(self):
        """(bu oturumda inen MB, MB/sn) — izlenecek önbellek klasörü yoksa None (BirdNET)."""
        import time
        dirs = model_registry.download_cache_dirs(self.model_id)
        if not dirs:
            return None
        now, size = time.monotonic(), model_registry.dir_size_mb(dirs)
        base = getattr(self, "_dl_base", None)
        if base is None:
            self._dl_base, self._dl_last = size, (now, size)
            return 0.0, 0.0
        last_t, last_s = self._dl_last
        speed = max(0.0, (size - last_s) / (now - last_t)) if now > last_t else 0.0
        self._dl_last = (now, size)
        return max(0.0, size - base), speed

    def _update_model_wait_text(self):
        info = self._model_download_info()
        if info is None:
            self.status_label.setText(
                f"{self.model_spec.ad} modeli yükleniyor / indiriliyor — ilk seferde dakikalar sürebilir, "
                "pencereyi kapatmayın")
            return
        mb, speed = info
        if mb < 0.5:
            detay = "henüz veri inmedi (model daha önce indirildiyse diskten yükleniyor)"
        else:
            detay = f"bu oturumda inen veri: {mb:,.0f} MB (≈ {speed:,.1f} MB/sn); yüzde çubuğu konsol penceresinde de görünür"
        self.status_label.setText(
            f"{self.model_spec.ad} modeli yükleniyor / indiriliyor — {detay}. Toplam boyut bilinmediği için yüzde "
            "gösterilemiyor; sayı artıyorsa indirme sürüyor. Pencereyi kapatmayın.")

    def _set_model_wait(self, active: bool):
        """Model yüklenirken çubuğu belirsiz (hareketli) yapar ve önbellek boyutunu 2 sn'de bir günceller."""
        from PyQt5.QtCore import QTimer
        timer = getattr(self, "_model_timer", None)
        if active:
            self.progress_bar.setRange(0, 0)
            if timer is None:
                timer = QTimer(self)
                timer.timeout.connect(self._update_model_wait_text)
                self._model_timer = timer
            self._update_model_wait_text()
            timer.start(2000)
        else:
            if timer is not None:
                timer.stop()
            self._dl_base = None
            self.progress_bar.setRange(0, 100)

    def _step_info(self, stage: str):
        """(adım numarası, toplam adım) — çapraz doğrulama kapalıysa o adım sayılmaz."""
        cv = bool(getattr(self, "chk_cv", None) and self.chk_cv.isChecked())
        lc = cv and bool(getattr(self, "chk_curve", None) and self.chk_curve.isChecked())
        order = ["samples", "embedding"] + (["cv"] if cv else []) + (["lc"] if lc else []) + ["training", "done"]
        key = "embedding" if stage == "model" else stage
        return (order.index(key) + 1 if key in order else 0), len(order)

    def _set_busy(self, busy: bool):
        self.progress_bar.setRange(0, 0 if busy else 100)

    def _on_progress(self, stage: str, current: int, total: int):
        if stage == "model" and current < total:
            self._set_model_wait(True)
            return
        self._set_model_wait(False)
        n, n_total = self._step_info(stage)
        prefix = f"Adım {n}/{n_total} · " if n else ""
        pct = int((current / total) * 100) if total else 0
        busy = False
        if stage == "model":
            busy = True
            loaded = "Gömme" if self.model_spec.backend == "combined" else self.model_spec.ad.split()[0]
            text = (f"{prefix}{loaded} modeli yüklendi. Ses parçaları işleniyor, sayaç ilk grup bitince görünür. "
                    "İşlem sürüyor, pencereyi kapatmayın.")
        elif stage == "samples":
            busy = current < total
            text = f"{prefix}Etiketli segmentler taranıyor ve eğitime uygun türler seçiliyor."
        elif stage == "embedding":
            what = f"{self.model_spec.ad} ile sayısal vektörlere"
            text = (f"{prefix}Segmentler {what} dönüştürülüyor: {current}/{total}. "
                    "Çoğunlukla en uzun adım budur.")
        elif stage == "cv":
            if current == 0:
                busy = True
                text = (f"{prefix}Çapraz doğrulama başlıyor. Model her kat için yeniden eğitilip kayıt bazlı "
                        "test ediliyor, ilk kat birkaç dakika sürebilir.")
            else:
                text = (f"{prefix}Çapraz doğrulama (kayıt bazlı): {current}/{total} eğitim tamamlandı. "
                        "Başarının güvenilirliği ölçülüyor.")
        elif stage == "lc":
            if current == 0:
                busy = True
                text = f"{prefix}Öğrenme eğrisi başlıyor (eğitim kayıtlarının %25/50/75'i ile ek eğitimler)."
            else:
                text = f"{prefix}Öğrenme eğrisi: {current}/{total} (çapraz doğrulamadan gelen %100 sonuçları dahil)."
        elif stage == "training":
            head = "MLP" if self.model_spec.egitici == "mlp" else self.model_spec.ad
            if current == 0:
                busy = True
                text = f"{prefix}Son sınıflandırıcı ({head}) eğitimi başlıyor."
            else:
                text = (f"{prefix}Son sınıflandırıcı ({head}) eğitiliyor: epoch {current}/{total}. "
                        "Doğrulama kaybı düzelmezse eğitim erken durabilir.")
        elif stage == "done":
            text = f"{prefix}Sonuçlar kaydediliyor ve grafikler hazırlanıyor."
        else:
            text = f"{stage}: {current}/{total}"
        self._set_busy(busy)
        self.status_label.setText(text)
        if not busy:
            self.progress_bar.setValue(pct)

    def _on_error(self, message: str):
        self._set_model_wait(False)
        self.btn_run.setEnabled(True)
        self.status_label.setText("Bir hata oluştu.")
        QMessageBox.critical(self, "Tür Sınıflandırma Eğitimi", message)

    def _current_settings(self) -> dict:
        return {"min_samples": self.spn_min_samples.value(), "min_recordings": self.spn_min_recordings.value(),
                "test_size": self.spn_test_size.value(), "val_size": self.spn_val_size.value(),
                "max_epochs": self.spn_max_epochs.value(), "bird_only": self.chk_bird_only.isChecked(),
                "class_weights": self.chk_class_weights.isChecked(), "cv": self.chk_cv.isChecked(),
                "learning_curve": self.chk_curve.isChecked()}

    def _apply_settings(self, st: dict):
        """Kayıtlı sonucun üretildiği ayarları forma geri yazar (yeniden eğitim aynı ayarla başlasın)."""
        try:
            self.spn_min_samples.setValue(int(st.get("min_samples", self.spn_min_samples.value())))
            self.spn_min_recordings.setValue(int(st.get("min_recordings", self.spn_min_recordings.value())))
            self.spn_test_size.setValue(float(st.get("test_size", self.spn_test_size.value())))
            self.spn_val_size.setValue(float(st.get("val_size", self.spn_val_size.value())))
            self.spn_max_epochs.setValue(int(st.get("max_epochs", self.spn_max_epochs.value())))
            self.chk_bird_only.setChecked(bool(st.get("bird_only", self.chk_bird_only.isChecked())))
            self.chk_class_weights.setChecked(bool(st.get("class_weights", self.chk_class_weights.isChecked())))
            self.chk_cv.setChecked(bool(st.get("cv", self.chk_cv.isChecked())))
            self.chk_curve.setChecked(bool(st.get("learning_curve", self.chk_curve.isChecked())))
        except (TypeError, ValueError):
            pass

    @staticmethod
    def _dataset_csv_path() -> str:
        return os.path.join(csv_project_root(), "bird_sound_dataset.csv")

    def _restore_last_result(self):
        """Pencere açılınca en son kaydedilen sonucu (varsa) tarih-saatiyle birlikte gösterir."""
        try:
            stored = last_result_store.load_last_result(self.result_dir)
        except Exception:  # noqa: BLE001  (kayıt bozuksa pencere yine de boş açılsın)
            stored = None
        if stored is None:
            return
        stale = None
        try:
            now_fp = classification_diagnosis.data_fingerprint(self._dataset_csv_path())
            old_fp = stored.fingerprint
            if old_fp and now_fp and old_fp.get("hash") != now_fp.get("hash"):
                stale = (f"Veri seti bu eğitimden sonra değişti (eğitimde {old_fp.get('n_segment')} segment, şimdi "
                         f"{now_fp.get('n_segment')}). Gösterilen sonuç güncel veriyi yansıtmıyor olabilir; "
                         "güncel ölçüm için yeniden eğitin.")
        except Exception:  # noqa: BLE001
            pass
        self._apply_settings(stored.settings)
        self.last_result = stored.result
        self._show_result(stored.result, stored.details, stored.saved_at, restored=True, stale_note=stale)
        self.status_label.setText(f"Kayıtlı sonuç gösteriliyor (son eğitim: {last_result_store.format_time(stored.saved_at)}).")

    def _on_finished(self, result):
        self._set_model_wait(False)
        self.btn_run.setEnabled(True)
        self.last_result = result
        self.progress_bar.setValue(100)
        self.status_label.setText("Eğitim tamamlandı.")

        details = result.training_result.format_summary()
        details += f"\n\nModel şuraya kaydedildi: {result.save_dir}"
        try:   # analiz penceresi tek bölme sonucunu da gösterebilsin
            classification_diagnosis.save_single_split(self.result_dir, result.training_result)
        except Exception:
            pass
        try:   # kayıt bazlı hata ve tür varlığı raporları (CSV)
            from ..ml import recording_level
            if result.training_result.test_predictions:
                recording_level.write_reports(self.result_dir, result.training_result.test_predictions)
                details += ("\nKayıt bazlı rapor: test_kayit_bazli.csv, test_tur_varligi.csv, "
                            "test_segment_tahminleri.csv")
        except Exception:
            pass

        lc = result.training_result.learning_curve
        if lc and "ozet" in lc:
            try:
                from ..ml import learning_curve as _lc
                _lc.write_csvs(self.result_dir, lc)
                details += "\nÖğrenme eğrisi: ogrenme_egrisi.csv, ogrenme_egrisi_ozet.csv"
            except Exception:  # noqa: BLE001
                pass

        zu = result.training_result.zenginlik_uyumu
        if zu and "esikler" in zu:
            try:
                from ..ml import richness_validation as _rv
                _rv.write_csvs(self.result_dir, zu)
                details += "\nTür zenginliği uyumu: zenginlik_ozet.csv, zenginlik_nokta.csv"
            except Exception:  # noqa: BLE001
                pass

        saved_at = datetime.now()
        try:   # son sonucu sakla: pencere sonra açılınca yeniden eğitmeden gösterilir
            fp = classification_diagnosis.data_fingerprint(self._dataset_csv_path())
            last_result_store.save_last_result(self.result_dir, result, details, self._current_settings(),
                                               fp, when=saved_at)
        except Exception as exc:  # noqa: BLE001  (kayıt başarısızsa eğitim sonucu yine de gösterilir)
            self.status_label.setText(f"Eğitim tamamlandı (sonuç diske kaydedilemedi: {exc}).")
        self._show_result(result, details, saved_at, restored=False, stale_note=None)

    def _show_result(self, result, details, saved_at, restored, stale_note):
        """Taze ya da diskten yüklenmiş sonucu dört sekmeye yazar."""
        self.details_view.setPlainText(details)
        self._populate_results(result, saved_at=saved_at, restored=restored, stale_note=stale_note)

        self._clear_layout(self._scroll_layout)
        cv_f1 = (result.training_result.cv_summary or {}).get("tur_f1")
        per_class_f1 = cv_f1 or {
            cls: info["f1-score"]
            for cls, info in result.training_result.per_class_report.items()
            if cls in result.training_result.class_names
        }
        if per_class_f1:
            counts = pd.Series(per_class_f1).sort_values(ascending=False)
            counts.index = [species_lookup.chart_label(str(c), with_latin=True) for c in counts.index]
            canvas = self._make_bar_chart_canvas(
                counts, "Tür Başına F1 (çapraz doğrulama)" if cv_f1 else "Tür Başına Test F1 Skoru (tek bölme)",
                "F1 skoru", highlight_thresholds=False, bar_color=RENK["yesil"],
            )
            self._add_chart_with_caption(
                canvas, "Her türün F1 skoru (1.0 = kusursuz). Kısa çubuk, modelin o türde zorlandığını gösterir.")

        lc = result.training_result.learning_curve
        if lc and "ozet" in lc:
            from ..ml import learning_curve as _lc
            self._add_chart_with_caption(
                self._wrap_figure_as_canvas(_lc.make_figure(lc["ozet"], self.model_spec.ad)),
                "Eğitim kayıtlarının oranına göre başarı (kayıt bazlı, aynı bölmeler; test kümesi sabit). Çubuklar katlar arası "
                "standart sapmadır, güven aralığı değildir. Eğri %100'e doğru hâlâ yükseliyorsa daha fazla etiket kazanç "
                "sağlar; düzleşmişse kazanç azalmıştır.")
        elif lc and "hata" in lc:
            self._add_chart_with_caption(QLabel(""), f"Öğrenme eğrisi çalıştırılamadı: {lc['hata']}")

        zu = result.training_result.zenginlik_uyumu
        if zu and zu.get("nokta"):
            from ..ml import richness_validation as _rv
            scatter_canvas = self._wrap_figure_as_canvas(_rv.make_scatter_figure(zu["nokta"], self.model_spec.ad))
            self._attach_scatter_hover(scatter_canvas, zu["nokta"])
            self._add_chart_with_caption(
                scatter_canvas,
                "Aynı (uzman, model) değerine düşen noktalar tek işarette toplanır; içindeki sayı nokta adedidir. İşaretin üzerine gelince noktanın ayrıntıları (habitat, konum, sahte ve kaçırılan türler) görünür. Köşegen = iki listede de "
                "aynı sayıda tür (aynı türler olması gerekmez; tür bazlı uyum için Jaccard, kesinlik ve duyarlılığa bakın). "
                "Köşegenin üstü modelin sahte tür eklediğini, altı tür kaçırdığını gösterir (güven eşiği yok, ilk tekrar).")
            self._add_point_table(zu["nokta"])

        cv_cm = (result.training_result.cv_summary or {}).get("karisiklik")
        cv_names = (result.training_result.cv_summary or {}).get("siniflar")
        if cv_cm and cv_names:
            cv_canvas = self._make_confusion_matrix_canvas(
                np.asarray(cv_cm), [species_lookup.chart_label(str(c)) for c in cv_names],
                title="Karışıklık Matrisi (çapraz doğrulama, birleştirilmiş)")
            self._add_chart_with_caption(
                cv_canvas,
                "Satır = gerçek tür, sütun = tahmin edilen tür. Tüm kat ve tekrarların test tahminleri toplanmıştır; "
                "çubuk grafikteki F1 değerleri bu matristen hesaplanır.")

        cm_canvas = self._make_confusion_matrix_canvas(
            result.training_result.confusion_matrix,
            [species_lookup.chart_label(str(c)) for c in result.training_result.class_names]
        )
        self._add_chart_with_caption(
            cm_canvas,
            "Satır = gerçek tür, sütun = tahmin edilen tür; köşegen dışındaki sayılar karışıklıklardır. "
            "Bu matris tek test kümesinden gelir; bir kayıt baskınsa yanıltıcı olabilir.")

        self.tabs.setCurrentIndex(1)   # Sonuç sekmesi

    def _attach_scatter_hover(self, canvas, nokta_rows):
        """İşaretin üzerine gelince, haritadaki gibi, o noktanın ayrıntılarını gösterir."""
        from PyQt5.QtWidgets import QToolTip
        from ..ml import richness_validation as _rv
        groups = _rv.group_points(nokta_rows)
        fig = canvas.figure
        state = {"key": None}

        def on_move(event):
            if event.x is None or event.y is None or event.inaxes is None:
                state["key"] = None
                QToolTip.hideText()
                canvas.setCursor(Qt.ArrowCursor)
                return
            key = _rv.find_marker_at(fig, event.x, event.y)
            if key is None or key not in groups:
                state["key"] = None
                QToolTip.hideText()
                canvas.setCursor(Qt.ArrowCursor)
                return
            canvas.setCursor(Qt.PointingHandCursor)
            if state["key"] != key:
                state["key"] = key
                QToolTip.showText(QCursor.pos(), _rv.hover_html(groups[key], species_lookup.chart_label), canvas)

        canvas.mpl_connect("motion_notify_event", on_move)
        canvas.mpl_connect("figure_leave_event", lambda e: QToolTip.hideText())
        canvas._hover_handler = on_move

    def _add_point_table(self, rows):
        """Nokta bazlı ayrıntı: hangi nokta, habitat tipi, konum ve modelin eklediği / kaçırdığı türler."""
        cols = [("nokta", "Nokta (Habitat_ID)"), ("habitat_tipi", "Habitat tipi"), ("konum", "Konum"),
                ("n_kayit", "Kayıt"), ("n_segment", "Segment"), ("tur_uzman", "Uzman tür"), ("tur_model", "Model tür"),
                ("sahte_turler", "Sahte türler (modelin eklediği)"), ("kacirilan_turler", "Kaçırılan türler (uzmanın bulduğu)")]
        numeric = {"n_kayit", "n_segment", "tur_uzman", "tur_model"}

        def names(text):
            return "; ".join(species_lookup.chart_label(t) for t in str(text).split("; ") if t)

        tbl = QTableWidget(len(rows), len(cols))
        tbl.setHorizontalHeaderLabels([c[1] for c in cols])
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setAlternatingRowColors(True)
        tbl.setWordWrap(True)
        for i, r in enumerate(rows):
            for j, (key, _) in enumerate(cols):
                val = r.get(key, "")
                item = QTableWidgetItem()
                if key in numeric:
                    item.setData(Qt.DisplayRole, int(val or 0))
                    item.setTextAlignment(Qt.AlignCenter)
                else:
                    item.setText(names(val) if key in ("sahte_turler", "kacirilan_turler") else str(val))
                if key == "sahte_turler" and val:
                    item.setBackground(QColor("#fff4e0"))
                if key == "kacirilan_turler" and val:
                    item.setBackground(QColor("#fde8e8"))
                tbl.setItem(i, j, item)
        tbl.resizeColumnsToContents()
        tbl.setColumnWidth(7, max(tbl.columnWidth(7), 260))
        tbl.setColumnWidth(8, max(tbl.columnWidth(8), 260))
        tbl.resizeRowsToContents()
        tbl.setSortingEnabled(True)
        tbl.setMinimumHeight(min(60 + 34 * len(rows), 460))
        tbl.setMinimumWidth(980)
        cap = QLabel("Nokta ayrıntıları (ilk tekrar, güven eşiği yok). Sütun başlığına tıklayınca sıralanır. Sahte tür: modelin noktada "
                     "bulduğu ama uzmanın bulmadığı tür; kaçırılan: uzmanın bulduğu ama modelin bulmadığı tür. Tablo "
                     "zenginlik_nokta.csv olarak da kaydedilir.")
        cap.setWordWrap(True)
        cap.setStyleSheet("color:#4b5563; font-size:13px;")
        self._scroll_layout.addWidget(cap)
        self._scroll_layout.addWidget(tbl)

    def _make_confusion_matrix_canvas(self, cm: np.ndarray, class_names: List[str],
                                      title: str = "Karışıklık Matrisi (tek test kümesi)"):
        from matplotlib.figure import Figure
        from ..visualization.plot_style import style_context
        n = len(class_names)
        with style_context():
            side = max(5, 0.7 * n)
            fig = Figure(figsize=(side, side), dpi=100)
            ax = fig.add_subplot(111)
            im = ax.imshow(cm, cmap="Blues")
            ax.grid(False)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(class_names, rotation=90, fontsize=9)
            ax.set_yticklabels(class_names, fontsize=9)
            ax.set_xlabel("Tahmin Edilen")
            ax.set_ylabel("Gerçek")
            ax.set_title(title)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=9, color="white" if cm[i, j] > cm.max() / 2 else "black")
            fig.colorbar(im, ax=ax, shrink=0.7)
            fig.tight_layout()
        return self._wrap_figure_as_canvas(fig)


def logo_pixmap(path: str, logical_px: int, dpr: float = 1.0):
    """
    Logoyu EKRANIN GERÇEK piksel boyutuna (mantıksal boyut x ekran ölçeği) küçültür. Büyük bir görüntüyü tek adımda
    küçültmek ya da önce 2x yapıp çizim sırasında tekrar küçültmek kaba/pürüzlü sonuç verir; burada kaynak,
    yarıya indirme adımlarıyla ve en son tam hedef boyuta kaliteli (smooth) ölçeklenir, çizimde yeniden ölçeklenmez.
    """
    from PyQt5.QtGui import QImage, QPixmap
    img = QImage(path)
    if img.isNull():
        return None
    target = max(8, int(round(logical_px * max(1.0, dpr))))
    while img.width() > target * 2:
        img = img.scaled(img.width() // 2, img.height() // 2, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    img = img.scaled(target, target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    pm = QPixmap.fromImage(img)
    pm.setDevicePixelRatio(max(1.0, dpr))
    return pm


class BirdSoundApp(QMainWindow):
    """Kuş sesi analizi ve etiketleme uygulamasının ana sınıfı (PyQt5 arayüzünü yönetir)"""

    def __init__(self):
        super().__init__()
        self.ses_dosyasi = None
        self.current_full_media_path = None
        self.modified_temp_full = None
        self.y = None
        self.sr = None
        self.duration = 0.0
        self.view_mode = 'waveform'
        self.spec_cache = None
        self.melspec_cache = None
        self.selection_start = 0.0
        self.selection_end = 0.0
        self.is_selecting = False
        self.current_position = 0.0
        self.position_line = None
        self.selection_position_line = None
        self.selection_rect = None
        # Spektrogram üstündeki dalga formu şeridi (isteğe bağlı) ve onun yansıtılan seçim/konum çizgisi
        self.ax_wave = None
        self._wave_sel = None
        self._wave_pos = None
        self.is_playing = False
        self.temp_file = None
        self._is_looping = False
        self._selection_dirty = True
        self._selection_sig = None
        self.current_xlim = None
        self.last_pan_x = None
        # CSV yolu, çalışma dizininden bağımsız olması için proje kökünden mutlak yol olarak
        # çözülür; csv_handler.py'deki init_csv/append_segment_to_csv ile aynı
        # get_project_root() mantığını kullanır.
        self.csv_file = os.path.join(csv_project_root(), "bird_sound_dataset.csv")
        self.controls_panel = None
        self.pan_slider = None
        self.file_info = None
        self.original_file_info = None
        self.segment_save_message = ""
        self.right_click_start_pos = None
        self.bird_calls = []  # UnifiedDetection listesi (yalnızca BirdNET)
        self.birdnet_raw_detections = []
        self.bird_call_rects = []
        self.detected_bird_calls_table = None

        self.habitat_data = HABITAT_DATA
        self.bird_species_data = BIRD_SPECIES_DATA
        self.config_file = "bird_sound_config.json"
        self.config_data = load_config()

        # İş parçacığı referansları
        self.loader_thread = None
        self.proc_thread = None
        self.detection_thread = None
        self.spec_thread = None
        self.feature_thread = None

        # Bellek yönetimi
        self.memory_watchdog_timer = QTimer()
        self.memory_watchdog_timer.timeout.connect(self.check_memory_usage)

        self.setWindowIcon(self.create_icon())

        # Not: sınıfın kendi load_config() metodu burada çağrılmaz; modül düzeyindeki
        # config.load_config() sonucunu farklı anahtar adlarıyla ve yanlış dosya
        # yoluyla ezip "Kişiler ve Doğrulama" ile "Ekipman" sekmelerini boş bırakıyordu.
        # Doğru veri yukarıda self.config_data = load_config() ile zaten yüklenir.
        self.initUI()
        self.init_csv()

        # Bellek bekçisini başlat (30 saniyede bir kontrol)
        self.memory_watchdog_timer.start(30000)

    def cleanup_memory(self):
        """Bellek temizliği yapar"""
        try:
            if hasattr(self, 'y') and self.y is not None:
                del self.y
                self.y = None

            if hasattr(self, 'spec_cache'):
                del self.spec_cache
                self.spec_cache = None

            if hasattr(self, 'melspec_cache'):
                del self.melspec_cache
                self.melspec_cache = None

            # Çöp toplayıcıyı çalıştır
            import gc
            gc.collect()

            print("Bellek temizliği tamamlandı")
        except Exception as e:
            print(f"Bellek temizleme hatası: {e}")

    def check_memory_usage(self):
        """Bellek kullanımını izler, gerekirse temizler"""
        try:
            import psutil
            import gc

            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            if memory_mb > 1000:  # 1 GB'tan fazlaysa
                print(f"Yüksek bellek kullanımı: {memory_mb:.1f} MB")
                gc.collect()

                # Kullanıcı uyarısı (isteğe bağlı)
                if memory_mb > 2000:  # 2 GB'tan fazlaysa
                    if hasattr(self, 'lbl_file_info'):
                        self.lbl_file_info.setText(
                            f"⚠️ Yüksek bellek kullanımı ({memory_mb:.0f} MB). "
                            f"Lütfen dosyayı kapatıp yeniden açın."
                        )
        except ImportError:
            pass
        except Exception as e:
            print(f"Bellek kontrol hatası: {e}")

    def create_menu_bar(self):
        """Uygulama menü çubuğunu oluşturur"""
        menubar = self.menuBar()

        # --------------------
        # 📁 Dosya Menüsü
        # --------------------
        file_menu = menubar.addMenu('📁 Dosya')

        open_action = QAction('📂 Ses Dosyası Aç...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.load_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction('🚪 Çıkış', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --------------------
        # 👁️ Görünüm Menüsü
        # --------------------
        view_menu = menubar.addMenu('👁️ Görünüm')

        waveform_action = QAction('📈 Dalga Formu', self)
        waveform_action.setShortcut('Ctrl+1')
        waveform_action.triggered.connect(lambda: self.set_view_mode('waveform'))
        view_menu.addAction(waveform_action)

        spec_action = QAction('📊 Spektrogram (Doğrusal)', self)
        spec_action.setShortcut('Ctrl+2')
        spec_action.triggered.connect(lambda: self.set_view_mode('spec'))
        view_menu.addAction(spec_action)

        melspec_action = QAction('🎵 Spektrogram (Mel)', self)
        melspec_action.setShortcut('Ctrl+3')
        melspec_action.triggered.connect(lambda: self.set_view_mode('melspec'))
        view_menu.addAction(melspec_action)

        view_menu.addSeparator()

        zoom_in_action = QAction('🔍 Yakınlaştır', self)
        zoom_in_action.setShortcut('Ctrl++')
        zoom_in_action.triggered.connect(lambda: self.zoom_buttons(1 / 1.2))
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction('🔍 Uzaklaştır', self)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(lambda: self.zoom_buttons(1.2))
        view_menu.addAction(zoom_out_action)

        zoom_reset_action = QAction('🔍 Yakınlaştırmayı Sıfırla', self)
        zoom_reset_action.setShortcut('Ctrl+0')
        zoom_reset_action.triggered.connect(self.reset_zoom)
        view_menu.addAction(zoom_reset_action)

        # --------------------
        # 🎵 Oynatma Menüsü
        # --------------------
        play_menu = menubar.addMenu('🎵 Oynatma')

        play_action = QAction('▶ Oynat/Duraklat (Tümü)', self)
        play_action.setShortcut('Space')
        play_action.triggered.connect(self.toggle_play)
        play_menu.addAction(play_action)

        stop_action = QAction('⏹ Durdur (Tümü)', self)
        stop_action.setShortcut('Ctrl+Space')
        stop_action.triggered.connect(self.stop_audio)
        play_menu.addAction(stop_action)

        rewind_action = QAction('⏪ 5 sn Geri', self)
        rewind_action.setShortcut('Ctrl+Left')
        rewind_action.triggered.connect(self.rewind)
        play_menu.addAction(rewind_action)

        forward_action = QAction('⏩ 5 sn İleri', self)
        forward_action.setShortcut('Ctrl+Right')
        forward_action.triggered.connect(self.forward)
        play_menu.addAction(forward_action)

        play_menu.addSeparator()

        play_selection_action = QAction('▶ Seçimi Oynat', self)
        play_selection_action.setShortcut('Ctrl+P')
        play_selection_action.triggered.connect(self.toggle_selection_play)
        play_menu.addAction(play_selection_action)

        loop_selection_action = QAction('🔁 Seçimi Döngüye Al', self)
        loop_selection_action.setShortcut('Ctrl+L')
        loop_selection_action.triggered.connect(self.loop_selection)
        play_menu.addAction(loop_selection_action)

        stop_selection_action = QAction('⏹ Seçimi Durdur', self)
        stop_selection_action.setShortcut('Ctrl+S')
        stop_selection_action.triggered.connect(self.stop_selection)
        play_menu.addAction(stop_selection_action)

        clear_selection_action = QAction('🗑️ Seçimi Temizle', self)
        clear_selection_action.setShortcut('Ctrl+D')
        clear_selection_action.triggered.connect(self.clear_selection)
        play_menu.addAction(clear_selection_action)

        # --------------------
        # ⚙️ İşlem Menüsü
        # --------------------
        process_menu = menubar.addMenu('⚙️ Ses İşleme')

        process_action = QAction('▶ İşlemleri Çalıştır (gürültü azaltma / LUFS)', self)
        process_action.setShortcut('Ctrl+R')
        process_action.triggered.connect(self.run_processing_pipeline)
        process_menu.addAction(process_action)

        # --------------------
        # 🔬 Analiz Menüsü
        # --------------------
        # Düzen, "🔬 Analiz" sekmesindeki kartlarla aynıdır: önce tespit, sonra veri seti analizi,
        # sonra tür sınıflandırma iş akışı (1. klasörler → 2. eğitim → 3. başarı sınırı).
        # "Ses İşleme" menüsü yalnızca ses ön işlemeyi, "Veri Seti" menüsü yalnızca CSV/segment
        # okuma-yazma işlemlerini kapsar.
        analysis_menu = menubar.addMenu('🔬 Analiz')

        detect_action = QAction('🔎 Kuş Ötüşlerini Tespit Et', self)
        detect_action.setShortcut('Ctrl+B')
        detect_action.triggered.connect(self.detect_bird_calls)
        analysis_menu.addAction(detect_action)

        clear_detect_action = QAction('🗑️ Tespiti Temizle', self)
        clear_detect_action.setShortcut('Ctrl+Shift+B')
        clear_detect_action.triggered.connect(self.clear_bird_call_detection)
        analysis_menu.addAction(clear_detect_action)

        analysis_menu.addSeparator()

        data_menu = analysis_menu.addMenu('📊 Veri Seti Analizi')

        dataset_summary_action = QAction('📈 Veri Seti Özeti', self)
        dataset_summary_action.setShortcut('Ctrl+U')
        dataset_summary_action.triggered.connect(self.show_dataset_summary)
        data_menu.addAction(dataset_summary_action)

        habitat_info_action = QAction('🌿 Habitat Bilgileri', self)
        habitat_info_action.setShortcut('Ctrl+H')
        habitat_info_action.triggered.connect(self.show_habitat_info)
        data_menu.addAction(habitat_info_action)

        species_richness_action = QAction('🧮 Tür Zenginliği Belirleme', self)
        species_richness_action.setShortcut('Ctrl+G')
        species_richness_action.triggered.connect(self.show_species_richness)
        data_menu.addAction(species_richness_action)

        location_map_action = QAction('🗺️ Konum Haritası', self)
        location_map_action.setShortcut('Ctrl+M')
        location_map_action.triggered.connect(self.show_location_map)
        data_menu.addAction(location_map_action)

        data_quality_action = QAction('🧹 Veri Kalitesi Denetimi', self)
        data_quality_action.setShortcut('Ctrl+Shift+C')
        data_quality_action.triggered.connect(self.show_data_quality)

        catalog_action = QAction('🗂 Tanım Verileri Yöneticisi (Tür, Habitat, Açılır Listeler)', self)
        catalog_action.setShortcut('Ctrl+Y')
        catalog_action.triggered.connect(self.show_catalog_manager)

        clf_menu = analysis_menu.addMenu('🧠 Tür Sınıflandırma')

        classification_folders_action = QAction('🗂️ 1. Sınıflandırma Klasörleri', self)
        classification_folders_action.setShortcut('Ctrl+K')
        classification_folders_action.triggered.connect(self.show_classification_folders)
        clf_menu.addAction(classification_folders_action)

        train_classifier_action = QAction('🎓 2. Tür Sınıflandırma Eğitimi (model seç)...', self)
        train_classifier_action.setShortcut('Ctrl+T')
        train_classifier_action.triggered.connect(lambda _checked=False: self.show_model_picker())
        clf_menu.addAction(train_classifier_action)

        for _n, (_mid, _spec) in enumerate(model_registry.MODELS.items()):
            _act = QAction(f'🧠 {_spec.ad} Eğitimi', self)
            if _n < 3:
                _act.setShortcut(f'F{5 + _n}')  # F5, F6, F7: sırayla BirdNET 2.4, Perch 2.0, birleşik
            _act.triggered.connect(lambda _checked=False, m=_mid: self.show_classifier_training(m))
            clf_menu.addAction(_act)

        model_compare_action = QAction('📊 Model Karşılaştırma', self)
        model_compare_action.setShortcut('Ctrl+Shift+K')
        model_compare_action.triggered.connect(lambda _checked=False: self.show_model_comparison())
        clf_menu.addAction(model_compare_action)

        clf_menu.addSeparator()

        classification_diagnosis_action = QAction('🔬 3. Başarı Sınırı Analizi', self)
        classification_diagnosis_action.setShortcut('Ctrl+Shift+D')
        classification_diagnosis_action.triggered.connect(self.show_classification_diagnosis)
        clf_menu.addAction(classification_diagnosis_action)

        # --------------------
        # 📊 Veri Seti Menüsü
        # --------------------
        # Burada CSV/segment okuma-yazma ve veri seti bakımı (denetim, tanım verileri) vardır;
        # analitik görünümler ve sınıflandırma iş akışı "🔬 Analiz" menüsündedir.
        dataset_menu = menubar.addMenu('📊 Veri Seti')

        view_csv_action = QAction('📊 CSV Görüntüle', self)
        view_csv_action.setShortcut('Ctrl+V')
        view_csv_action.triggered.connect(self.view_csv)
        dataset_menu.addAction(view_csv_action)

        view_csv_inapp_action = QAction('📋 Veri Setini Pencerede Göster', self)
        view_csv_inapp_action.setShortcut('Ctrl+Shift+V')
        view_csv_inapp_action.triggered.connect(self.show_csv_viewer)
        dataset_menu.addAction(view_csv_inapp_action)

        export_csv_action = QAction('📤 CSV Dışa Aktar...', self)
        export_csv_action.setShortcut('Ctrl+E')
        export_csv_action.triggered.connect(self.export_csv)
        dataset_menu.addAction(export_csv_action)

        dataset_menu.addSeparator()

        save_segment_action = QAction('💾 Segmenti Veri Setine Ekle', self)
        save_segment_action.setShortcut('Ctrl+Shift+S')
        save_segment_action.triggered.connect(self.save_segment)
        dataset_menu.addAction(save_segment_action)

        save_full_action = QAction('💾 Geçerli Kaydı WAV Olarak Kaydet...', self)
        save_full_action.setShortcut('Ctrl+W')
        save_full_action.triggered.connect(self.save_full_audio_to_wav)
        dataset_menu.addAction(save_full_action)

        # Veri seti bakımı: denetim ve tanım verileri (analiz değil, veri yönetimi; şeritte de aynı grupta)
        dataset_menu.addSeparator()
        dataset_menu.addAction(data_quality_action)
        dataset_menu.addAction(catalog_action)

        # --------------------
        # ❓ Yardım Menüsü
        # --------------------
        help_menu = menubar.addMenu('❓ Yardım')

        about_action = QAction('ℹ️ Hakkında', self)
        about_action.setShortcut('F1')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        shortcuts_action = QAction('⌨️ Kısayollar', self)
        shortcuts_action.setShortcut('F2')
        shortcuts_action.triggered.connect(self.show_shortcuts)
        help_menu.addAction(shortcuts_action)

    def save_config(self):
        """
        self.config_data içeriğini yapılandırma dosyasına geri kaydeder.

        Dosya yazma mantığı burada tutulmaz; yolu proje köküne göre DOĞRU
        çözen (get_project_root) modül düzeyindeki `config.save_config()`
        fonksiyonuna devredilir.
        """
        success = config_save_config(self.config_data, self.config_file)
        if not success:
            print("Yapılandırma dosyası kaydedilemedi (yukarıdaki config_save_config günlüğüne bakın).")

    def init_csv(self):
        """CSV dosyasını başlıklarla oluşturur (yoksa)."""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([
                    "Original_Recording_Name", "Original_File_Path", "Original_Duration",
                    "Channels", "File_Format", "Bit_Depth", "Recording_Date", "Recording_Time",
                    "Filename", "File_Path", "Start_Time", "End_Time", "Duration",
                    "Call_Type", "type_Count", "Order", "Family", "Scientific_Name", "Turkish_Name",
                    "Background_Order", "Background_Family", "Background_Scientific_Name",
                    "Background_Turkish_Name", "Background_Noise_Type",
                    "Notes_For_Other_Species", "Location",
                    "Recordist", "Ornithologist", "Verification_Status", "Confidence_Level",
                    "Microphone", "Recorder",
                    "Habitat_ID", "Habitat_Type", "Code", "UTM_Zone", "UTM_Easting", "UTM_Northing", "Lat", "Lon",
                    "Google_Maps", "Notes", "Project"
                ])

    def clear_feature_analysis(self):
        """Özellik analizi sonuçlarını ve ilgili iş parçacıklarını temizler."""
        pass

    def toggle_play(self):
        """Oynat/duraklat durumunu değiştirir (Tüm Kayıt)."""
        if self.selection_player.state() == QMediaPlayer.PlayingState:
            self.stop_selection()

        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.btn_play.setText("▶")
            self.is_playing = False
        else:
            self.player.play()
            self.btn_play.setText("⏸")
            self.is_playing = True

    def stop_audio(self):
        """Sesi durdurur (Tüm Kayıt)."""
        self.player.stop()
        self.btn_play.setText("▶")
        self.is_playing = False
        self.current_position = 0
        self.plot_waveform()

    def rewind(self):
        """5 saniye geri sarar (Tüm Kayıt)."""
        new_pos = max(0, self.player.position() - 5000)
        self.player.setPosition(new_pos)

    def forward(self):
        """5 saniye ileri sarar (Tüm Kayıt)."""
        new_pos = min(self.player.duration(), self.player.position() + 5000)
        self.player.setPosition(new_pos)

    def change_speed(self):
        """Oynatma hızını değiştirir."""
        speed = self.sld_speed.value() / 10.0
        self.lbl_speed.setText(f"{speed:.1f}x")
        self.player.setPlaybackRate(speed)

    def clear_selection(self):
        """Seçimi temizler."""
        self.selection_start = 0.0
        self.selection_end = 0.0
        self.mark_selection_dirty(stop_current=True)
        self.update_sliders_from_selection()
        self.update_selection_duration()
        self.plot_waveform()

    def ensure_selection_media_ready(self):
        """Seçim ortamının hazır olduğundan emin olur (gerekirse geçici dosya oluşturur)."""
        need_rebuild = self._selection_dirty
        cur_sig = self.current_selection_signature()

        if self._selection_sig != cur_sig:
            need_rebuild = True

        if self.temp_file is None or not os.path.exists(self.temp_file):
            need_rebuild = True

        if need_rebuild:
            self.create_selection_temp_file()
            self._selection_sig = cur_sig
            self._selection_dirty = False
            self.set_selection_media()

    def toggle_selection_play(self):
        """Seçili bölümü oynatır/duraklatır."""
        if self.y is None or abs(self.selection_end - self.selection_start) < 0.01:
            return

        if self.player.state() == QMediaPlayer.PlayingState:
            self.stop_audio()

        state = self.selection_player.state()

        if state == QMediaPlayer.PlayingState:
            self.selection_player.pause()
            self.btn_play_selection.setText("▶ Oynat")
        elif state == QMediaPlayer.PausedState:
            self.selection_player.play()
            self.btn_play_selection.setText("⏸ Duraklat")
        else:
            self.ensure_selection_media_ready()

            if self.temp_file and os.path.exists(self.temp_file):
                self.selection_player.setPlaybackRate(1.0)
                self.selection_player.play()
                self.btn_play_selection.setText("⏸ Duraklat")
                self.btn_stop_selection.setEnabled(True)
            else:
                QMessageBox.critical(self, "Hata", "Seçim için geçici ortam dosyası oluşturulamadı/bulunamadı.")

    def create_selection_temp_file(self):
        """Seçim için geçici WAV dosyası oluşturur."""
        if self.y is None or self.sr is None:
            return

        try:
            start_time = min(self.selection_start, self.selection_end)
            end_time = max(self.selection_start, self.selection_end)

            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)

            start_sample = max(0, min(start_sample, len(self.y)))
            end_sample = max(start_sample, min(end_sample, len(self.y)))

            if start_sample >= end_sample:
                return

            y_selected = self.y[start_sample:end_sample].copy()

            if self.temp_file and os.path.exists(self.temp_file):
                try:
                    os.unlink(self.temp_file)
                except Exception:
                    pass

            self.temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            write_wav_safe(self.temp_file, y_selected, self.sr)

        except Exception as e:
            print(f"❌ Seçim dosyası oluşturma hatası: {e}")
            self.temp_file = None
            QMessageBox.critical(self, "Hata", f"Seçilen aralık için geçici dosya oluşturulamadı: {str(e)}")

    def set_selection_media(self):
        """Seçim ortamını oynatıcıya bağlar."""
        try:
            if self.temp_file and os.path.exists(self.temp_file):
                self.selection_player.setMedia(QMediaContent(QUrl()))
                media_content = QMediaContent(QUrl.fromLocalFile(self.temp_file))
                self.selection_player.setMedia(media_content)
            else:
                self.selection_player.setMedia(QMediaContent(QUrl()))
        except Exception as e:
            print(f"Ortam ayarlama hatası: {e}")

    def loop_selection(self):
        """Seçili bölümü döngüde oynatır."""
        if self.y is None or abs(self.selection_end - self.selection_start) < 0.01:
            return

        if self.player.state() == QMediaPlayer.PlayingState:
            self.stop_audio()

        self._is_looping = True
        self.ensure_selection_media_ready()

        try:
            self.selection_player.mediaStatusChanged.disconnect(self.on_selection_media_status_changed)
        except Exception:
            pass

        self.selection_player.mediaStatusChanged.connect(self.on_selection_media_status_changed)

        if self.temp_file and os.path.exists(self.temp_file):
            self.selection_player.setPlaybackRate(1.0)
            self.selection_player.play()
            self.btn_play_selection.setText("⏸ Duraklat")
            self.btn_stop_selection.setEnabled(True)
        else:
            QMessageBox.critical(self, "Hata", "Seçim ortam içeriği yüklenemedi.")
            self._is_looping = False

    def stop_selection(self):
        """Seçim oynatmasını durdurur."""
        self.selection_player.stop()
        self.btn_play_selection.setText("▶ Oynat")
        self.btn_stop_selection.setEnabled(False)
        self._is_looping = False

        if self.selection_position_line:
            try:
                self.selection_position_line.remove()
            except Exception:
                pass
            self.selection_position_line = None
            self.canvas.draw_idle()

        try:
            self.selection_player.mediaStatusChanged.disconnect(self.on_selection_media_status_changed)
        except Exception:
            pass

    def on_selection_player_state_changed(self, state):
        """Seçim oynatıcısının durumu değiştiğinde çağrılır."""
        if state == QMediaPlayer.PlayingState:
            self.btn_play_selection.setText("⏸ Duraklat")
            self.btn_stop_selection.setEnabled(True)
        elif state == QMediaPlayer.PausedState:
            self.btn_play_selection.setText("▶ Oynat")
            self.btn_stop_selection.setEnabled(True)
        else:
            self.btn_play_selection.setText("▶ Oynat")
            self.btn_stop_selection.setEnabled(False)

            if self.selection_position_line:
                try:
                    self.selection_position_line.remove()
                except Exception:
                    pass
                self.selection_position_line = None
                self.canvas.draw_idle()

            if self._is_looping:
                self._is_looping = False

    def on_file_load_error(self, error_msg: str):
        """Dosya yükleme hatasında çağrılır."""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        QMessageBox.critical(self, "Hata", error_msg)
        self.lbl_file_info.setText("Dosya yükleme hatası")

    def enable_controls(self, enabled: bool):
        """Kontrol öğelerini etkinleştirir/devre dışı bırakır."""
        for b in (self.btn_play, self.btn_stop, self.btn_rewind, self.btn_forward,
                  self.btn_zoom_in, self.btn_zoom_out, self.btn_zoom_reset):
            b.setEnabled(enabled)

        has_selection = self.selection_end > self.selection_start
        for b in (self.btn_save, self.btn_play_selection, self.btn_loop_selection):
            b.setEnabled(enabled and has_selection)

        self.btn_stop_selection.setEnabled(False)
        if hasattr(self, 'btn_run_process'):
            self.btn_run_process.setEnabled(enabled)

    def on_turkish_name_changed(self, turkish_name):
        if not turkish_name:
            self.txt_scientific_name.clear()
            self.txt_family.clear()
            self.txt_order.clear()
        else:
            for bird in self.bird_species_data:
                if bird.get("turkish_name") == turkish_name:
                    self.txt_scientific_name.setText(bird.get("scientific_name", ""))
                    self.txt_family.setText(bird.get("family", ""))
                    self.txt_order.setText(bird.get("order", ""))
                    break
            else:
                self.txt_scientific_name.clear()
                self.txt_family.clear()
                self.txt_order.clear()

    def on_background_turkish_name_changed(self, background_turkish_name):
        if not background_turkish_name:
            self.txt_background_order.clear()
            self.txt_background_family.clear()
            self.txt_background_scientific_name.clear()
            return

        for bird in self.bird_species_data:
            if bird.get("turkish_name") == background_turkish_name:
                self.txt_background_order.setText(bird.get("order", ""))
                self.txt_background_family.setText(bird.get("family", ""))
                self.txt_background_scientific_name.setText(bird.get("scientific_name", ""))
                return

        self.txt_background_order.clear()
        self.txt_background_family.clear()
        self.txt_background_scientific_name.clear()

    def on_konum_changed(self, konum):
        """Konum değiştiğinde Habitat ID açılır listesini günceller."""
        if not konum:
            self.cmb_habitat_id.clear()
            self.cmb_habitat_id.addItem("")
            for habitat in self.habitat_data:
                self.cmb_habitat_id.addItem(str(habitat["habitat_ID"]), habitat)
            return

        self.cmb_habitat_id.clear()
        self.cmb_habitat_id.addItem("")

        for habitat in self.habitat_data:
            if habitat.get("konum", "") == konum:
                self.cmb_habitat_id.addItem(str(habitat["habitat_ID"]), habitat)

    def on_habitat_id_changed(self, index):
        """Habitat ID seçildiğinde konum ve habitat bilgilerini otomatik doldurur."""
        try:
            if index <= 0:
                self.txt_habitat.clear()
                self.txt_kod.clear()
                self.txt_utm_zone.clear()
                self.txt_utm_easting.clear()
                self.txt_utm_northing.clear()
                self.txt_lat.clear()
                self.txt_lon.clear()
                self.txt_google_maps.clear()
                self.btn_open_maps.setEnabled(False)
                return

            habitat = self.cmb_habitat_id.currentData()
            if habitat:
                self.txt_habitat.setText(habitat.get("habitat_tipi", ""))
                self.txt_kod.setText(habitat.get("kodu", ""))
                self.txt_utm_zone.setText(str(habitat.get("utm_zone", "")))
                self.txt_utm_easting.setText(str(habitat.get("utm_easting", "")))
                self.txt_utm_northing.setText(str(habitat.get("utm_northing", "")))
                self.txt_lat.setText(str(habitat.get("lat", "")))
                self.txt_lon.setText(str(habitat.get("lon", "")))

                konum = habitat.get("konum", "")
                if konum and self.cmb_location.currentText() != konum:
                    self.cmb_location.setCurrentText(konum)

                maps_link = habitat.get("google_maps", "")
                self.txt_google_maps.setText(maps_link)
                self.btn_open_maps.setEnabled(bool(maps_link and maps_link.startswith('http')))

        except Exception as e:
            print(f"Habitat ID değişiminde hata: {e}")

    def create_context_menu(self):
        """Sağ tık bağlam menüsünü oluşturur."""
        self.context_menu = QMenu(self)

        self.action_play_selection = QAction("▶ Seçimi Oynat", self)
        self.action_pause_selection = QAction("⏸ Seçimi Duraklat", self)
        self.action_stop_selection = QAction("⏹ Seçimi Durdur", self)
        self.action_clear_selection = QAction("🗑️ Seçimi Temizle", self)

        self.action_play_full = QAction("▶ Tüm Sesi Oynat/Duraklat", self)
        self.action_stop_full = QAction("⏹ Tüm Sesi Durdur", self)

        self.action_zoom_in = QAction("🔍 Yakınlaştır", self)
        self.action_zoom_out = QAction("🔍 Uzaklaştır", self)
        self.action_zoom_reset = QAction("🔍 Yakınlaştırmayı Sıfırla", self)

        self.context_menu.addAction(self.action_play_selection)
        self.context_menu.addAction(self.action_pause_selection)
        self.context_menu.addAction(self.action_stop_selection)
        self.context_menu.addAction(self.action_clear_selection)

        self.context_menu.addSeparator()
        self.context_menu.addAction(self.action_play_full)
        self.context_menu.addAction(self.action_stop_full)

        self.context_menu.addSeparator()
        self.context_menu.addAction(self.action_zoom_in)
        self.context_menu.addAction(self.action_zoom_out)
        self.context_menu.addAction(self.action_zoom_reset)

        self.action_play_selection.triggered.connect(self.toggle_selection_play)
        self.action_pause_selection.triggered.connect(self.selection_player.pause)
        self.action_stop_selection.triggered.connect(self.stop_selection)
        self.action_clear_selection.triggered.connect(self.clear_selection)

        self.action_play_full.triggered.connect(self.toggle_play)
        self.action_stop_full.triggered.connect(self.stop_audio)

        self.action_zoom_in.triggered.connect(lambda: self.zoom_buttons(1 / 1.2))
        self.action_zoom_out.triggered.connect(lambda: self.zoom_buttons(1.2))
        self.action_zoom_reset.triggered.connect(self.reset_zoom)

    def show_context_menu(self, event):
        """Sağ tık menüsünü gösterir ve eylem durumlarını günceller."""
        if event.inaxes == self.ax:
            has_selection = abs(self.selection_end - self.selection_start) > 0.01
            has_audio = self.y is not None

            self.action_play_selection.setEnabled(has_selection and has_audio)

            is_selection_playing = self.selection_player.state() == QMediaPlayer.PlayingState
            is_selection_loaded = self.selection_player.mediaStatus() != QMediaPlayer.NoMedia

            self.action_pause_selection.setEnabled(is_selection_playing)
            self.action_stop_selection.setEnabled(is_selection_playing or is_selection_loaded)

            self.action_clear_selection.setEnabled(has_selection)

            is_full_playing = self.player.state() == QMediaPlayer.PlayingState
            is_full_loaded = self.player.mediaStatus() != QMediaPlayer.NoMedia

            self.action_play_full.setText("⏸ Tümünü Duraklat" if is_full_playing else "▶ Tümünü Oynat")

            try:
                self.action_play_full.triggered.disconnect()
            except Exception:
                pass
            self.action_play_full.triggered.connect(self.toggle_play)
            self.action_play_full.setEnabled(has_audio and is_full_loaded)

            self.action_stop_full.setEnabled(has_audio and is_full_loaded)

            self.context_menu.exec_(QCursor.pos())

    def open_google_maps(self):
        """Google Maps bağlantısını tarayıcıda açar."""
        maps_url = self.txt_google_maps.text().strip()
        if maps_url and maps_url.startswith('http'):
            QDesktopServices.openUrl(QUrl(maps_url))
        else:
            QMessageBox.information(self, "Bilgi", "Geçerli bir Google Maps bağlantısı bulunamadı.")

    def set_view_mode(self, mode):
        """Görünüm modunu ayarlar (Dalga Formu, Doğrusal Spektrogram, Mel Spektrogram)."""
        modes = {'waveform': 0, 'spec': 1, 'melspec': 2}
        if mode in modes:
            self.cmb_view.setCurrentIndex(modes[mode])
            self.change_view()

    def show_about(self):
        """Hakkında bilgisini gösterir."""
        about_text = """
        <h3> 🐦 Yapay Zeka İle Kuş Seslerinin Akustik Analizi Ve Tür Zenginliğinin Belirlenmesi </h3>
        <h3> 🐦 BioSeg Labeling Studio v2.0 </h3>
        <p><b>BirdNET Destekli Biyoakustik Segmentasyon, Etiketleme ve Analiz Arayüzü</b></p>
        <p>© 2026 | Gökhan TURAN</p>
        <p><b>Özellikler:</b></p>
        <ul>
            <li>🎵 Gürültü azaltma (spektral kapılama) ve LUFS ses seviyesi normalizasyonu;
                kuş ötüşü tespitinden önce tek tıkla uygulanabilir</li>
            <li>🐦 BirdNET-Analyzer ile kuş ötüşü tespiti (Cornell K. Lisa Yang Merkezi)</li>
            <li>🇹🇷 Otomatik Türkçe/bilimsel ad eşleştirme (518 türlük derlenmiş veri tabanı)</li>
            <li>🔊 Arka plan tür bilgileri ve habitat üst veri alanlarıyla zengin veri seti etiketlemesi</li>
            <li>📊 Veri seti analizi: veri seti özeti, habitat bilgileri, konum haritası ve veri kalitesi denetimi
                (yinelenen/örtüşen/eksik dosyalı kayıtlar)</li>
            <li>🐦 Tür zenginliği belirleme: nokta başına tür sayısı, tür birikim eğrisi, habitat ve bölge
                karşılaştırması, uzman etiketi ile model çıktısının uyumu</li>
            <li>🧠 Tür sınıflandırma: BirdNET 2.4, Perch 2.0 ve ikisinin birleşimi dondurulmuş öznitelik
                çıkarıcı olarak kullanılır; kayıt bazlı çapraz doğrulama, modeller arası eşleştirilmiş
                karşılaştırma ve başarı sınırı analizi</li>
        </ul>
        <p><i>Not: Tür zenginliği, standart örnekleme koşullarında akustik olarak tespit edilen tür
        zenginliğidir; gerçek tür sayısının tahmini değildir.</i></p>
        """
        QMessageBox.about(self, "Hakkında", about_text)

    @staticmethod
    def _shortcut_label(seq) -> str:
        """QKeySequence -> Türkçe gösterim (Space→Boşluk, Left/Right→ok)."""
        t = seq.toString(QKeySequence.PortableText)
        for en, tr in (("Space", "Boşluk"), ("Left", "←"), ("Right", "→")):
            t = t.replace(en, tr)
        return t

    @staticmethod
    def _action_label(text: str) -> str:
        """Menü metninden baştaki simgeyi ve sondaki '...' işaretini temizler (kısayol listesi için)."""
        t = text.replace("&", "").strip()
        parts = t.split(" ", 1)
        if len(parts) == 2 and not any(ch.isalnum() for ch in parts[0]):
            t = parts[1]
        while t.startswith("🐦"):
            t = t[1:].lstrip()
        return t.rstrip(".").rstrip("…").strip()

    def shortcut_rows(self):
        """Menü çubuğundaki GERÇEK eylemlerden [(menü başlığı, [(kısayol, ad), ...])] üretir (liste hiç eskimez)."""
        groups = []

        def walk(menu, title):
            rows = None
            for g in groups:
                if g[0] == title:
                    rows = g[1]
                    break
            if rows is None:
                rows = []
                groups.append((title, rows))
            for act in menu.actions():
                sub = act.menu()
                if sub is not None:
                    walk(sub, title)
                    continue
                if act.isSeparator() or act.shortcut().isEmpty():
                    continue
                rows.append((self._shortcut_label(act.shortcut()), self._action_label(act.text())))

        for act in self.menuBar().actions():
            if act.menu() is not None:
                walk(act.menu(), act.text().replace("&", ""))
        return [g for g in groups if g[1]]

    def shortcuts_html(self) -> str:
        import html
        out = ["<h3>⌨️ Kısayol Tuşları</h3>"]
        for title, rows in self.shortcut_rows():
            out.append(f"<h4>{html.escape(title)}</h4><table cellpadding='2'>")
            out += [f"<tr><td><b>{html.escape(k)}</b></td><td>&nbsp;&nbsp;&nbsp;&nbsp;</td><td>{html.escape(n)}</td></tr>"
                    for k, n in rows]
            out.append("</table>")
        out.append("<h4>🎮 Grafik Kontrolleri</h4><table cellpadding='2'>"
                   "<tr><td><b>Fare Tekerleği</b></td><td>&nbsp;&nbsp;&nbsp;&nbsp;</td><td>Yakınlaştır</td></tr>"
                   "<tr><td><b>Shift + Tekerlek</b></td><td>&nbsp;&nbsp;&nbsp;&nbsp;</td><td>Yatay Kaydırma</td></tr>"
                   "<tr><td><b>Sağ Tık + Sürükle</b></td><td>&nbsp;&nbsp;&nbsp;&nbsp;</td><td>Kaydırma</td></tr>"
                   "<tr><td><b>Sol Tık + Sürükle</b></td><td>&nbsp;&nbsp;&nbsp;&nbsp;</td><td>Seçim</td></tr>"
                   "<tr><td><b>A / D</b></td><td>&nbsp;&nbsp;&nbsp;&nbsp;</td><td>Yatay Kaydırma</td></tr></table>")
        return "".join(out)

    def show_shortcuts(self):
        """Kısayol tuşlarını gösterir (liste menüden otomatik üretilir; uzun liste kaydırılabilir)."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Kısayollar")
        lay = QVBoxLayout(dlg)
        view = QTextBrowser()
        view.setHtml(self.shortcuts_html())
        lay.addWidget(view)
        dlg.resize(460, 640)
        dlg.exec_()

    def change_view(self):
        """Görünüm modunu değiştirir ve grafiği günceller."""
        if not hasattr(self, "cmb_view"):
            return

        idx = self.cmb_view.currentIndex()
        modes = ['waveform', 'spec', 'melspec']
        new_mode = modes[idx] if 0 <= idx < len(modes) else 'waveform'

        # Renk, kontrast, frekans aralığı ve şerit yalnızca spektrogram görünümlerinde anlamlı
        for _w in (self.cmb_cmap, self.spn_contrast, self.spn_fmin, self.spn_fmax, self.chk_wave_strip):
            _w.setEnabled(new_mode != 'waveform')

        if new_mode != self.view_mode:
            self.view_mode = new_mode
            if self.y is not None and self.sr is not None and \
                    (self.view_mode == 'spec' and self.spec_cache is None or self.view_mode == 'melspec' and self.melspec_cache is None):
                self.start_spectrogram_compute()
            else:
                self.plot_waveform()

    def start_spectrogram_compute(self):
        """Spektrogram hesabını arka plan iş parçacığında başlatır."""
        if self.y is None or self.sr is None:
            return

        if self.spec_thread and self.spec_thread.isRunning():
            self.spec_thread.quit()
            self.spec_thread.wait(500)

        self.lbl_file_info.setText("Spektrogramlar arka planda hesaplanıyor... Lütfen bekleyin.")
        QApplication.setOverrideCursor(Qt.BusyCursor)

        self.spec_thread = SpectrogramComputeThread(self.y, self.sr)
        self.spec_thread.finished.connect(self.on_spectrogram_computed)
        self.spec_thread.error.connect(self.on_spectrogram_error)
        self.spec_thread.start()

    def on_spectrogram_computed(self, mode, S_db, extent):
        """Spektrogram hesabı tamamlandığında çağrılır."""
        if mode == 'spec':
            self.spec_cache = (S_db, extent)
        elif mode == 'melspec':
            self.melspec_cache = (S_db, extent)

        if self.view_mode == mode:
            QApplication.restoreOverrideCursor()
            self.plot_waveform()

        if self.spec_cache is not None and self.melspec_cache is not None:
            QApplication.restoreOverrideCursor()
            self.lbl_file_info.setText(self.create_file_info_text())

    def on_spectrogram_error(self, msg):
        QApplication.restoreOverrideCursor()
        print(f"Spektrogram hatası: {msg}")

    def apply_compact_mode(self, on: bool = True):
        """
        Kompakt mod stil ayarlarını VE uygulamanın marka temasını (renkler, sekme
        stili, düğmeler, giriş alanları, tablolar, menüler) uygular. Kendi
        stil tanımı olan düğmeler (yeşil "Oynat", kırmızı "Durdur", mor "Kuş
        Ötüşlerini Tespit Et" gibi; renk bilerek anlam taşır) bu genel stilden
        ETKİLENMEZ: Qt, bir bileşenin kendi setStyleSheet() tanımına devralınan
        tanıma göre öncelik verir; yalnızca "sade" düğmeler marka rengini alır.
        İşlevsel bir değişiklik yoktur, yalnızca görsel düzenleme yapar.
        """
        base = 14 if on else 15
        pad = "6px 12px" if on else "9px 16px"

        # --- BioSeg v2.0 marka paleti (koyu camgöbeği; v1.0'ın görünümünden farklı) ---
        primary = theme.BRAND_GREEN
        primary_dark = "#0B6F57"
        primary_darker = "#085A47"
        primary_light = "#E3F3EE"
        border_color = "#D7E0DE"
        bg_color = "#F7F9F9"
        text_color = "#26312E"
        disabled_bg = "#B7C4C1"

        self.setStyleSheet(f"""
        QMainWindow, QWidget {{
            background: {bg_color};
            color: {text_color};
        }}

        QGroupBox {{
            font-size: {base + 1}px;
            font-weight: 600;
            margin-top: 10px;
            border: 1px solid {border_color};
            border-radius: 8px;
            background: white;
            color: {primary_dark};
            padding-top: 6px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            color: {primary_dark};
        }}

        QLabel, QCheckBox {{ font-size: {base}px; }}
        QComboBox, QSlider, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox,
        QTableWidget, QTableView {{ font-size: {base}px; }}

        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ min-height: 24px; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {{
            border: 1px solid {border_color};
            border-radius: 4px;
            padding: 3px 6px;
            background: white;
            selection-background-color: {primary_light};
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
            border: 1px solid {primary};
        }}
        QLineEdit:read-only {{ background: {bg_color}; color: #5b6663; }}

        QPushButton {{
            font-size: {base}px;
            padding: {pad};
            background: {theme.gradient()};
            color: white;
            border: none;
            border-radius: 6px;
        }}
        QPushButton:hover {{ background: {theme.gradient(theme.BRAND_BLUE_HOVER, theme.BRAND_GREEN_HOVER)}; }}
        QPushButton:pressed {{ background: {theme.gradient(theme.BRAND_BLUE_PRESSED, theme.BRAND_GREEN_PRESSED)}; }}
        QPushButton:disabled {{ background: {disabled_bg}; color: #EEF2F1; }}

        QTabWidget::pane {{
            border: 1px solid {border_color};
            border-radius: 6px;
            top: -1px;
            background: white;
        }}
        QTabBar::tab {{
            background: #E3ECEA;
            padding: 8px 22px;
            margin-right: 3px;
            margin-top: 3px;
            border: 1px solid {border_color};
            border-bottom: none;
            border-top-left-radius: 9px;
            border-top-right-radius: 9px;
            color: #3d4f4b;
            font-weight: 700;
            font-size: {base + 1}px;
        }}
        QTabBar::tab:hover {{
            color: {primary_dark};
            background: #CFE6E1;
        }}
        QTabBar::tab:selected {{
            color: white;
            background: {theme.gradient()};
            border: 1px solid {primary};
            border-bottom: none;
            font-weight: 700;
            margin-top: 0px;
        }}
        QTabBar::scroller {{ width: 26px; }}

        QTableWidget {{
            gridline-color: {border_color};
            background: white;
            alternate-background-color: {primary_light};
        }}
        QHeaderView::section {{
            background: {primary_dark};
            color: white;
            padding: 4px;
            border: none;
        }}

        QSlider::groove:horizontal {{
            height: 4px;
            background: {border_color};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {primary};
            width: 14px;
            margin: -6px 0;
            border-radius: 7px;
        }}

        QMenuBar {{ background: {theme.gradient()}; color: white; }}
        QMenuBar::item {{ background: transparent; padding: 4px 10px; }}
        QMenuBar::item:selected {{ background: rgba(255,255,255,0.22); }}
        QMenu {{ background: white; border: 1px solid {border_color}; }}
        QMenu::item:selected {{ background: {primary_light}; color: {primary_dark}; }}

        QScrollBar:vertical, QScrollBar:horizontal {{
            background: {bg_color};
            border: none;
        }}
        QScrollBar::handle {{
            background: {border_color};
            border-radius: 4px;
        }}
        QScrollBar::handle:hover {{ background: {primary}; }}
        """)

        # Qt, sekme genişliğini stil sayfasındaki yazı tipinden DEĞİL, sekme çubuğunun
        # kendi yazı tipinden hesaplar; bu yüzden büyütülmüş/kalın yazı sekmede kırpılır.
        # Sekme çubuklarına aynı yazı tipini doğrudan veriyoruz ve kırpmayı kapatıyoruz.
        from PyQt5.QtGui import QFont
        tab_font = QFont(self.font())
        tab_font.setPixelSize(base + 1)
        tab_font.setBold(True)
        for tw in self.findChildren(QTabWidget):
            bar = tw.tabBar()
            bar.setFont(tab_font)
            bar.setElideMode(Qt.ElideNone)
            bar.setExpanding(False)
            bar.setUsesScrollButtons(True)
            bar.updateGeometry()

    def load_file(self):
        """Kullanıcıdan ses dosyası seçmesini ister ve yüklemeyi başlatır."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Ses Dosyası Seç", "", "WAV Dosyaları (*.wav)")
        if not file_path:
            return

        if os.path.splitext(file_path)[1].lower() != ".wav":
            QMessageBox.information(self, "Uyarı", "Yalnızca .wav dosyaları desteklenmektedir.")
            return

        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 500:
                reply = QMessageBox.question(
                    self,
                    "Büyük Dosya Uyarısı",
                    f"Dosya boyutu {file_size_mb:.1f} MB. Bu boyuttaki dosyalar bellek sorunlarına yol açabilir.\n"
                    f"Devam etmek istiyor musunuz?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
        except Exception as e:
            print(f"Dosya boyutu kontrol hatası: {e}")

        self.ses_dosyasi = file_path

        threads_to_stop = [self.loader_thread, self.proc_thread, self.detection_thread, self.spec_thread,
                           self.feature_thread]
        for thread in threads_to_stop:
            if thread and thread.isRunning():
                thread.quit()
                thread.wait(500)

        self.progress_dialog = QProgressDialog("Dosya yükleniyor...", "İptal", 0, 100, self)
        self.progress_dialog.setWindowTitle("Lütfen Bekleyin")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.show()

        self.loader_thread = FileLoaderThread(file_path)
        self.progress_dialog.canceled.connect(self.on_load_canceled)
        self.loader_thread.progress.connect(self.update_progress_dialog)
        self.loader_thread.finished.connect(self.on_file_loaded)
        self.loader_thread.error.connect(self.on_file_load_error)
        self.loader_thread.start()

    def on_load_canceled(self):
        threads_to_stop = [self.loader_thread, self.spec_thread, self.feature_thread]
        for thread in threads_to_stop:
            if thread and thread.isRunning():
                thread.quit()
                thread.wait(2000)

        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        self.lbl_file_info.setText("Yükleme iptal edildi")

    def update_progress_dialog(self, value: int):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setValue(value)

    def on_file_loaded(self, y, sr, duration: float, file_info: dict):
        self.cleanup_memory()

        self.y, self.sr, self.duration = y, sr, duration
        self.file_info = file_info
        self.original_file_info = file_info.copy()

        self.file_info['processed_duration'] = duration
        self.file_info['processed_sample_rate'] = sr

        if 'original_sample_rate' not in self.file_info:
            self.file_info['original_sample_rate'] = file_info['sample_rate']

        self.current_xlim = (0, self.duration)
        self.spec_cache = None
        self.melspec_cache = None

        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

        self.clear_feature_analysis()
        self.clear_bird_call_detection()

        self.lbl_segment_message.setText("")
        self.update_file_info_display()
        self.lbl_total_time.setText(self.format_time(self.duration))

        max_slider_value = int(self.duration * 100)
        self.sld_start.setRange(0, max_slider_value)
        self.sld_end.setRange(0, max_slider_value)
        self.sld_end.setValue(max_slider_value)

        self.selection_start = 0.0
        self.selection_end = self.duration
        self.mark_selection_dirty(stop_current=True)
        self.update_sliders_from_selection()
        self.update_selection_duration()
        self.plot_waveform()
        self.enable_controls(True)
        self.btn_clear.setEnabled(True)

        if librosa is not None:
            self.btn_detect_bird_calls.setEnabled(True)

        self.current_full_media_path = self.ses_dosyasi
        media_content = QMediaContent(QUrl.fromLocalFile(self.current_full_media_path))
        self.player.setMedia(media_content)
        self._clear_modified_full_temp()
        self.btn_save_full.setEnabled(True)

        self.start_spectrogram_compute()

    @staticmethod
    def format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def current_selection_signature(self):
        return (
            round(self.selection_start, 6),
            round(self.selection_end, 6),
            self.sr,
            len(self.y) if self.y is not None else 0,
        )

    def mark_selection_dirty(self, stop_current: bool = False):
        self._selection_dirty = True
        if stop_current and self.selection_player.state() == QMediaPlayer.PlayingState:
            self.stop_selection()

        if self.selection_position_line:
            try:
                self.selection_position_line.remove()
            except Exception:
                pass
            self.selection_position_line = None
            self.canvas.draw_idle()

    # ---------------- Spektrogram görünüm ayarları (yalnızca gösterim) ----------------
    def on_view_settings_changed(self, *_):
        if getattr(self, "y", None) is not None:
            self.plot_waveform()

    def reset_view_settings(self):
        for w in (self.cmb_cmap, self.spn_contrast, self.spn_fmin, self.spn_fmax, self.chk_wave_strip):
            w.blockSignals(True)
        self.cmb_cmap.setCurrentIndex(0)
        self.spn_contrast.setValue(0)
        self.spn_fmin.setValue(0)
        self.spn_fmax.setValue(0)
        self.chk_wave_strip.setChecked(False)
        for w in (self.cmb_cmap, self.spn_contrast, self.spn_fmin, self.spn_fmax, self.chk_wave_strip):
            w.blockSignals(False)
        self.on_view_settings_changed()

    def _wave_strip_wanted(self) -> bool:
        return (getattr(self, 'chk_wave_strip', None) is not None and self.chk_wave_strip.isChecked()
                and self.view_mode in ('spec', 'melspec')
                and self.y is not None and self.y.size > 0)

    def _sync_axes_layout(self, dual: bool):
        """Üstte dalga formu şeridi varsa iki satırlı, yoksa tek eksenli düzen. self.ax her zaman etkileşimli ana eksendir."""
        fig = self.figure
        if dual and self.ax_wave is None:
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.14)
            self.ax.set_subplotspec(gs[1])
            self.ax_wave = fig.add_subplot(gs[0], sharex=self.ax)
        elif not dual and self.ax_wave is not None:
            try:
                self.ax_wave.remove()
            except Exception:
                pass
            self.ax_wave = None
            self.ax.set_subplotspec(fig.add_gridspec(1, 1)[0])
        if not dual:
            self._wave_sel = None
            self._wave_pos = None

    def _draw_wave_strip(self):
        """Ana eksenin üstündeki dalga formu şeridi (salt görüntü; seçim ana eksende yapılır)."""
        aw = self.ax_wave
        aw.clear()
        n = len(self.y)
        step = max(1, n // 10000)
        aw.plot(np.linspace(0, self.duration, n)[::step], self.y[::step], color='#1f4e79', linewidth=0.5)
        aw.set_ylabel('Genlik')
        aw.grid(True, alpha=0.25)
        aw.tick_params(labelbottom=False)
        aw.set_title(self._view_title())
        for det in self.bird_calls:
            aw.add_patch(Rectangle((det.start_s, 0), det.end_s - det.start_s, 1, alpha=0.3, color="green",
                                   transform=aw.get_xaxis_transform()))
        start_val = min(self.selection_start, self.selection_end)
        width = abs(self.selection_end - self.selection_start)
        self._wave_sel = Rectangle((start_val, 0), max(width, 0.0), 1, alpha=0.3, color='red', zorder=10,
                                   transform=aw.get_xaxis_transform())
        self._wave_sel.set_visible(width > 1e-3)
        aw.add_patch(self._wave_sel)
        self._wave_pos = aw.axvline(x=self.current_position, color='r', linestyle='-', linewidth=1.5)

    def _view_title(self) -> str:
        name = {'waveform': 'Dalga Formu', 'spec': 'Spektrogram (Doğrusal)', 'melspec': 'Spektrogram (Mel)'}[self.view_mode]
        return f'{name} — Seçmek için tıklayıp sürükleyin (Sağ tık: Menü)'

    def save_view_png(self):
        """Grafiği 300 dpi PNG kaydeder; seçim, oynatma çizgisi ve tespit bölgeleri kaydedilmez."""
        if self.y is None:
            QMessageBox.information(self, "Kaydedilemedi", "Önce bir ses dosyası açın.")
            return
        src_path = getattr(getattr(self, "loader_thread", None), "file_path", "") or "goruntu"
        base = os.path.splitext(os.path.basename(src_path))[0]
        mode = {'waveform': 'dalga', 'spec': 'spektrogram', 'melspec': 'mel'}[self.view_mode]
        path, _ = QFileDialog.getSaveFileName(
            self, "Görünümü PNG olarak kaydet", os.path.join(csv_project_root(), f"{base}_{mode}.png"), "PNG (*.png)")
        if not path:
            return
        hidden = []
        for art in [self.selection_rect, self.position_line, self.selection_position_line, self._wave_sel,
                    self._wave_pos] + list(self.bird_call_rects):
            if art is not None and art.get_visible():
                art.set_visible(False)
                hidden.append(art)
        try:
            self.figure.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
        except Exception as e:
            QMessageBox.warning(self, "Kaydedilemedi", f"PNG yazılamadı:\n{e}")
        else:
            QMessageBox.information(self, "Kaydedildi", f"Kaydedildi (300 dpi):\n{path}")
        finally:
            for art in hidden:
                art.set_visible(True)
            self.canvas.draw_idle()

    def plot_waveform(self):
        self._sync_axes_layout(self._wave_strip_wanted())
        if self.ax_wave is not None:
            self.ax_wave.clear()
        self.ax.clear()
        if self.y is None or self.y.size == 0:
            self.ax.text(0.5, 0.5, "Dosya Yüklenmedi", ha='center', va='center', transform=self.ax.transAxes,
                         fontsize=16)
            self.ax.set_xlabel('Zaman (sn)')
            self.ax.set_ylabel('Genlik / Frekans')
            self.canvas.draw()
            return

        if self.view_mode == 'waveform':
            if len(self.y) > 10000:
                step = len(self.y) // 10000
                time_arr = np.linspace(0, self.duration, len(self.y))[::step]
                y_down = self.y[::step]
            else:
                time_arr = np.linspace(0, self.duration, len(self.y))
                y_down = self.y
            self.ax.plot(time_arr, y_down, 'b-', alpha=0.7, linewidth=0.5)
            self.ax.set_ylabel('Genlik')
        else:
            cache = self.spec_cache if self.view_mode == 'spec' else self.melspec_cache
            if cache is None:
                self.ax.text(0.5, 0.5, "Spektrogram hesaplanıyor...",
                             ha='center', va='center', transform=self.ax.transAxes, color='orange', fontsize=12)
                self.ax.set_xlabel('Zaman (sn)')
                self.ax.set_ylabel('Frekans (Hz)')
                self.canvas.draw()
                self.start_spectrogram_compute()
                return

            S_db, extent = cache
            try:
                if S_db.size == 0:
                    self.ax.text(0.5, 0.5, "Ses çok kısa, spektrogram oluşturulamadı.",
                                 ha='center', va='center', transform=self.ax.transAxes, color='red', fontsize=10)
                else:
                    from ..visualization import spectrogram_view as sv
                    vmin, vmax = sv.contrast_limits(S_db, self.spn_contrast.value())
                    self.ax.imshow(S_db, origin='lower', aspect='auto', extent=extent, vmin=vmin, vmax=vmax,
                                   cmap=self.cmb_cmap.currentData() or 'magma')
                    nyq = self.sr / 2.0
                    lo, hi = sv.y_limits(self.view_mode, self.spn_fmin.value(), self.spn_fmax.value(), nyq,
                                         n_mels=extent[3])
                    self.ax.set_ylim(lo, hi)
                    if self.view_mode == 'melspec':
                        pos, lab = sv.mel_ticks(self.spn_fmin.value(), self.spn_fmax.value(), nyq, n_mels=extent[3])
                        self.ax.set_yticks(pos)
                        self.ax.set_yticklabels(lab)
                self.ax.set_ylabel('Frekans (Hz)' if self.view_mode == 'spec' else 'Frekans (mel ölçeğinde)')
            except Exception as e:
                self.ax.text(0.5, 0.5, f"Spektrogram hatası:\n{e}",
                             ha='center', va='center', transform=self.ax.transAxes, color='red', fontsize=10)
                self.ax.set_ylabel('')
                self.canvas.draw()
                return

        self.ax.set_xlabel('Zaman (sn)')
        if self.ax_wave is not None:
            self._draw_wave_strip()      # başlık şeritte; ana eksende başlık yok
        else:
            self.ax.set_title(self._view_title())
        self.ax.grid(True, alpha=0.25 if self.view_mode == 'waveform' else False)

        for call_rect in self.bird_call_rects:
            try:
                call_rect.remove()
            except Exception:
                pass
        self.bird_call_rects = []

        y_min, y_max = self.ax.get_ylim()
        height = y_max - y_min

        for det in self.bird_calls:
            start = det.start_s
            end = det.end_s

            rect = Rectangle(
                (start, y_min),
                end - start,
                height,
                alpha=0.3, color="green",
                label=f"BirdNET: {det.birdnet_species}"
            )
            self.ax.add_patch(rect)
            self.bird_call_rects.append(rect)

        start_val = min(self.selection_start, self.selection_end)
        width = abs(self.selection_end - self.selection_start)

        if width > 1e-3:
            if not self.selection_rect or self.selection_rect not in self.ax.patches:
                if self.selection_rect:
                    try:
                        self.selection_rect.remove()
                    except Exception:
                        pass

                self.selection_rect = Rectangle(
                    (start_val, y_min),
                    width,
                    height,
                    alpha=0.3, color='red',
                    zorder=10
                )
                self.ax.add_patch(self.selection_rect)
            else:
                self.selection_rect.set_xy((start_val, y_min))
                self.selection_rect.set_width(width)
                self.selection_rect.set_height(height)
                self.selection_rect.set_visible(True)
        else:
            if self.selection_rect:
                self.selection_rect.set_visible(False)

        if not self.position_line or self.position_line not in self.ax.lines:
            self.position_line = self.ax.axvline(x=self.current_position, color='r', linestyle='-', linewidth=2)
        else:
            self.position_line.set_xdata([self.current_position, self.current_position])

        if self.selection_position_line and self.selection_player.state() != QMediaPlayer.PlayingState:
            try:
                self.selection_position_line.remove()
            except Exception:
                pass
            self.selection_position_line = None

        if self.current_xlim is not None:
            self.ax.set_xlim(self.current_xlim)
        else:
            self.ax.set_xlim(0, max(self.duration, 1e-6))

        self.update_pan_slider_state()
        self.canvas.draw()

    def zoom_buttons(self, scale_factor: float):
        if self.duration <= 0:
            return

        x_min, x_max = self.ax.get_xlim()
        cur_width = x_max - x_min
        xdata = (x_min + x_max) / 2
        new_width = max((cur_width * scale_factor), 1e-3)
        left = xdata - new_width / 2
        right = xdata + new_width / 2
        left = max(0.0, left)
        right = min(self.duration, right)
        if right - left < 1e-3:
            right = left + 1e-3

        if right == self.duration and (right - left) < new_width:
            left = max(0.0, right - new_width)

        self.current_xlim = (left, right)
        self.ax.set_xlim(self.current_xlim)
        self.update_pan_slider_state()
        self.canvas.draw()

    def reset_zoom(self):
        self.current_xlim = (0, self.duration)
        self.ax.set_xlim(self.current_xlim)
        self.update_pan_slider_state()
        self.canvas.draw()

    def on_scroll(self, event):
        if self.duration <= 0 or event.inaxes != self.ax:
            return

        if event.key == 'shift':
            x_min, x_max = self.ax.get_xlim()
            cur_width = x_max - x_min
            step = (cur_width) * (0.1 if event.button == 'up' else -0.1)
            left = max(0.0, x_min - step)
            right = left + cur_width

            if right > self.duration:
                right = self.duration
                left = max(0.0, right - cur_width)

            self.current_xlim = (left, right)
            self.ax.set_xlim(self.current_xlim)
            self.update_pan_slider_state()
            self.canvas.draw()
            return

        x_min, x_max = self.ax.get_xlim()
        cur_width = x_max - x_min
        xdata = event.xdata if event.xdata is not None else (x_min + x_max) / 2
        base_scale = 1.2
        scale_factor = (1 / base_scale) if event.button == 'up' else base_scale
        new_width = cur_width * scale_factor
        left = xdata - (xdata - x_min) * (new_width / cur_width)
        right = left + new_width

        left = max(0.0, left)
        right = min(self.duration, right)

        if right - left < 1e-3:
            right = left + 1e-3

        if right == self.duration and (right - left) < new_width:
            left = max(0.0, right - new_width)

        self.current_xlim = (left, right)
        self.ax.set_xlim(self.current_xlim)
        self.update_pan_slider_state()
        self.canvas.draw()

    def on_pan_slider(self, value: int):
        if self.duration <= 0 or self.current_xlim is None:
            return

        view_width = self.current_xlim[1] - self.current_xlim[0]
        if view_width >= self.duration:
            return

        max_left = max(0.0, self.duration - view_width)
        left = (value / 1000.0) * max_left
        right = left + view_width
        self.current_xlim = (left, right)
        self.ax.set_xlim(self.current_xlim)
        self.canvas.draw()

    def update_pan_slider_state(self):
        if self.duration <= 0 or self.current_xlim is None:
            self.pan_slider.setEnabled(False)
            return

        view_width = self.current_xlim[1] - self.current_xlim[0]
        need_pan = bool(view_width < float(self.duration) - 1e-6)
        self.pan_slider.setEnabled(need_pan)

        if need_pan:
            max_left = max(0.0, self.duration - view_width)
            pos = int((self.current_xlim[0] / max_left) * 1000) if max_left > 0 else 0
            pos = min(max(pos, 0), 1000)
            if self.pan_slider.value() != pos:
                self.pan_slider.blockSignals(True)
                self.pan_slider.setValue(pos)
                self.pan_slider.blockSignals(False)

    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        if event.button == 3:
            self.last_pan_x = event.xdata
            self.right_click_start_pos = (event.x, event.y)
            return

        if event.button == 1:
            self.is_selecting = True
            self.selection_start = float(event.xdata)
            self.selection_end = float(event.xdata)
            self.mark_selection_dirty(stop_current=False)
            self.update_sliders_from_selection()
            self.update_selection_duration()
            self.plot_waveform()

    def on_release(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 3:
            if (self.right_click_start_pos is not None and
                    abs(event.x - self.right_click_start_pos[0]) < 3 and
                    abs(event.y - self.right_click_start_pos[1]) < 3):
                self.show_context_menu(event)

            self.last_pan_x = None
            self.right_click_start_pos = None
            return

        if event.xdata is None:
            self.is_selecting = False
            return

        if event.button == 1:
            self.is_selecting = False
            self.selection_end = float(event.xdata)

            if self.selection_start > self.selection_end:
                self.selection_start, self.selection_end = self.selection_end, self.selection_start

            self.selection_start = max(0.0, min(self.selection_start, self.duration))
            self.selection_end = max(0.0, min(self.selection_end, self.duration))

            self.mark_selection_dirty(stop_current=True)
            self.update_sliders_from_selection()
            self.update_selection_duration()
            self.plot_waveform()

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return

        if self.last_pan_x is not None and event.xdata is not None:
            x_min, x_max = self.ax.get_xlim()
            cur_width = x_max - x_min
            dx = self.last_pan_x - event.xdata
            left = max(0.0, x_min + dx)
            right = left + cur_width

            if right > self.duration:
                right = self.duration
                left = max(0.0, right - cur_width)

            self.current_xlim = (left, right)
            self.ax.set_xlim(self.current_xlim)
            self.update_pan_slider_state()
            self.canvas.draw()
            self.last_pan_x = event.xdata
            return

        if not self.is_selecting or event.xdata is None:
            return

        self.selection_end = float(event.xdata)

        temp_start = min(self.selection_start, self.selection_end)
        temp_end = max(self.selection_start, self.selection_end)

        temp_start = max(0.0, min(temp_start, self.duration))
        temp_end = max(0.0, min(temp_end, self.duration))

        self.lbl_start.setText(f"Başlangıç: {temp_start:.3f} sn")
        self.lbl_end.setText(f"Bitiş: {temp_end:.3f} sn")

        if self.selection_rect:
            current_y_min, current_y_max = self.ax.get_ylim()
            self.selection_rect.set_width(temp_end - temp_start)
            self.selection_rect.set_xy((temp_start, current_y_min))
            self.selection_rect.set_visible(True)
            if self._wave_sel is not None:
                self._wave_sel.set_xy((temp_start, 0))
                self._wave_sel.set_width(temp_end - temp_start)
                self._wave_sel.set_visible(True)
            self.canvas.draw_idle()

    def keyPressEvent(self, event):
        if self.duration <= 0:
            super().keyPressEvent(event)
            return

        key = event.key()
        mods = event.modifiers()

        if mods & Qt.ControlModifier:
            if key == Qt.Key_O:
                self.load_file()
                return
            if key == Qt.Key_V:
                self.view_csv()
                return
            if key == Qt.Key_E:
                self.export_csv()
                return
            if key == Qt.Key_Q:
                self.close()
                return
            if key == Qt.Key_1:
                self.set_view_mode('waveform')
                return
            if key == Qt.Key_2:
                self.set_view_mode('spec')
                return
            if key == Qt.Key_3:
                self.set_view_mode('melspec')
                return
            if key == Qt.Key_R:
                self.run_processing_pipeline()
                return
            if key == Qt.Key_B:
                self.detect_bird_calls()
                return
            if key == Qt.Key_P:
                self.toggle_selection_play()
                return
            if key == Qt.Key_L:
                self.loop_selection()
                return
            if key == Qt.Key_S:
                self.stop_selection()
                return
            if key == Qt.Key_D:
                self.clear_selection()
                return
            if key == Qt.Key_W:
                self.save_full_audio_to_wav()
                return
            if key == Qt.Key_Plus or key == Qt.Key_Equal:
                self.zoom_buttons(1 / 1.2)
                return
            if key == Qt.Key_Minus:
                self.zoom_buttons(1.2)
                return
            if key == Qt.Key_0:
                self.reset_zoom()
                return
            if key == Qt.Key_Space:
                self.stop_audio()
                return

            if mods & Qt.ShiftModifier and key == Qt.Key_S:
                self.save_segment()
                return
            if mods & Qt.ShiftModifier and key == Qt.Key_B:
                self.clear_bird_call_detection()
                return

        if key in (Qt.Key_F1, Qt.Key_F2):
            if key == Qt.Key_F1:
                self.show_about()
            elif key == Qt.Key_F2:
                self.show_shortcuts()
            return

        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_buttons(1 / 1.2)
            return
        if key in (Qt.Key_Minus,):
            self.zoom_buttons(1.2)
            return
        if key == Qt.Key_0:
            self.reset_zoom()
            return

        base = 0.01
        if mods & Qt.ControlModifier:
            base = 5.0
        elif mods & Qt.AltModifier:
            base = 0.005

        if key in (Qt.Key_Left, Qt.Key_Right):
            sign = -1 if key == Qt.Key_Left else 1

            if mods & Qt.ControlModifier:
                new_pos = self.player.position() + sign * 5000
                self.player.setPosition(new_pos)

            else:
                if (mods & Qt.ShiftModifier):
                    self.selection_end = np.clip(self.selection_end + sign * base, 0, self.duration)
                    if self.selection_end < self.selection_start:
                        self.selection_end = self.selection_start
                else:
                    width = max(0.0, self.selection_end - self.selection_start)
                    new_start = np.clip(self.selection_start + sign * base, 0, max(0, self.duration - width))
                    self.selection_start = new_start
                    self.selection_end = new_start + width

                self.mark_selection_dirty(stop_current=True)
                self.update_sliders_from_selection()
                self.update_selection_duration()
                self.plot_waveform()
            return

        if key in (Qt.Key_A, Qt.Key_D):
            x_min, x_max = self.ax.get_xlim()
            cur_width = x_max - x_min
            step = cur_width * (0.1 if key == Qt.Key_D else -0.1)
            left = max(0.0, x_min + step)
            right = left + cur_width

            if right > self.duration:
                right = self.duration
                left = max(0.0, right - cur_width)

            self.current_xlim = (left, right)
            self.ax.set_xlim(self.current_xlim)
            self.update_pan_slider_state()
            self.canvas.draw()
            return

        if key == Qt.Key_Space:
            self.toggle_play()
            return

        super().keyPressEvent(event)

    def update_selection_duration(self):
        start = min(self.selection_start, self.selection_end)
        end = max(self.selection_start, self.selection_end)
        duration = end - start

        if duration <= 0.01:
            self.lbl_selection_duration.setText("Seçim yok")
            for b in (self.btn_play_selection, self.btn_loop_selection, self.btn_stop_selection, self.btn_save):
                b.setEnabled(False)
        else:
            self.lbl_selection_duration.setText(f"Seçilen süre: {duration:.3f} saniye")
            for b in (self.btn_play_selection, self.btn_loop_selection, self.btn_save):
                b.setEnabled(True)
            self.btn_stop_selection.setEnabled(False)

    def update_sliders_from_selection(self):
        if self.duration > 0:
            start = min(self.selection_start, self.selection_end)
            end = max(self.selection_start, self.selection_end)

            start_value = int(start * 100)
            end_value = int(end * 100)

            self.sld_start.blockSignals(True)
            self.sld_end.blockSignals(True)

            self.sld_start.setValue(start_value)
            self.sld_end.setValue(end_value)

            self.sld_start.blockSignals(False)
            self.sld_end.blockSignals(False)

            self.lbl_start.setText(f"Başlangıç: {start:.3f} sn")
            self.lbl_end.setText(f"Bitiş: {end:.3f} sn")

    def update_time_from_slider(self):
        self.selection_start = self.sld_start.value() / 100.0
        self.selection_end = self.sld_end.value() / 100.0

        self.selection_start = max(0.0, min(self.selection_start, self.duration))
        self.selection_end = max(0.0, min(self.selection_end, self.duration))

        if self.sld_start.sender() == self.sld_start and self.selection_start > self.selection_end:
            self.selection_end = self.selection_start
            self.sld_end.blockSignals(True)
            self.sld_end.setValue(int(self.selection_end * 100))
            self.sld_end.blockSignals(False)
        elif self.sld_end.sender() == self.sld_end and self.selection_end < self.selection_start:
            self.selection_start = self.selection_end
            self.sld_start.blockSignals(True)
            self.sld_start.setValue(int(self.selection_start * 100))
            self.sld_start.blockSignals(False)

        self.lbl_start.setText(f"Başlangıç: {self.selection_start:.3f} sn")
        self.lbl_end.setText(f"Bitiş: {self.selection_end:.3f} sn")

        self.mark_selection_dirty(stop_current=True)
        self.update_selection_duration()
        self.plot_waveform()

    def update_position(self, position_ms: int):
        self.current_position = position_ms / 1000
        self.lbl_current_time.setText(self.format_time(self.current_position))

        if self.position_line:
            self.position_line.set_xdata([self.current_position, self.current_position])
            if self._wave_pos is not None:
                self._wave_pos.set_xdata([self.current_position, self.current_position])
            self.canvas.draw_idle()

    def update_selection_position(self, position_ms: int):
        if self.selection_player.state() == QMediaPlayer.PlayingState:
            selection_start_time = min(self.selection_start, self.selection_end)
            selection_current_pos = selection_start_time + (position_ms / 1000.0)

            selection_end_time = max(self.selection_start, self.selection_end)

            if self._is_looping and position_ms >= self.selection_player.duration() - 50:
                self.selection_player.setPosition(0)
                return

            if selection_current_pos >= selection_end_time - 0.01:
                self.stop_selection()
                return

            if self.selection_position_line and self.selection_position_line in self.ax.lines:
                self.selection_position_line.set_xdata([selection_current_pos, selection_current_pos])
            else:
                self.selection_position_line = self.ax.axvline(
                    x=selection_current_pos,
                    color='green',
                    linestyle='-',
                    linewidth=2,
                    zorder=11
                )

            self.canvas.draw_idle()

    def update_duration(self, duration_ms: int):
        if duration_ms > 0:
            self.sld_progress.setRange(0, duration_ms)

    def update_progress(self):
        try:
            if self.player and self.player.duration() > 0:
                self.sld_progress.setValue(self.player.position())
        except Exception:
            pass

    def seek_audio(self, position: int):
        self.player.setPosition(position)

    def run_processing_pipeline(self):
        if self.y is None or self.sr is None:
            QMessageBox.information(self, "Bilgi", "Önce bir ses dosyası yükleyin.")
            return

        if self.proc_thread and self.proc_thread.isRunning():
            self.proc_thread.quit()
            self.proc_thread.wait(500)

        use_selection = (self.cmb_proc_source.currentIndex() == 1)
        t0 = min(self.selection_start, self.selection_end) if use_selection else 0.0
        t1 = max(self.selection_start, self.selection_end) if use_selection else self.duration

        if use_selection and not (t1 > t0 + 0.01):
            QMessageBox.information(self, "Bilgi",
                                    "Seçilen aralık çok kısa veya geçersiz. Lütfen bir aralık seçin veya 'Tüm kayıt' seçeneğini kullanın.")
            return

        do_denoise = self.chk_denoise.isChecked()
        do_lufs = self.chk_lufs.isChecked()

        if do_denoise and nr is None:
            QMessageBox.information(self, "Bilgi", "noisereduce kurulu değil. Kurmak için: pip install noisereduce")
            return
        if do_lufs and pyln is None:
            QMessageBox.information(self, "Bilgi", "pyloudnorm kurulu değil. Kurmak için: pip install pyloudnorm")
            return

        self.lbl_proc_status.setText("İşlem başladı…")
        self.btn_run_process.setEnabled(False)
        QApplication.setOverrideCursor(Qt.BusyCursor)

        self.proc_thread = AudioProcessThread(
            y=self.y, sr=self.sr, t0=t0, t1=t1,
            do_denoise=do_denoise, do_lufs=do_lufs, target_lufs=-16.0
        )
        self.proc_thread.progress.connect(lambda p: self.lbl_proc_status.setText(f"İşleniyor… %{p}"))
        self.proc_thread.finished.connect(self.on_processing_finished)
        self.proc_thread.error.connect(self.on_processing_error)
        self.proc_thread.start()

    def on_processing_finished(self, result: dict):
        QApplication.restoreOverrideCursor()
        self.btn_run_process.setEnabled(True)

        y_out = result.get('y_out', None)
        t0 = float(result.get('t0', 0.0))
        t1 = float(result.get('t1', self.duration))

        if y_out is None:
            self.lbl_proc_status.setText("İşlem bitti ancak çıktı üretilmedi.")
            return

        start = int(t0 * self.sr)
        end = int(t1 * self.sr)

        start = max(0, min(start, len(self.y)))
        end = max(start, min(end, len(self.y)))

        target_length = end - start

        if len(y_out) != target_length:
            if len(y_out) > target_length:
                y_out = y_out[:target_length]
            else:
                y_out = np.pad(y_out, (0, target_length - len(y_out)), mode='constant')

        if start >= 0 and end <= len(self.y) and target_length > 0:
            self.y[start:end] = y_out

        self.spec_cache = None
        self.melspec_cache = None
        self.plot_waveform()
        self.lbl_proc_status.setText("İşlem tamamlandı. Dalga formu güncellendi. Spektrogram yeniden hesaplanacak.")

        self.start_spectrogram_compute()

        self._rebuild_full_temp_and_bind_player()
        self.mark_selection_dirty(stop_current=True)
        self.btn_save_full.setEnabled(True)

    def on_processing_error(self, msg: str):
        QApplication.restoreOverrideCursor()
        self.btn_run_process.setEnabled(True)
        self.lbl_proc_status.setText(f"Hata: {msg}")
        QMessageBox.critical(self, "Hata", msg)

    def _rebuild_full_temp_and_bind_player(self):
        if self.y is None or self.sr is None:
            return

        try:
            self._clear_modified_full_temp()
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            self.modified_temp_full = temp_file.name

            write_wav_safe(self.modified_temp_full, self.y, self.sr)
            self.current_full_media_path = self.modified_temp_full

            media_content = QMediaContent(QUrl.fromLocalFile(self.current_full_media_path))
            self.player.setMedia(media_content)

            self.lbl_total_time.setText(self.format_time(self.duration))
        except Exception as e:
            self.lbl_proc_status.setText(f"Tüm kaydın hazırlanmasında hata: {e}")

    def _clear_modified_full_temp(self):
        try:
            if self.modified_temp_full and os.path.exists(self.modified_temp_full):
                os.unlink(self.modified_temp_full)
        except Exception as e:
            print(f"Geçici dosya silinirken hata: {e}")
        finally:
            self.modified_temp_full = None

    def save_full_audio_to_wav(self):
        if self.y is None or self.sr is None:
            QMessageBox.information(self, "Bilgi", "Kaydedilecek ses bulunamadı.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Geçerli Kaydı WAV Olarak Kaydet", "", "WAV (*.wav)")
        if not file_path:
            return

        try:
            write_wav_safe(file_path, self.y, self.sr)
            QMessageBox.information(self, "Başarılı", f"Kaydedildi: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Kaydedilemedi: {str(e)}")

    def update_file_info_display(self):
        if self.file_info is None:
            return

        file_info_text = self.create_file_info_text()
        self.lbl_file_info.setText(file_info_text)

    def create_file_info_text(self):
        if self.file_info is None:
            return "Dosya bilgisi bulunamadı"

        info = self.file_info

        duration = info.get('processed_duration', info.get('duration', 0.0))
        sample_rate = info.get('processed_sample_rate', info.get('sample_rate', 'Bilinmiyor'))
        original_sample_rate = info.get('original_sample_rate', 'Bilinmiyor')

        sr_info = f"{sample_rate}Hz"
        if sample_rate != original_sample_rate:
            sr_info += f" (Orijinal: {original_sample_rate}Hz)"

        file_info_text = (
            f"<table width='100%' border='0' cellspacing='0' cellpadding='5' style='font-size: 15px;'>"
            f"<tr>"
            f"<td width='50%' valign='top' style='border-right: 1px solid #ccc; padding-right: 15px;'>"
            f"<b>📁 Temel Bilgiler</b><br>"
            f"═══════════════════════════════<br>"
            f"• 📄 <b>Dosya:</b> {os.path.basename(self.ses_dosyasi)}<br>"
            f"• 📂 <b>Yol:</b> {self.ses_dosyasi}<br>"
            f"• 💾 <b>Boyut:</b> {info['file_size_mb']:.2f} MB<br>"
            f"• ⏱️ <b>Süre:</b> {duration:.2f} sn<br>"
            f"• 📅 <b>Kayıt Tarihi:</b> {info['recording_date']}<br>"
            f"• 🕒 <b>Kayıt Saati:</b> {info['recording_time']}<br>"
            f"</td>"
            f"<td width='50%' valign='top'>"
            f"<b>🔧 Teknik Özellikler</b><br>"
            f"═══════════════════════════════<br>"
            f"• 🔊 <b>Kanal:</b> {info['channels']}<br>"
            f"• 📊 <b>Örnekleme:</b> {sr_info}<br>"
            f"• 🎚️ <b>Bit Derinliği:</b> {info['bit_depth']}<br>"
            f"• 📁 <b>Biçim:</b> {info['file_format']}<br>"
            f"• 🔄 <b>Kare Sayısı:</b> {len(self.y):,}<br>"
            f"</td>"
            f"</tr>"
            f"</table>"
        )

        return file_info_text

    def update_detection_params(self):
        """Geçerli BirdNET parametre değerlerini durum etiketinde gösterir."""
        self.lbl_detection_status.setText(
            f"Parametreler: BirdNET min. güven={self.spn_birdnet_conf.value():.2f}"
        )

    def detect_bird_calls(self):
        """
        BirdNET-Analyzer'ı TÜM kayıt üzerinde çalıştırır. Bu sekmedeki "Gürültü
        Azaltma"/"LUFS Normalizasyonu" ön işleme kutuları İŞARETLİYSE, tarama
        başlamadan ÖNCE bu adımlar tüm kayda uygulanır (self.y KALICI olarak
        güncellenir, dalga formu/spektrogram da yenilenir); ardından BirdNET
        işlenmiş sesi analiz eder. Böylece kullanıcı unutsa bile gürültü
        azaltma/ses seviyesi normalizasyonu otomatik uygulanır.
        """
        if self.y is None or self.sr is None:
            QMessageBox.information(self, "Bilgi", "Önce bir ses dosyası yükleyin.")
            return

        if not self.ses_dosyasi or not os.path.exists(self.ses_dosyasi):
            QMessageBox.warning(self, "Uyarı", "Orijinal ses dosyasının yolu diskte bulunamadı.")
            return

        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.quit()
            self.detection_thread.wait(500)

        # Uzun sürebilecek bir ön işlemi boşa harcamamak için konum alanları ÖNCE doğrulanır.
        lat = lon = None
        lat_text = self.txt_birdnet_lat.text().strip()
        lon_text = self.txt_birdnet_lon.text().strip()
        if lat_text and lon_text:
            try:
                lat = float(lat_text)
                lon = float(lon_text)
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Enlem/Boylam sayısal olmalıdır (örn. 37.76).")
                return
        self._pending_birdnet_lat = lat
        self._pending_birdnet_lon = lon

        self.btn_detect_bird_calls.setEnabled(False)
        QApplication.setOverrideCursor(Qt.BusyCursor)

        do_denoise = self.chk_denoise_before_detect.isChecked()
        do_lufs = self.chk_lufs_before_detect.isChecked()

        if do_denoise or do_lufs:
            self.lbl_detection_status.setText("Ön işleme uygulanıyor (Gürültü Azaltma/LUFS Normalizasyonu)...")

            if getattr(self, "_pre_detect_proc_thread", None) and self._pre_detect_proc_thread.isRunning():
                self._pre_detect_proc_thread.quit()
                self._pre_detect_proc_thread.wait(500)

            self._pre_detect_proc_thread = AudioProcessThread(
                y=self.y, sr=self.sr, t0=0.0, t1=self.duration,
                do_denoise=do_denoise, do_lufs=do_lufs, target_lufs=-16.0,
            )
            self._pre_detect_proc_thread.progress.connect(
                lambda p: self.lbl_detection_status.setText(f"Ön işleme… %{p}")
            )
            self._pre_detect_proc_thread.finished.connect(self._on_pre_detection_processing_finished)
            self._pre_detect_proc_thread.error.connect(self._on_pre_detection_processing_error)
            self._pre_detect_proc_thread.start()
        else:
            self._start_species_detection_thread(self.ses_dosyasi)

    def _on_pre_detection_processing_finished(self, result: dict):
        """
        "Kuş Ötüşlerini Tespit Et" öncesinde çalıştırılan Gürültü Azaltma/LUFS
        ön işlemesinin sonucu: self.y'yi ("İşlemleri Çalıştır" düğmesi gibi, ama
        TÜM kayıt üzerinde) KALICI olarak günceller, işlenmiş sesi geçici bir
        WAV'a yazar ve BirdNET taramasını başlatır.
        """
        y_out = result.get('y_out', None)
        if y_out is not None and len(y_out) > 0:
            target_length = len(self.y)
            if len(y_out) != target_length:
                if len(y_out) > target_length:
                    y_out = y_out[:target_length]
                else:
                    y_out = np.pad(y_out, (0, target_length - len(y_out)), mode='constant')
            self.y = y_out

            self.spec_cache = None
            self.melspec_cache = None
            self.plot_waveform()
            self.start_spectrogram_compute()
            self._rebuild_full_temp_and_bind_player()

        try:
            if getattr(self, "_pre_detect_temp_wav", None) and os.path.exists(self._pre_detect_temp_wav):
                os.unlink(self._pre_detect_temp_wav)
        except Exception as e:
            print(f"Geçici ön işleme dosyası silinemedi: {e}")

        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        self._pre_detect_temp_wav = tmp.name
        write_wav_safe(self._pre_detect_temp_wav, self.y, self.sr)

        self.lbl_detection_status.setText("Ön işleme tamamlandı. BirdNET taraması başlatılıyor...")
        self._start_species_detection_thread(self._pre_detect_temp_wav)

    def _on_pre_detection_processing_error(self, msg: str):
        QApplication.restoreOverrideCursor()
        self.btn_detect_bird_calls.setEnabled(True)
        QMessageBox.critical(
            self, "Ön İşleme Hatası",
            f"'Kuş Ötüşlerini Tespit Et' öncesindeki Gürültü Azaltma/LUFS Normalizasyonu sırasında hata oluştu:\n{msg}",
        )

    def _start_species_detection_thread(self, analysis_filepath: str):
        """
        `analysis_filepath`: BirdNET'in analiz edeceği dosya; ön işleme
        uygulanmadıysa orijinal dosya, uygulandıysa işlenmiş sesin yazıldığı
        geçici WAV dosyasıdır.
        """
        self.lbl_detection_status.setText("Kuş ötüşleri tespit ediliyor (BirdNET)...")

        self.detection_thread = SpeciesDetectionThread(
            self.y, self.sr, analysis_filepath,
            birdnet_min_conf=self.spn_birdnet_conf.value(),
            birdnet_lat=self._pending_birdnet_lat,
            birdnet_lon=self._pending_birdnet_lon,
        )
        self.detection_thread.progress.connect(
            lambda p, msg: self.lbl_detection_status.setText(f"{msg} ({p}%)")
        )
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.error.connect(self.on_detection_error)
        self.detection_thread.start()

    def on_detection_finished(self, unified_rows, perch_raw, birdnet_raw):
        QApplication.restoreOverrideCursor()
        self.btn_detect_bird_calls.setEnabled(True)
        self.btn_clear_detection.setEnabled(True)

        self.bird_calls = unified_rows
        self.birdnet_raw_detections = birdnet_raw
        self.detected_bird_calls_table.setRowCount(0)

        if len(unified_rows) == 0:
            self.lbl_detection_status.setText(
                "❌ Tespit yok: güven eşiğini düşürmeyi deneyin veya BirdNET'in "
                "doğru kurulduğunu kontrol edin (README'ye bakın).")
            self.plot_waveform()
            return

        self.lbl_detection_status.setText(f"{len(unified_rows)} segment tespit edildi.")

        columns = [
            "Seg. ID", "Başlangıç (sn)", "Bitiş (sn)", "Süre (sn)",
            "BirdNET — Latince", "BirdNET — Türkçe", "BirdNET Güven (%)",
        ]
        self.detected_bird_calls_table.setColumnCount(len(columns))
        self.detected_bird_calls_table.setHorizontalHeaderLabels(columns)
        self.detected_bird_calls_table.setRowCount(len(self.bird_calls))

        for row, det in enumerate(self.bird_calls):
            duration = det.end_s - det.start_s

            values = [
                str(det.row_id),
                f"{det.start_s:.3f}",
                f"{det.end_s:.3f}",
                f"{duration:.3f}",
                det.birdnet_species or "—",
                det.birdnet_turkish or ("(listede yok)" if det.has_birdnet else "—"),
                f"{det.birdnet_confidence:.1f}" if det.birdnet_confidence is not None else "—",
            ]
            for col, val in enumerate(values):
                self.detected_bird_calls_table.setItem(row, col, QTableWidgetItem(val))

        self.detected_bird_calls_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.detected_bird_calls_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.detected_bird_calls_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)

        self.plot_waveform()

    def on_detection_error(self, msg):
        """
        Ölümcül olmayan uyarıyı gösterir; kullanıcı tespit sırasında bir
        iletişim kutusunu kapatmaya zorlanmasın diye kipsiz (non-modal) bir
        mesaj olarak sunulur; mesaj durum etiketine eklenir ve ayrıca bir kez
        uyarı olarak gösterilir.
        """
        print(f"Tespit uyarısı: {msg}")
        self.lbl_detection_status.setText(f"⚠ {msg.splitlines()[0]}")
        QMessageBox.warning(self, "Tespit Uyarısı", msg.splitlines()[0])

    def clear_bird_call_detection(self):
        self.bird_calls = []
        self.birdnet_raw_detections = []
        for rect in self.bird_call_rects:
            if rect in self.ax.patches:
                rect.remove()
        self.bird_call_rects = []
        self.detected_bird_calls_table.setRowCount(0)
        self.lbl_detection_status.setText("Kuş ötüşü tespiti temizlendi")
        self.btn_clear_detection.setEnabled(False)

        self.clear_feature_analysis()
        self.plot_waveform()

    def on_bird_call_item_clicked(self, row, column):
        """
        Tıklanan segmenti seçer ve oynatır; tür etiketleme formunu (Türkçe ad /
        bilimsel ad / familya / takım) bu segment için BirdNET'in tahminine
        göre otomatik doldurur (v1.0'daki kümeleme tabanlı group_label_map
        yayılımının yerini alır). Tahmin edilen bilimsel ad derlenmiş 518
        türlük veri tabanında yoksa bilimsel ad yine gösterilir, ancak Türkçe
        açılır liste elle girilmek üzere boş bırakılır.
        """
        if not self.bird_calls or row >= len(self.bird_calls):
            return

        det = self.bird_calls[row]

        self.selection_start = det.start_s
        self.selection_end = det.end_s

        self.mark_selection_dirty(stop_current=True)
        self.update_sliders_from_selection()
        self.update_selection_duration()
        self.plot_waveform()

        self.toggle_selection_play()

        suggested_species = det.best_species_for_labeling

        if suggested_species is None:
            self.cmb_turkish_name.setCurrentText("")
            self.lbl_segment_message.setText(
                f"Segment {det.row_id} seçildi ve çalıyor. Burada tür önerisi "
                "yok. Lütfen etiketi elle atayın."
            )
            return

        info = species_lookup.lookup(suggested_species)

        if info is not None:
            self.cmb_turkish_name.setCurrentText(info["turkish_name"])
            self.lbl_segment_message.setText(
                f"Segment {det.row_id}: BirdNET ile önerilen tür '{info['turkish_name']}' "
                f"({suggested_species}). Çalıyor — kaydetmeden önce lütfen doğrulayın."
            )
        else:
            # Bilimsel ad veri tabanında yok; yine de kullanıcıya gösterilir.
            self.cmb_turkish_name.setCurrentText("")
            self.txt_scientific_name.setText(suggested_species)
            self.txt_family.clear()
            self.txt_order.clear()
            self.lbl_segment_message.setText(
                f"Segment {det.row_id}: BirdNET '{suggested_species}' türünü tahmin etti; bu tür "
                "yerel 518 türlük veri tabanında yok. Bilimsel ad önceden dolduruldu; "
                "lütfen Türkçe adı elle seçin/onaylayın."
            )

    def save_segment(self):
        if self.y is None:
            QMessageBox.warning(self, "Uyarı", "Önce bir ses dosyası yükleyin.")
            return

        selection_duration = abs(self.selection_end - self.selection_start)

        if selection_duration == 0:
            QMessageBox.warning(self, "Uyarı", "Hiçbir alan seçilmedi. Lütfen bir aralık seçin.")
            return

        if selection_duration < 0.01:
            QMessageBox.warning(self, "Uyarı",
                                "Seçilen alan çok kısa (< 0,01 sn). Lütfen daha uzun bir alan seçin.")
            return

        total_duration = self.duration

        if total_duration > 0 and (selection_duration / total_duration) > 0.95:
            reply = QMessageBox.question(
                self,
                "Tüm Kayıt Seçildi",
                f"Kaydın tamamını seçtiniz (Seçim: {selection_duration:.2f} sn, Toplam: {total_duration:.2f} sn).\n\n"
                "Devam etmek istiyor musunuz?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                return

        turkish_name = self.cmb_turkish_name.currentText().strip()
        call_type = self.cmb_call_type.currentText().strip()

        if not turkish_name or not call_type:
            QMessageBox.critical(self, "Hata", "Lütfen 'Türkçe Ad' ve 'Çağrı Türü' alanlarını doldurun.")
            return

        location = self.cmb_location.currentText().strip()
        if not location:
            QMessageBox.critical(self, "Hata", "Lütfen 'Konum' alanını seçin.")
            return

        habitat_id_index = self.cmb_habitat_id.currentIndex()
        if habitat_id_index <= 0:
            QMessageBox.critical(self, "Hata", "Lütfen 'Habitat ID' alanını seçin.")
            return

        type_count = self.cmb_type_count.currentData() or self.cmb_type_count.currentText()
        if type_count == "Multiple":
            background_turkish_name = self.cmb_background_turkish_name.currentText().strip()
            if not background_turkish_name:
                QMessageBox.critical(self, "Hata",
                                     "Tür sayısı 'Çoklu' seçildiğinde 'Arka Plan Türkçe Adı' alanı doldurulmalıdır.")
                return

        try:
            segment_filename = self._write_single_segment_to_disk_and_csv()

            if self.cmb_habitat_id.currentIndex() > 0:
                habitat = self.cmb_habitat_id.currentData()
                habitat_id = str(habitat.get("habitat_ID", ""))
                habitat_tipi = habitat.get("habitat_tipi", "")
                habitat_kodu = habitat.get("kodu", "")
            else:
                habitat_id = ""
                habitat_tipi = self.txt_habitat.text() or ""
                habitat_kodu = self.txt_kod.text() or ""

            if self.file_info:
                kayit_tarihi = self.file_info.get('recording_date', 'Bilinmiyor')
                kayit_saati = self.file_info.get('recording_time', 'Bilinmiyor')
            else:
                kayit_tarihi = "Bilinmiyor"
                kayit_saati = "Bilinmiyor"

            message = (
                f"<div style='font-family: Arial, sans-serif;'>"
                f"<h3 style='text-align: center; color: #2e7d32;'>✅ SEGMENT KAYDEDİLDİ</h3>"
                f"<hr>"
                f"<div><b>Dosya:</b> {segment_filename}</div>"
                f"<div><b>Tür:</b> {turkish_name}</div>"
                f"<div><b>Tür Sayısı:</b> {self.cmb_type_count.currentText()}</div>"
                f"<div><b>Konum:</b> {location}</div>"
                f"<div><b>Habitat ID:</b> {habitat_id}</div>"
                f"<div><b>Arka Plan Türü:</b> {self.cmb_background_turkish_name.currentText()}</div>"
                f"<div><b>Arka Plan Gürültüsü:</b> {self.cmb_background_noise.currentText()}</div>"
                f"<div><b>Segment Süresi:</b> {selection_duration:.3f} sn</div>"
                f"<hr>"
                f"<div style='font-size: 14px; color: #666;'>"
                f"'segments' klasörüne kaydedildi</div>"
                f"</div>"
            )

            msg_box = QMessageBox()
            msg_box.setWindowTitle("BioSeg Labeling Studio v2.0")
            msg_box.setTextFormat(Qt.RichText)
            msg_box.setText(message)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.exec_()

            self.lbl_segment_message.setText(f"✅ Tekil segment kaydedildi: {segment_filename}")

        except Exception as e:
            self.lbl_segment_message.setText(f"❌ Kaydetme hatası: {str(e)}")
            QMessageBox.critical(self, "Hata", f"Segment kaydedilemedi: {str(e)}")

    def _write_single_segment_to_disk_and_csv(self):
        import re

        start_time = min(self.selection_start, self.selection_end)
        end_time = max(self.selection_start, self.selection_end)

        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)

        start_sample = max(0, min(start_sample, len(self.y)))
        end_sample = max(start_sample, min(end_sample, len(self.y)))

        if start_sample >= end_sample:
            raise ValueError("Geçersiz veya çok kısa ses dilimi.")

        y_selected = self.y[start_sample:end_sample]

        # PyInstaller uyumlu: derlenmiş (frozen) sürümde "segments/" klasörü .exe'nin
        # yanında bulunur (CSV/yapılandırma dosyalarıyla aynı kural; bkz.
        # csv_handler.get_project_root()); aksi halde geliştirme ortamındaki gibi
        # src/segments/ kullanılır.
        if getattr(sys, "frozen", False):
            segments_dir = os.path.join(os.path.dirname(sys.executable), "segments")
        else:
            segments_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "segments")

        os.makedirs(segments_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.ses_dosyasi or "recording"))[0]

        habitat_id_str = ""
        if self.cmb_habitat_id.currentIndex() > 0:
            habitat = self.cmb_habitat_id.currentData()
            habitat_id_str = str(habitat.get("habitat_ID", ""))
        else:
            numbers = re.findall(r'\d{2,}', base_name)
            if numbers:
                habitat_id_str = numbers[0]

        turkce_kus_turu_adi = self.cmb_turkish_name.currentText().strip()
        if not turkce_kus_turu_adi:
            turkce_kus_turu_adi = "UNKNOWN"
        else:
            turkce_kus_turu_adi = turkce_kus_turu_adi.replace(" ", "_").replace(",", "")

        tarih = ""
        tarih_match = re.search(r'(\d{8})', base_name)
        if tarih_match:
            tarih = tarih_match.group(1)
        else:
            if self.file_info and 'creation_time' in self.file_info:
                tarih = self.file_info['creation_time'].strftime("%Y%m%d")
            else:
                tarih = datetime.now().strftime("%Y%m%d")

        parts = [base_name]
        if habitat_id_str:
            parts.append(f"Habitat{habitat_id_str}")
        parts.append(turkce_kus_turu_adi)
        parts.append(tarih)
        segment_filename_base = "_".join(parts) + ".wav"
        segment_filename_base = re.sub(r'[<>:"/\\|?*]', '', segment_filename_base)

        segment_filename = segment_filename_base
        counter = 1
        while os.path.exists(os.path.join(segments_dir, segment_filename)):
            name_without_ext, ext = os.path.splitext(segment_filename_base)
            segment_filename = f"{name_without_ext}_{counter}{ext}"
            counter += 1

        segment_filepath = os.path.join(segments_dir, segment_filename)
        write_wav_safe(segment_filepath, y_selected, self.sr)

        habitat_id_index = self.cmb_habitat_id.currentIndex()
        if habitat_id_index > 0:
            habitat = self.cmb_habitat_id.currentData()
            habitat_ID = habitat.get("habitat_ID", "")
            habitat_tipi = habitat.get("habitat_tipi", "")
            kod = habitat.get("kodu", "")
            utm_zone = habitat.get("utm_zone", "")
            utm_easting = habitat.get("utm_easting", "")
            utm_northing = habitat.get("utm_northing", "")
            lat = habitat.get("lat", "")
            lon = habitat.get("lon", "")
            google_maps = habitat.get("google_maps", "")
        else:
            habitat_ID = ""
            habitat_tipi = self.txt_habitat.text()
            kod = self.txt_kod.text()
            utm_zone = self.txt_utm_zone.text()
            utm_easting = self.txt_utm_easting.text()
            utm_northing = self.txt_utm_northing.text()
            lat = self.txt_lat.text()
            lon = self.txt_lon.text()
            google_maps = self.txt_google_maps.text()

        with open(self.csv_file, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([
                os.path.basename(self.ses_dosyasi) if self.ses_dosyasi else "(temp)",
                self.ses_dosyasi if self.ses_dosyasi else "",
                round(self.file_info['duration'], 3) if self.file_info else 0,
                self.file_info['channels'] if self.file_info else "",
                self.file_info['file_format'] if self.file_info else "WAV",
                self.file_info['bit_depth'] if self.file_info else "16 bit",
                self.file_info['recording_date'] if self.file_info else "",
                self.file_info['recording_time'] if self.file_info else "",
                segment_filename,
                segment_filepath,
                round(start_time, 3),
                round(end_time, 3),
                round(end_time - start_time, 3),
                self.cmb_call_type.currentText(),
                self.cmb_type_count.currentText(),
                self.txt_order.text(),
                self.txt_family.text(),
                self.txt_scientific_name.text(),
                self.cmb_turkish_name.currentText(),
                self.txt_background_order.text(),
                self.txt_background_family.text(),
                self.txt_background_scientific_name.text(),
                self.cmb_background_turkish_name.currentText(),
                self.cmb_background_noise.currentText(),
                self.txt_other_species.toPlainText(),
                self.cmb_location.currentText(),
                self.cmb_recordist.currentText(),
                self.cmb_ornithologist.currentText(),
                self.cmb_verification_status.currentText(),
                self.spn_confidence.value(),
                self.cmb_microphone.currentText(),
                self.cmb_recorder.currentText(),
                habitat_ID,
                habitat_tipi,
                kod,
                utm_zone,
                utm_easting,
                utm_northing,
                lat,
                lon,
                google_maps,
                self.txt_notes.toPlainText(),
                self.config_data.get("project", "")
            ])

        return segment_filename

    def view_csv(self):
        try:
            if os.path.exists(self.csv_file):
                if sys.platform.startswith("win"):
                    os.startfile(self.csv_file)
                elif sys.platform == "darwin":
                    os.system(f"open '{self.csv_file}'")
                else:
                    os.system(f"xdg-open '{self.csv_file}'")
            else:
                QMessageBox.information(self, "Bilgi", "CSV dosyası henüz oluşturulmadı.")
        except Exception:
            QMessageBox.information(self, "Bilgi", f"CSV dosyası: {self.csv_file}")

    def export_csv(self):
        if not os.path.exists(self.csv_file):
            QMessageBox.information(self, "Bilgi", "Henüz oluşturulmuş bir CSV dosyası yok.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "CSV Dosyasını Kaydet", "", "CSV Dosyaları (*.csv)")
        if file_path:
            try:
                import shutil
                shutil.copy2(self.csv_file, file_path)
                QMessageBox.information(self, "Başarılı", f"CSV dosyası {file_path} olarak kaydedildi.")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"CSV dosyası kaydedilemedi: {str(e)}")

    def show_dataset_summary(self):
        """
        `bird_sound_dataset.csv` dosyasının güncel durumunu (toplam kayıt,
        tür/bölge/habitat/doğrulama dağılımı) grafiklerle özetleyen bir pencere
        açar. Tür zenginliği burada DEĞİL, show_species_richness içindedir.
        """
        csv_full_path = os.path.join(csv_project_root(), self.csv_file) \
            if not os.path.isabs(self.csv_file) else self.csv_file

        df = dataset_summary.load_dataset(csv_full_path)
        if df is None:
            QMessageBox.information(
                self, "Veri Seti Özeti",
                "Henüz veri seti (CSV) bulunamadı ya da dosya boş.\n\n"
                "En az bir segmenti 'Veri Setine Ekle' ile kaydedin ve tekrar deneyin.",
            )
            return

        summary = dataset_summary.compute_summary(df, config=coverage_report.load_project_config())
        self._dataset_summary_dialog = DatasetSummaryDialog(summary, parent=self)
        self._dataset_summary_dialog.show()

    def show_species_richness(self):
        """
        Veri Seti Özeti'nden AYRI olarak, habitat ve bölge bazında tür
        zenginliğini (her grupta kaydedilen FARKLI tür sayısı) gösteren bir
        pencere açar.
        """
        csv_full_path = os.path.join(csv_project_root(), self.csv_file) \
            if not os.path.isabs(self.csv_file) else self.csv_file

        df = dataset_summary.load_dataset(csv_full_path)
        if df is None:
            QMessageBox.information(
                self, "Tür Zenginliği Belirleme",
                "Henüz veri seti (CSV) bulunamadı ya da dosya boş.\n\n"
                "En az bir segmenti 'Veri Setine Ekle' ile kaydedin ve tekrar deneyin.",
            )
            return

        summary = dataset_summary.compute_summary(df, config=coverage_report.load_project_config())
        self._species_richness_dialog = SpeciesRichnessDialog(summary, parent=self)
        self._species_richness_dialog.show()

    def _compute_classification_diagnosis(self):
        csv_full_path = os.path.join(csv_project_root(), self.csv_file) \
            if not os.path.isabs(self.csv_file) else self.csv_file
        df = dataset_summary.load_dataset(csv_full_path)
        if df is None:
            QMessageBox.information(
                self, "Başarı Sınırı Analizi",
                "Henüz veri seti (CSV) bulunamadı ya da dosya boş.\n\n"
                "En az bir segmenti 'Veri Setine Ekle' ile kaydedin ve tekrar deneyin.",
            )
            return None
        folder = csv_project_root()
        try:
            stored = model_registry.stored_results(folder)
        except Exception:  # noqa: BLE001
            stored = []
        kw = classification_diagnosis.from_stored(stored)
        return classification_diagnosis.diagnose(
            df, single_split=classification_diagnosis.load_single_split(folder),
            times=classification_diagnosis.file_times(folder),
            data_time=os.path.getmtime(csv_full_path) if os.path.exists(csv_full_path) else None,
            fp_now=classification_diagnosis.data_fingerprint(csv_full_path), **kw)

    def show_csv_viewer(self):
        """Veri setine (CSV) kaydedilenleri uygulama içinde tablo olarak gösterir."""
        from .csv_viewer import CsvViewerDialog
        self._csv_viewer_dialog = CsvViewerDialog(self.csv_file, open_external=self.view_csv, parent=self)
        self._csv_viewer_dialog.show()

    def show_data_quality(self):
        """Salt-okunur veri kalitesi denetimi (yinelenen/örtüşen segment, boş alan, tutarlılık, dosya)."""
        from .data_quality_dialog import DataQualityDialog
        root = csv_project_root()
        csv_full = self.csv_file if os.path.isabs(self.csv_file) else os.path.join(root, self.csv_file)
        # Segment dosyaları geliştirme ortamında src/segments/ altına yazılır (save_segment); paketli sürümde
        # .exe'nin yanındaki segments/ kullanılır. Hepsi taranır, yinelenenler tekilleştirilir.
        dirs = list(dict.fromkeys(os.path.abspath(d) for d in (
            os.path.join(src_root_dir(), "segments"), os.path.join(root, "segments"),
            os.path.join(os.path.dirname(csv_full), "segments"))))
        self._data_quality_dialog = DataQualityDialog(csv_full, search_dirs=dirs, parent=self)
        self._data_quality_dialog.show()

    def show_catalog_manager(self):
        """Tür veri tabanı, habitat noktaları ve açılır listeler için ekle/güncelle/sil/listele penceresi."""
        from .catalog_dialog import CatalogDialog
        root = csv_project_root()
        csv_full = self.csv_file if os.path.isabs(self.csv_file) else os.path.join(root, self.csv_file)
        self._catalog_dialog = CatalogDialog(csv_path=csv_full, parent=self)
        self._catalog_dialog.saved.connect(self.reload_catalog_data)
        self._catalog_dialog.show()

    def show_backup_restore(self):
        """Tanım dosyalarının yedeklerini listeler; seçilen yedeği geri yükler ve formu yeniler."""
        from .prep_panel import BackupRestoreDialog
        dlg = BackupRestoreDialog(src_root_dir(), parent=self)
        dlg.restored.connect(lambda _name: self.reload_catalog_data())
        self._backup_dialog = dlg
        dlg.exec_()

    def apply_label_defaults(self):
        """Ön Hazırlık'ta kaydedilen varsayılan Kaydeden/Ornitolog/Mikrofon/Kayıt cihazı değerlerini forma yazar."""
        from ..data import on_hazirlik
        vals = on_hazirlik.load_label_defaults(src_root_dir())
        for field, attr in (("Recordist", "cmb_recordist"), ("Ornithologist", "cmb_ornithologist"),
                            ("Microphone", "cmb_microphone"), ("Recorder", "cmb_recorder")):
            cmb = getattr(self, attr, None)
            if cmb is not None and vals.get(field):
                cmb.setEditText(vals[field])

    def reload_catalog_data(self):
        """JSON tanım dosyalarını diskten yeniden okur ve etiketleme formundaki açılır listeleri yeniler.
        Seçili/yazılı değerler korunur."""
        from ..data.config import load_json_data
        self.habitat_data = load_json_data("habitat_data.json", [])
        self.bird_species_data = load_json_data("bird_species_data.json", [])
        self.config_data = load_config()

        def refill(cmb, items, blank=False):
            if cmb is None:
                return
            cur = cmb.currentText()
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItems(([""] if blank else []) + list(items))
            cmb.blockSignals(False)
            idx = cmb.findText(cur)
            if idx >= 0:
                cmb.setCurrentIndex(idx)
            elif cmb.isEditable():
                cmb.setEditText(cur)

        cfg = self.config_data.get
        names = sorted({b.get("turkish_name", "") for b in self.bird_species_data if b.get("turkish_name")})
        for attr, key, blank in (("cmb_call_type", "call_types", False),
                                 ("cmb_background_noise", "background_noise", True),
                                 ("cmb_recordist", "recordists", False),
                                 ("cmb_ornithologist", "ornithologists", False),
                                 ("cmb_verification_status", "verification_statuses", False),
                                 ("cmb_microphone", "microphones", False),
                                 ("cmb_recorder", "recorders", False)):
            refill(getattr(self, attr, None), cfg(key, []), blank)
        refill(getattr(self, "cmb_turkish_name", None), names, True)
        refill(getattr(self, "cmb_background_turkish_name", None), names, True)
        if getattr(self, "cmb_turkish_name", None) is not None:
            self.on_turkish_name_changed(self.cmb_turkish_name.currentText())
        if getattr(self, "cmb_background_turkish_name", None) is not None:
            self.on_background_turkish_name_changed(self.cmb_background_turkish_name.currentText())
        locs = sorted({h.get("konum", "") for h in self.habitat_data if h.get("konum", "")})
        refill(getattr(self, "cmb_location", None), locs, True)
        cmb_h = getattr(self, "cmb_habitat_id", None)
        if cmb_h is not None:
            cur = cmb_h.currentText()
            self.on_konum_changed(self.cmb_location.currentText())
            i = cmb_h.findText(cur)
            if i >= 0:
                cmb_h.setCurrentIndex(i)
        if getattr(self, "txt_project", None) is not None and cfg("project"):
            self.txt_project.setText(cfg("project"))
        if getattr(self, "prep_panel", None) is not None:
            self.prep_panel.refresh_lists()

    def show_classification_diagnosis(self):
        """Sınıflandırmanın sınırının model mi veri mi olduğunu gösteren pencere."""
        try:
            diagnosis = self._compute_classification_diagnosis()
            if diagnosis is None:
                return
            self._classification_diagnosis_dialog = ClassificationDiagnosisDialog(diagnosis, parent=self)
            self._classification_diagnosis_dialog.show()
        except Exception as exc:  # pencere açılamazsa program kapanmasın; hatayı göster
            tb = traceback.format_exc()
            box = QMessageBox(QMessageBox.Warning, "Başarı Sınırı Analizi",
                              f"Başarı Sınırı Analizi penceresi açılamadı:\n{type(exc).__name__}: {exc}", parent=self)
            box.setDetailedText(tb)
            box.exec_()

    def _compute_habitat_info(self) -> dict:
        """habitat_data.json ile CSV'yi birleştirip Habitat Bilgileri verisini üretir (CSV yoksa yalnızca envanter)."""
        csv_full_path = os.path.join(csv_project_root(), self.csv_file) \
            if not os.path.isabs(self.csv_file) else self.csv_file
        df = dataset_summary.load_dataset(csv_full_path)
        return habitat_info.build(self.habitat_data, df)

    def show_habitat_info(self):
        """Tanımlı tüm noktaların (habitat_data.json) envanterini ve etiketleme kapsamını gösterir."""
        result = self._compute_habitat_info()
        self._habitat_info_dialog = HabitatInfoDialog(result, parent=self)
        self._habitat_info_dialog.show()

    def show_location_map(self):
        """
        `Lat`/`Lon` sütunlarındaki kayıt noktalarını gösteren harita/koordinat
        serpme grafiği penceresini açar.
        """
        csv_full_path = os.path.join(csv_project_root(), self.csv_file) \
            if not os.path.isabs(self.csv_file) else self.csv_file

        df = dataset_summary.load_dataset(csv_full_path)
        if df is None:
            QMessageBox.information(
                self, "Konum Haritası",
                "Henüz veri seti (CSV) bulunamadı ya da dosya boş.\n\n"
                "En az bir segmenti 'Veri Setine Ekle' ile kaydedin ve tekrar deneyin.",
            )
            return

        points = location_map.extract_location_points(df)
        self._location_map_dialog = LocationMapDialog(points, parent=self)
        self._location_map_dialog.show()

    def show_classification_folders(self):
        """
        `segments/` klasöründeki ses dosyalarını Scientific_Name'e göre
        `segments/classification/<tür>/` altına kopyalayan pencereyi açar.
        """
        self._classification_folders_dialog = ClassificationFoldersDialog(parent=self)
        self._classification_folders_dialog.show()

    def show_classifier_training(self, model_id=model_registry.DEFAULT_MODEL_ID):
        """
        `segments/classification/` içindeki veriyle seçilen modelin gömmeleri üzerinde
        özel bir MLP sınıflandırıcı eğiten pencereyi açar. Her model kendi penceresinde ve kendi
        sonuç klasöründe çalışır.
        """
        if not isinstance(model_id, str):   # QAction.triggered(bool) argümanı
            model_id = model_registry.DEFAULT_MODEL_ID
        dialogs = self.__dict__.setdefault("_model_dialogs", {})
        dlg = ClassifierTrainingDialog(parent=self, model_id=model_id)
        dialogs[model_id] = dlg
        if model_id == model_registry.DEFAULT_MODEL_ID:
            self._classifier_training_dialog = dlg
        dlg.show()

    def show_model_picker(self):
        """Model seçici: BirdNET / Perch 2.0 eğitim pencereleri ve Model Karşılaştırma."""
        dlg = ModelPickerDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self._open_model_choice(dlg.choice)

    def _open_model_choice(self, choice):
        if choice == ModelPickerDialog.COMPARE:
            self.show_model_comparison()
        elif choice:
            self.show_classifier_training(choice)

    def show_model_comparison(self):
        """Kayıtlı model sonuçlarını yan yana gösteren karşılaştırma penceresini açar."""
        self._model_comparison_dialog = ModelComparisonDialog(parent=self)
        self._model_comparison_dialog.show()

    def create_icon(self):
        try:
            from PyQt5.QtGui import QIcon, QPixmap, QPainter, QFont, QColor
            from PyQt5.QtCore import Qt

            pixmap = QPixmap(256, 256)
            pixmap.fill(Qt.transparent)

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)

            font = QFont()
            font.setPointSize(180)

            bird_emoji = "🐦"

            painter.setFont(font)
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, bird_emoji)

            painter.end()

            icon = QIcon(pixmap)
            return icon

        except Exception as e:
            print(f"Simge oluşturulamadı: {e}")
            return QIcon()

    def closeEvent(self, event):
        threads_to_stop = [
            ('loader_thread', self.loader_thread),
            ('proc_thread', self.proc_thread),
            ('detection_thread', self.detection_thread),
            ('spec_thread', self.spec_thread),
        ]

        for name, thread in threads_to_stop:
            if thread and thread.isRunning():
                print(f"{name} durduruluyor...")
                thread.quit()
                thread.wait(1000)
                if thread.isRunning():
                    thread.terminate()
                    thread.wait()

        temp_files = [self.temp_file, self.modified_temp_full]
        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    print(f"Geçici dosya silindi: {temp_file}")
                except Exception as e:
                    print(f"Geçici dosya silinemedi: {e}")

        if hasattr(self, 'memory_watchdog_timer'):
            self.memory_watchdog_timer.stop()

        self.cleanup_memory()

        event.accept()

    LOGO_PX = 72   # alt çubuktaki logonun mantıksal (ölçekten bağımsız) boyutu

    def _add_thesis_footer(self):
        """Alt çubuk: ortada üniversite logosu, solda ilk iki satır, sağda son iki satır (app_info.py)."""
        import html
        from .app_info import footer_lines, find_logo, THESIS_TITLE
        inst, title, person, advisors = (html.escape(t) for t in footer_lines())

        def block(html_text, align):
            lab = QLabel(html_text)
            lab.setTextFormat(Qt.RichText)
            lab.setWordWrap(True)
            lab.setAlignment(align | Qt.AlignVCenter)
            lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # uzun satır pencereyi genişletmesin
            return lab

        left = block(f"<div style='color:#6b7280; font-size:11px;'>{inst}</div>"
                     f"<div style='color:#0f4c75; font-size:13px; font-weight:700;'>{title}</div>", Qt.AlignRight)
        right = block(f"<div style='color:#4b5563; font-size:11px;'>{person}</div>"
                      f"<div style='color:#4b5563; font-size:11px;'>{advisors}</div>", Qt.AlignLeft)
        left.setToolTip(THESIS_TITLE)

        box = QWidget()
        box.setObjectName("thesisFooter")
        row = QHBoxLayout(box)
        row.setContentsMargins(8, 2, 8, 2)
        row.setSpacing(14)
        row.addWidget(left, 1)
        logo_path = find_logo([csv_project_root(), src_root_dir()])
        if logo_path:
            pm = logo_pixmap(logo_path, self.LOGO_PX, self.devicePixelRatioF())
            if pm is not None:
                logo = QLabel()
                logo.setPixmap(pm)
                logo.setFixedSize(self.LOGO_PX, self.LOGO_PX)
                row.addWidget(logo, 0, Qt.AlignCenter)
        row.addWidget(right, 1)
        box.setFixedHeight(self.LOGO_PX + 8)
        bar = self.statusBar()
        bar.setSizeGripEnabled(False)
        bar.addWidget(box, 1)

    def _update_footer_visibility(self, *_):
        tabs = getattr(self, "main_tabs", None)
        hidden = getattr(self, "_footer_hidden_tabs", None)
        if tabs is None or hidden is None:
            return
        self.statusBar().setVisible(tabs.currentWidget() not in hidden)

    def initUI(self):
        self.setWindowTitle(
            'Yapay Zeka İle Kuş Seslerinin Akustik Analizi Ve Tür Zenginliğinin Belirlenmesi')
        self.create_menu_bar()

        screen_geo = QGuiApplication.primaryScreen().availableGeometry()
        w = min(1400, screen_geo.width() - 80)
        h = min(900, screen_geo.height() - 80)
        self.resize(w, h)

        self._add_thesis_footer()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        theme.add_hero(main_layout, "BioSeg Labeling Studio v2.0",
                       "Biyoakustik Etiketleme ve Analiz, Sınıflandırma ve Tür Zenginliği Belirleme", compact=True)

        self.controls_panel = QWidget()
        left_layout = QVBoxLayout(self.controls_panel)
        self.controls_panel.setMinimumHeight(400)

        def groupbox(title: str):
            g = QGroupBox(title)
            l = QVBoxLayout()
            l.setContentsMargins(12, 22, 12, 8)
            l.setSpacing(6)
            g.setLayout(l)
            return g, l

        def fixed_button(btn: QPushButton):
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            return btn

        # --- Ses Dosyası Grubu ---
        file_group, file_layout = groupbox("🔊 Ses Dosyası")
        top_row = QHBoxLayout()
        self.btn_load = fixed_button(QPushButton("📂 Ses Dosyası Yükle (.wav)"))
        self.btn_load.clicked.connect(self.load_file)
        self.btn_load.setStyleSheet("QPushButton { font-weight: bold; padding:6px 12px; min-height:32px; }")
        top_row.addWidget(self.btn_load)
        top_row.addStretch()
        file_layout.addLayout(top_row)

        self.lbl_file_info = QLabel("Dosya Seçilmedi! Lütfen önce bir .wav dosyası yükleyin.")
        self.lbl_file_info.setWordWrap(True)
        self.lbl_file_info.setStyleSheet(
            "QLabel { background:#f0f0f0; padding:6px 10px; border-radius:4px; font-size: 15px; }")
        file_layout.addWidget(self.lbl_file_info)

        # --- Görünüm Grubu ---
        view_group, v_layout = groupbox("👁️ Görünüm")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft)
        self.cmb_view = QComboBox()
        self.cmb_view.addItems(["Dalga Formu", "Spektrogram (Doğrusal)", "Spektrogram (Mel)"])
        self.cmb_view.currentIndexChanged.connect(self.change_view)
        self.cmb_view.setMaximumWidth(200)
        form.addRow(QLabel("Görünüm:"), self.cmb_view)

        # Gösterim ayarları: yalnızca ekrandaki görüntüyü değiştirir (hesaplama ve CSV etkilenmez)
        from ..visualization import spectrogram_view as _sv
        self.cmb_cmap = QComboBox()
        for _label, _name in _sv.COLORMAPS:
            self.cmb_cmap.addItem(_label, _name)
        self.cmb_cmap.setToolTip("Spektrogramın renk haritası. Gri tonlu: siyah-beyaz baskıda okunaklı şekil için.")
        self.cmb_cmap.setMaximumWidth(240)
        form.addRow(QLabel("Renk haritası:"), self.cmb_cmap)

        self.spn_contrast = QSpinBox()
        self.spn_contrast.setRange(0, 120)
        self.spn_contrast.setSingleStep(5)
        self.spn_contrast.setSuffix(" dB")
        self.spn_contrast.setSpecialValueText("Otomatik (tam aralık)")
        self.spn_contrast.setToolTip(
            "Gösterilen dinamik aralık. Küçük değer kontrastı artırır: en güçlü %0,5 sesin altından bu kadar dB "
            "aşağısına kadar olan bölge gösterilir. Gürültülü kayıtta zayıf ötüşü ayırmak için 40-60 dB deneyin.")
        self.spn_contrast.setMaximumWidth(240)
        form.addRow(QLabel("Kontrast:"), self.spn_contrast)

        freq_row = QHBoxLayout()
        self.spn_fmin = QSpinBox()
        self.spn_fmin.setRange(0, 24000)
        self.spn_fmin.setSingleStep(100)
        self.spn_fmin.setSuffix(" Hz")
        self.spn_fmax = QSpinBox()
        self.spn_fmax.setRange(0, 48000)
        self.spn_fmax.setSingleStep(500)
        self.spn_fmax.setSuffix(" Hz")
        self.spn_fmax.setSpecialValueText("Tümü")
        self.spn_fmin.setToolTip("Gösterilecek en düşük frekans. Alçak gürültü bölgesini kırpmak için 500-1000 Hz deneyin.")
        self.spn_fmax.setToolTip("Gösterilecek en yüksek frekans. 'Tümü' = örnekleme hızının yarısına kadar.")
        freq_row.addWidget(self.spn_fmin)
        freq_row.addWidget(QLabel("–"))
        freq_row.addWidget(self.spn_fmax)
        freq_row.addStretch()
        form.addRow(QLabel("Frekans aralığı:"), freq_row)

        self.chk_wave_strip = QCheckBox("Üstte dalga formunu da göster")
        self.chk_wave_strip.setToolTip("Spektrogramın üstüne ortak zaman eksenli bir dalga formu şeridi ekler. "
                                       "Seçim ve oynatma konumu şeritte de görünür; seçimi alttaki spektrogramda yapın.")
        form.addRow(QLabel(""), self.chk_wave_strip)
        v_layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_view_reset = QPushButton("↺ Görünüm Ayarlarını Sıfırla")
        self.btn_view_save = QPushButton("🖼 Görünümü PNG Kaydet (300 dpi)")
        self.btn_view_save.setToolTip("Grafiği olduğu gibi (seçim ve oynatma çizgisi olmadan) 300 dpi PNG olarak kaydeder.")
        btn_row.addWidget(self.btn_view_reset)
        btn_row.addWidget(self.btn_view_save)
        btn_row.addStretch()
        v_layout.addLayout(btn_row)
        v_layout.addStretch()

        self.cmb_cmap.currentIndexChanged.connect(self.on_view_settings_changed)
        self.spn_contrast.valueChanged.connect(self.on_view_settings_changed)
        self.spn_fmin.valueChanged.connect(self.on_view_settings_changed)
        self.spn_fmax.valueChanged.connect(self.on_view_settings_changed)
        self.chk_wave_strip.toggled.connect(self.on_view_settings_changed)
        for _w in (self.cmb_cmap, self.spn_contrast, self.spn_fmin, self.spn_fmax, self.chk_wave_strip):
            _w.setEnabled(False)   # açılışta dalga formu görünümü
        self.btn_view_reset.clicked.connect(self.reset_view_settings)
        self.btn_view_save.clicked.connect(self.save_view_png)

        # --- Oynatma ve Gürültü Azaltma Grubu ---
        playproc_group = QWidget()
        pp_layout = QVBoxLayout(playproc_group)

        control_box = QGroupBox("🎵 Tüm Sesi Oynatma")
        control_layout = QVBoxLayout(control_box)

        progress_col = QVBoxLayout()
        self.sld_progress = QSlider(Qt.Horizontal)
        self.sld_progress.setRange(0, 1000)
        self.sld_progress.sliderMoved.connect(self.seek_audio)
        self.sld_progress.setMinimumHeight(16)
        progress_col.addWidget(self.sld_progress)

        time_row = QHBoxLayout()
        self.lbl_current_time = QLabel("00:00")
        self.lbl_current_time.setStyleSheet("font-weight:bold;")
        self.lbl_total_time = QLabel("00:00")
        time_row.addWidget(self.lbl_current_time)
        time_row.addWidget(QLabel(" / "))
        time_row.addWidget(self.lbl_total_time)
        time_row.addStretch()
        progress_col.addLayout(time_row)
        control_layout.addLayout(progress_col)

        transport = QHBoxLayout()
        transport.setSpacing(4)
        control_btn_style = "QPushButton { font-size:16px; min-width:40px; min-height:32px; padding:4px 10px; }"

        self.btn_rewind = fixed_button(QPushButton("⏪"))
        self.btn_rewind.clicked.connect(self.rewind)
        self.btn_rewind.setEnabled(False)
        self.btn_rewind.setStyleSheet(control_btn_style)

        self.btn_play = fixed_button(QPushButton("▶"))
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        self.btn_play.setStyleSheet(control_btn_style)

        self.btn_stop = fixed_button(QPushButton("⏹"))
        self.btn_stop.clicked.connect(self.stop_audio)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(control_btn_style)

        self.btn_forward = fixed_button(QPushButton("⏩"))
        self.btn_forward.clicked.connect(self.forward)
        self.btn_forward.setEnabled(False)
        self.btn_forward.setStyleSheet(control_btn_style)

        for b in (self.btn_rewind, self.btn_play, self.btn_stop, self.btn_forward):
            transport.addWidget(b)
        transport.addStretch()
        control_layout.addLayout(transport)

        speed_row = QHBoxLayout()
        lbl_speed = QLabel("Oynatma Hızı:")
        self.sld_speed = QSlider(Qt.Horizontal)
        self.sld_speed.setRange(5, 20)
        self.sld_speed.setValue(10)
        self.sld_speed.valueChanged.connect(self.change_speed)
        self.sld_speed.setMinimumHeight(16)
        self.lbl_speed = QLabel("1.0x")
        self.lbl_speed.setStyleSheet("font-weight:bold;")
        speed_row.addWidget(lbl_speed)
        speed_row.addWidget(self.sld_speed, 1)
        speed_row.addWidget(self.lbl_speed)
        control_layout.addLayout(speed_row)

        pp_layout.addWidget(control_box)

        denoise_box = QGroupBox("🧹 Gürültü Azaltma / Normalizasyon")
        dg_layout = QVBoxLayout(denoise_box)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Kaynak:"))
        self.cmb_proc_source = QComboBox()
        self.cmb_proc_source.addItems(["Tüm kayıt", "Yalnızca seçili aralık"])
        src_row.addWidget(self.cmb_proc_source)
        src_row.addStretch()
        dg_layout.addLayout(src_row)

        opts = QGroupBox("İşlemler")
        opts_l = QHBoxLayout(opts)
        self.chk_denoise = QCheckBox("🧹 Gürültü Azaltma")
        self.chk_denoise.setChecked(True)
        self.chk_denoise.setToolTip("Kayıttaki arka plan gürültüsünü azaltır.")
        self.chk_lufs = QCheckBox("🔊 LUFS Normalizasyonu")
        self.chk_lufs.setChecked(True)
        self.chk_lufs.setToolTip("Ses seviyesini standart bir ses şiddetine normalleştirir.")
        opts_l.addWidget(self.chk_denoise)
        opts_l.addWidget(self.chk_lufs)
        opts_l.addStretch()
        dg_layout.addWidget(opts)

        btn_row = QHBoxLayout()
        self.btn_run_process = QPushButton("▶ İşlemleri Çalıştır")
        self.btn_run_process.clicked.connect(self.run_processing_pipeline)
        self.btn_save_full = QPushButton("💾 Geçerli Kaydı WAV Olarak Kaydet")
        self.btn_save_full.setEnabled(False)
        btn_row.addWidget(self.btn_run_process)
        btn_row.addWidget(self.btn_save_full)
        btn_row.addStretch()
        dg_layout.addLayout(btn_row)

        self.lbl_proc_status = QLabel("Hazır")
        self.lbl_proc_status.setStyleSheet("QLabel{background:#eef; padding:6px; border-radius:4px;}")
        dg_layout.addWidget(self.lbl_proc_status)

        pp_layout.addWidget(denoise_box)

        # --- Yakınlaştırma Grubu ---
        def group_zoom():
            g, z_layout = groupbox("🔍 Yakınlaştırma ve Kaydırma")
            hz_top = QHBoxLayout()
            hz_top.setSpacing(4)
            self.btn_zoom_in = fixed_button(QPushButton("+"))
            self.btn_zoom_in.clicked.connect(lambda: self.zoom_buttons(1 / 1.2))
            self.btn_zoom_out = fixed_button(QPushButton("-"))
            self.btn_zoom_out.clicked.connect(lambda: self.zoom_buttons(1.2))
            self.btn_zoom_reset = fixed_button(QPushButton("Yakınlaştırmayı Sıfırla"))
            self.btn_zoom_reset.clicked.connect(self.reset_zoom)

            for b in (self.btn_zoom_in, self.btn_zoom_out, self.btn_zoom_reset):
                b.setStyleSheet("QPushButton { min-height:32px; padding:4px 8px; }")

            hz_top.addWidget(self.btn_zoom_in)
            hz_top.addWidget(self.btn_zoom_out)
            hz_top.addWidget(self.btn_zoom_reset)
            hz_top.addStretch()
            z_layout.addLayout(hz_top)

            self.pan_slider = QSlider(Qt.Horizontal)
            self.pan_slider.setRange(0, 1000)
            self.pan_slider.setEnabled(False)
            self.pan_slider.valueChanged.connect(self.on_pan_slider)
            self.pan_slider.setMinimumHeight(16)
            z_layout.addWidget(QLabel("Yatay Kaydırma"))
            z_layout.addWidget(self.pan_slider)

            pan_hint = QLabel(
                "Sağ tık + sürükle: kaydır | Sağ tık: menü | Fare tekerleği: yakınlaştır | Shift+tekerlek: kaydır")
            pan_hint.setWordWrap(True)
            pan_hint.setStyleSheet("color:#666")
            z_layout.addWidget(pan_hint)
            return g

        zoom_group = group_zoom()

        # --- Aralık Seçimi Grubu ---
        selection_group, selection_layout = groupbox("⏰ Aralık Seçimi ve Oynatma")
        sel_form = QFormLayout()
        sel_form.setLabelAlignment(Qt.AlignLeft)
        sel_form.setFormAlignment(Qt.AlignLeft)

        self.sld_start = QSlider(Qt.Horizontal)
        self.sld_start.setRange(0, 0)
        self.sld_start.valueChanged.connect(self.update_time_from_slider)
        self.sld_start.sliderMoved.connect(self.update_time_from_slider)
        self.sld_start.setMinimumHeight(18)

        self.sld_end = QSlider(Qt.Horizontal)
        self.sld_end.setRange(0, 0)
        self.sld_end.valueChanged.connect(self.update_time_from_slider)
        self.sld_end.sliderMoved.connect(self.update_time_from_slider)
        self.sld_end.setMinimumHeight(18)

        sel_form.addRow(QLabel("Başlangıç:"), self.sld_start)
        sel_form.addRow(QLabel("Bitiş:"), self.sld_end)
        selection_layout.addLayout(sel_form)

        info_row = QHBoxLayout()
        self.lbl_start = QLabel("Başlangıç: 0.000 sn")
        self.lbl_end = QLabel("Bitiş: 0.000 sn")
        for lbl in (self.lbl_start, self.lbl_end):
            lbl.setStyleSheet("font-weight:bold;")
        info_row.addWidget(self.lbl_start)
        info_row.addStretch()
        info_row.addWidget(self.lbl_end)
        selection_layout.addLayout(info_row)

        selection_btn_layout = QHBoxLayout()
        selection_btn_layout.setSpacing(6)
        action_btn_style = "QPushButton { font-size:16px; padding:8px 14px; min-height:32px; font-weight:bold; color:white; }"

        self.btn_play_selection = fixed_button(QPushButton("▶ Oynat"))
        self.btn_play_selection.clicked.connect(self.toggle_selection_play)
        self.btn_play_selection.setEnabled(False)
        self.btn_play_selection.setStyleSheet(action_btn_style + "QPushButton { background:#4CAF50; }")

        self.btn_loop_selection = fixed_button(QPushButton("🔁 Döngü"))
        self.btn_loop_selection.clicked.connect(self.loop_selection)
        self.btn_loop_selection.setEnabled(False)
        self.btn_loop_selection.setStyleSheet(action_btn_style + "QPushButton { background:#2196F3; }")

        self.btn_stop_selection = fixed_button(QPushButton("⏹ Durdur"))
        self.btn_stop_selection.clicked.connect(self.stop_selection)
        self.btn_stop_selection.setEnabled(False)
        self.btn_stop_selection.setStyleSheet(action_btn_style + "QPushButton { background:#f44336; }")

        selection_btn_layout.addWidget(self.btn_play_selection)
        selection_btn_layout.addWidget(self.btn_loop_selection)
        selection_btn_layout.addWidget(self.btn_stop_selection)
        selection_btn_layout.addStretch()
        selection_layout.addLayout(selection_btn_layout)

        self.btn_clear = fixed_button(QPushButton("🗑️ Seçimi Temizle"))
        self.btn_clear.clicked.connect(self.clear_selection)
        self.btn_clear.setEnabled(False)
        self.btn_clear.setStyleSheet("QPushButton{min-height:32px;}")
        selection_layout.addWidget(self.btn_clear, alignment=Qt.AlignLeft)

        self.lbl_selection_duration = QLabel("Seçim yok")
        self.lbl_selection_duration.setWordWrap(True)
        selection_layout.addWidget(self.lbl_selection_duration)

        # --- Tespit Grubu ---
        detection_group, detection_layout = groupbox("🔍 BirdNET Destekli Kuş Ötüşü Tespiti")

        # Ön işleme (Gürültü Azaltma/LUFS) kutuları: "Oynatma ve İşleme" sekmesindekiyle
        # AYNI işlevi sunar; böylece kullanıcı "Kuş Ötüşlerini Tespit Et" öncesinde
        # gürültü azaltmayı/normalizasyonu UNUTMAZ. Bu iki kutu o sekmedekilerden
        # BAĞIMSIZDIR (otomatik eşitlenmezler), ancak ikisi de varsayılan olarak
        # işaretlidir; kullanıcı hiçbir şey yapmasa da ön işleme otomatik uygulanır.
        preproc_row = QHBoxLayout()
        preproc_row.addWidget(QLabel("<b>Tespit öncesi ön işleme:</b>"))
        self.chk_denoise_before_detect = QCheckBox("🧹 Gürültü Azaltma")
        self.chk_denoise_before_detect.setChecked(True)
        self.chk_denoise_before_detect.setToolTip(
            "Tespitten önce TÜM kayda gürültü azaltma (spektral kapılama) uygulanır. "
            "'Oynatma ve İşleme' sekmesindeki 'Gürültü Azaltma' kutusundan bağımsızdır."
        )
        self.chk_lufs_before_detect = QCheckBox("🔊 LUFS Normalizasyonu")
        self.chk_lufs_before_detect.setChecked(True)
        self.chk_lufs_before_detect.setToolTip(
            "Tespitten önce TÜM kayda LUFS ses şiddeti normalizasyonu uygulanır. "
            "'Oynatma ve İşleme' sekmesindeki 'LUFS Normalizasyonu' kutusundan bağımsızdır."
        )
        preproc_row.addWidget(self.chk_denoise_before_detect)
        preproc_row.addWidget(self.chk_lufs_before_detect)
        preproc_row.addStretch()
        detection_layout.addLayout(preproc_row)

        # --- BirdNET ayarları: TEK SATIR (yer kazanmak için) ---
        birdnet_row = QHBoxLayout()
        birdnet_row.addWidget(QLabel("<b>BirdNET:</b>"))

        birdnet_row.addWidget(QLabel("Min. güven:"))
        self.spn_birdnet_conf = QDoubleSpinBox()
        self.spn_birdnet_conf.setRange(0.01, 0.99)
        self.spn_birdnet_conf.setSingleStep(0.05)
        self.spn_birdnet_conf.setValue(0.10)
        self.spn_birdnet_conf.setMaximumWidth(75)
        self.spn_birdnet_conf.valueChanged.connect(self.update_detection_params)
        birdnet_row.addWidget(self.spn_birdnet_conf)

        birdnet_row.addWidget(QLabel("Enlem:"))
        self.txt_birdnet_lat = QLineEdit()
        self.txt_birdnet_lat.setPlaceholderText("isteğe bağlı")
        self.txt_birdnet_lat.setMaximumWidth(90)
        birdnet_row.addWidget(self.txt_birdnet_lat)

        birdnet_row.addWidget(QLabel("Boylam:"))
        self.txt_birdnet_lon = QLineEdit()
        self.txt_birdnet_lon.setPlaceholderText("isteğe bağlı")
        self.txt_birdnet_lon.setMaximumWidth(90)
        birdnet_row.addWidget(self.txt_birdnet_lon)

        birdnet_row.addStretch()
        detection_layout.addLayout(birdnet_row)

        detection_hint = QLabel(
            "ℹ️ BirdNET, kendi iç parçalama mantığıyla TÜM kaydı tarar. "
            "'Tespit öncesi ön işleme' kutuları açıksa, gürültü azaltma/"
            "ses seviyesi normalizasyonu taramadan hemen önce tüm kayda uygulanır; "
            "bu işlem dalga formunu ve kayıttaki sesi de kalıcı olarak günceller."
        )
        detection_hint.setWordWrap(True)
        detection_hint.setStyleSheet("QLabel { color: #555; font-size: 13px; }")
        detection_layout.addWidget(detection_hint)

        detection_btn_layout = QHBoxLayout()
        self.btn_detect_bird_calls = fixed_button(QPushButton("🔎 Kuş Ötüşlerini Tespit Et"))
        self.btn_detect_bird_calls.clicked.connect(self.detect_bird_calls)
        self.btn_detect_bird_calls.setEnabled(False)
        self.btn_detect_bird_calls.setStyleSheet(
            "QPushButton { background:#9C27B0; color:white; font-weight:bold; min-height:32px; }")

        self.btn_clear_detection = fixed_button(QPushButton("🗑️ Tespiti Temizle"))
        self.btn_clear_detection.clicked.connect(self.clear_bird_call_detection)
        self.btn_clear_detection.setEnabled(False)
        self.btn_clear_detection.setStyleSheet("QPushButton{min-height:32px;}")

        detection_btn_layout.addWidget(self.btn_detect_bird_calls)
        detection_btn_layout.addWidget(self.btn_clear_detection)
        detection_btn_layout.addStretch()
        detection_layout.addLayout(detection_btn_layout)

        self.lbl_detection_status = QLabel("Henüz tespit yapılmadı")
        self.lbl_detection_status.setWordWrap(True)
        self.lbl_detection_status.setStyleSheet("QLabel { background:#f0f0f0; padding:6px 10px; border-radius:4px; }")
        detection_layout.addWidget(self.lbl_detection_status)

        detection_layout.addWidget(QLabel("Tespit Edilen Kuş Ötüşleri:"))
        self.detected_bird_calls_table = QTableWidget()

        detection_columns = [
            "Seg. ID", "Başlangıç (sn)", "Bitiş (sn)", "Süre (sn)",
            "BirdNET — Latince", "BirdNET — Türkçe", "BirdNET Güven (%)",
        ]
        self.detected_bird_calls_table.setColumnCount(len(detection_columns))
        self.detected_bird_calls_table.setHorizontalHeaderLabels(detection_columns)

        self.detected_bird_calls_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.detected_bird_calls_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.detected_bird_calls_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.detected_bird_calls_table.setSortingEnabled(True)
        self.detected_bird_calls_table.cellClicked.connect(self.on_bird_call_item_clicked)
        self.detected_bird_calls_table.setMaximumHeight(260)
        detection_layout.addWidget(self.detected_bird_calls_table)

        # --- Veri Seti İşlemleri Grubu ---
        action_group, action_layout = groupbox("🏷️ Veri Seti İşlemleri")
        tabs = QTabWidget()

        # Tür ve Davranış Sekmesi
        species_tab = QWidget()
        species_layout = QVBoxLayout(species_tab)
        call_and_type_layout = QHBoxLayout()

        call_and_type_layout.addWidget(QLabel("Çağrı Türü:"))
        self.cmb_call_type = QComboBox()
        self.cmb_call_type.addItems(self.config_data.get("call_types", []))
        self.cmb_call_type.setMaximumWidth(200)
        call_and_type_layout.addWidget(self.cmb_call_type)

        call_and_type_layout.addSpacing(30)

        call_and_type_layout.addWidget(QLabel("Tür Sayısı:"))
        self.cmb_type_count = QComboBox()
        for _etiket, _deger in (("Tek", "Single"), ("Çoklu", "Multiple"), ("Bilinmiyor", "Unknown")): self.cmb_type_count.addItem(_etiket, _deger)
        self.cmb_type_count.setMaximumWidth(200)
        self.cmb_type_count.currentIndexChanged.connect(self.update_background_widgets_state)
        call_and_type_layout.addWidget(self.cmb_type_count)

        call_and_type_layout.addStretch()
        species_layout.addLayout(call_and_type_layout)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #cccccc;")
        species_layout.addWidget(line1)

        primary_title = QLabel("Birincil Tür")
        primary_title.setStyleSheet("font-weight: bold;")
        species_layout.addWidget(primary_title)

        primary_layout = QHBoxLayout()
        primary_layout.addWidget(QLabel("Türkçe Ad:"))
        self.cmb_turkish_name = QComboBox()
        turkish_names = sorted(list(set([bird["turkish_name"] for bird in self.bird_species_data])))
        self.cmb_turkish_name.addItems([""] + turkish_names)
        self.cmb_turkish_name.currentTextChanged.connect(self.on_turkish_name_changed)
        self.cmb_turkish_name.setMaximumWidth(200)
        self.cmb_turkish_name.setEditable(True)
        primary_layout.addWidget(self.cmb_turkish_name)

        primary_layout.addWidget(QLabel("Bilimsel Ad:"))
        self.txt_scientific_name = QLineEdit()
        self.txt_scientific_name.setReadOnly(True)
        self.txt_scientific_name.setMaximumWidth(200)
        primary_layout.addWidget(self.txt_scientific_name)

        primary_layout.addWidget(QLabel("Familya:"))
        self.txt_family = QLineEdit()
        self.txt_family.setReadOnly(True)
        self.txt_family.setMaximumWidth(200)
        primary_layout.addWidget(self.txt_family)

        primary_layout.addWidget(QLabel("Takım:"))
        self.txt_order = QLineEdit()
        self.txt_order.setReadOnly(True)
        self.txt_order.setMaximumWidth(200)
        primary_layout.addWidget(self.txt_order)

        primary_layout.addStretch()
        species_layout.addLayout(primary_layout)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #cccccc;")
        species_layout.addWidget(line2)

        background_title = QLabel("Arka Plan Tür Bilgileri")
        background_title.setStyleSheet("font-weight: bold;")
        species_layout.addWidget(background_title)

        background_layout = QHBoxLayout()
        background_layout.addWidget(QLabel("Arka Plan Türkçe Adı:"))
        self.cmb_background_turkish_name = QComboBox()
        self.cmb_background_turkish_name.addItems([""] + turkish_names)
        self.cmb_background_turkish_name.currentTextChanged.connect(self.on_background_turkish_name_changed)
        self.cmb_background_turkish_name.setMaximumWidth(200)
        self.cmb_background_turkish_name.setEditable(True)
        background_layout.addWidget(self.cmb_background_turkish_name)

        background_layout.addWidget(QLabel("Arka Plan Bilimsel Adı:"))
        self.txt_background_scientific_name = QLineEdit()
        self.txt_background_scientific_name.setReadOnly(True)
        self.txt_background_scientific_name.setMaximumWidth(200)
        background_layout.addWidget(self.txt_background_scientific_name)

        background_layout.addWidget(QLabel("Arka Plan Familyası:"))
        self.txt_background_family = QLineEdit()
        self.txt_background_family.setReadOnly(True)
        self.txt_background_family.setMaximumWidth(200)
        background_layout.addWidget(self.txt_background_family)

        background_layout.addWidget(QLabel("Arka Plan Takımı:"))
        self.txt_background_order = QLineEdit()
        self.txt_background_order.setReadOnly(True)
        self.txt_background_order.setMaximumWidth(400)
        background_layout.addWidget(self.txt_background_order)

        background_layout.addStretch()
        species_layout.addLayout(background_layout)

        other_species_layout = QHBoxLayout()
        other_species_layout.addWidget(QLabel("Diğer Tür Notları:"))
        self.txt_other_species = QTextEdit()
        self.txt_other_species.setFixedHeight(36)
        self.txt_other_species.setPlaceholderText("Listede olmayan türler için notlar...")
        self.txt_other_species.setMinimumWidth(200)
        other_species_layout.addWidget(self.txt_other_species)
        other_species_layout.addStretch()
        species_layout.addLayout(other_species_layout)

        line3 = QFrame()
        line3.setFrameShape(QFrame.HLine)
        line3.setStyleSheet("background-color: #cccccc;")
        species_layout.addWidget(line3)

        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Arka Plan Gürültü Türü:"))
        self.cmb_background_noise = QComboBox()
        self.cmb_background_noise.addItems([""] + self.config_data.get("background_noise", []))
        self.cmb_background_noise.setEditable(True)
        self.cmb_background_noise.setMaximumWidth(200)
        noise_layout.addWidget(self.cmb_background_noise)
        noise_layout.addStretch()
        species_layout.addLayout(noise_layout)

        species_layout.addStretch()

        # Konum ve Habitat Sekmesi
        location_tab = QWidget()
        location_layout = QVBoxLayout(location_tab)
        location_form = QFormLayout()
        location_form.setLabelAlignment(Qt.AlignLeft)
        location_form.setFormAlignment(Qt.AlignLeft)

        self.cmb_location = QComboBox()
        locations = sorted(list(set([h.get("konum", "") for h in self.habitat_data if h.get("konum", "")])))
        self.cmb_location.addItems([""] + locations)
        self.cmb_location.currentTextChanged.connect(self.on_konum_changed)
        self.cmb_location.setMaximumWidth(250)
        self.cmb_location.setEditable(True)
        location_form.addRow(QLabel("Konum:"), self.cmb_location)

        self.cmb_habitat_id = QComboBox()
        self.cmb_habitat_id.addItem("")
        for habitat in self.habitat_data:
            self.cmb_habitat_id.addItem(str(habitat["habitat_ID"]), habitat)
        self.cmb_habitat_id.currentIndexChanged.connect(self.on_habitat_id_changed)
        self.cmb_habitat_id.setMaximumWidth(200)
        location_form.addRow(QLabel("Habitat ID:"), self.cmb_habitat_id)

        self.txt_habitat = QLineEdit()
        self.txt_habitat.setReadOnly(True)
        self.txt_habitat.setPlaceholderText("Habitat tipi...")
        location_form.addRow(QLabel("Habitat Tipi:"), self.txt_habitat)

        self.txt_kod = QLineEdit()
        self.txt_kod.setReadOnly(True)
        location_form.addRow(QLabel("Kod:"), self.txt_kod)

        self.txt_utm_zone = QLineEdit()
        self.txt_utm_zone.setReadOnly(True)
        location_form.addRow(QLabel("UTM Bölgesi:"), self.txt_utm_zone)

        utm_layout = QHBoxLayout()
        self.txt_utm_easting = QLineEdit()
        self.txt_utm_easting.setReadOnly(True)
        self.txt_utm_northing = QLineEdit()
        self.txt_utm_northing.setReadOnly(True)
        utm_layout.addWidget(QLabel("Doğu:"))
        utm_layout.addWidget(self.txt_utm_easting)
        utm_layout.addWidget(QLabel("Kuzey:"))
        utm_layout.addWidget(self.txt_utm_northing)
        location_form.addRow(QLabel("UTM Koordinatları:"), utm_layout)

        lat_lon_layout = QHBoxLayout()
        self.txt_lat = QLineEdit()
        self.txt_lat.setReadOnly(True)
        self.txt_lon = QLineEdit()
        self.txt_lon.setReadOnly(True)
        lat_lon_layout.addWidget(QLabel("Enlem:"))
        lat_lon_layout.addWidget(self.txt_lat)
        lat_lon_layout.addWidget(QLabel("Boylam:"))
        lat_lon_layout.addWidget(self.txt_lon)
        location_form.addRow(QLabel("Koordinatlar:"), lat_lon_layout)

        maps_layout = QHBoxLayout()
        self.txt_google_maps = QLineEdit()
        self.txt_google_maps.setReadOnly(True)
        self.btn_open_maps = QPushButton("🌍 Haritada Aç")
        self.btn_open_maps.clicked.connect(self.open_google_maps)
        self.btn_open_maps.setEnabled(False)
        maps_layout.addWidget(self.txt_google_maps, 4)
        maps_layout.addWidget(self.btn_open_maps, 1)
        location_form.addRow(QLabel("Google Haritalar:"), maps_layout)

        location_layout.addLayout(location_form)
        location_layout.addStretch()

        # Kişiler ve Doğrulama Sekmesi
        verification_tab = QWidget()
        verification_layout = QVBoxLayout(verification_tab)
        verification_form = QFormLayout()
        verification_form.setLabelAlignment(Qt.AlignLeft)
        verification_form.setFormAlignment(Qt.AlignLeft)

        self.cmb_recordist = QComboBox()
        self.cmb_recordist.addItems(self.config_data.get("recordists", []))
        self.cmb_recordist.setEditable(True)
        verification_form.addRow(QLabel("Kaydeden:"), self.cmb_recordist)

        self.cmb_ornithologist = QComboBox()
        self.cmb_ornithologist.addItems(self.config_data.get("ornithologists", []))
        self.cmb_ornithologist.setEditable(True)
        verification_form.addRow(QLabel("Ornitolog:"), self.cmb_ornithologist)

        self.cmb_verification_status = QComboBox()
        self.cmb_verification_status.addItems(self.config_data.get("verification_statuses", []))
        verification_form.addRow(QLabel("Doğrulama Durumu:"), self.cmb_verification_status)

        self.spn_confidence = QSpinBox()
        self.spn_confidence.setRange(0, 100)
        self.spn_confidence.setValue(100)
        self.spn_confidence.setSuffix("%")
        verification_form.addRow(QLabel("Güven Düzeyi:"), self.spn_confidence)

        verification_layout.addLayout(verification_form)
        verification_layout.addStretch()

        # Ekipman Sekmesi
        equipment_tab = QWidget()
        equipment_layout = QVBoxLayout(equipment_tab)
        equipment_form = QFormLayout()
        equipment_form.setLabelAlignment(Qt.AlignLeft)
        equipment_form.setFormAlignment(Qt.AlignLeft)

        self.cmb_microphone = QComboBox()
        self.cmb_microphone.addItems(self.config_data.get("microphones", []))
        self.cmb_microphone.setEditable(True)
        equipment_form.addRow(QLabel("Mikrofon:"), self.cmb_microphone)

        self.cmb_recorder = QComboBox()
        self.cmb_recorder.addItems(self.config_data.get("recorders", []))
        self.cmb_recorder.setEditable(True)
        equipment_form.addRow(QLabel("Kayıt Cihazı:"), self.cmb_recorder)

        equipment_layout.addLayout(equipment_form)

        project_layout = QHBoxLayout()
        project_layout.addWidget(QLabel("Proje:"))
        self.txt_project = QLineEdit()
        self.txt_project.setText(self.config_data.get("project", ""))
        # Proje adı yalnızca gösterilir; değiştirmek için Veri Seti > Tanım Verileri Yöneticisi > Açılır Listeler.
        self.txt_project.setReadOnly(True)
        self.txt_project.setToolTip("Proje adı Tanım Verileri Yöneticisi'nden (Ctrl+Y) değiştirilir; "
                                    "kaydedilen her segmente bu ad yazılır.")
        project_layout.addWidget(self.txt_project, 1)
        equipment_layout.addLayout(project_layout)

        equipment_layout.addWidget(QLabel("Notlar:"))
        self.txt_notes = QTextEdit()
        self.txt_notes.setMaximumHeight(80)
        self.txt_notes.setPlaceholderText("Gözlem notları, davranışlar vb...")
        equipment_layout.addWidget(self.txt_notes)

        equipment_layout.addStretch()

        tabs.addTab(species_tab, "🐦 Tür ve Davranış")
        tabs.addTab(location_tab, "🌍 Konum ve Habitat")
        tabs.addTab(verification_tab, "👥 Kişiler ve Doğrulama")
        tabs.addTab(equipment_tab, "🎙️ Ekipman")
        tabs.tabBar().setElideMode(Qt.ElideNone)
        tabs.tabBar().setExpanding(False)
        tabs.setUsesScrollButtons(True)

        action_layout.addWidget(tabs)

        self.btn_save = fixed_button(QPushButton("💾 Veri Setine Ekle"))
        self.btn_save.clicked.connect(self.save_segment)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(
            "QPushButton { background:#FF9800; color:white; font-weight:bold; min-height:32px; }")
        action_layout.addWidget(self.btn_save, alignment=Qt.AlignLeft)

        self.lbl_segment_message = QLabel("")
        self.lbl_segment_message.setWordWrap(True)
        self.lbl_segment_message.setStyleSheet(
            "QLabel { background:#e8f5e8; padding:6px 10px; border-radius:4px; font-size: 14px; color: #2e7d32; }")
        action_layout.addWidget(self.lbl_segment_message)

        csv_row = QHBoxLayout()
        self.btn_view_csv = fixed_button(QPushButton("📊 CSV Görüntüle"))
        self.btn_view_csv.clicked.connect(self.view_csv)
        self.btn_export_csv = fixed_button(QPushButton("📤 CSV Dışa Aktar"))
        self.btn_export_csv.clicked.connect(self.export_csv)
        csv_row.addWidget(self.btn_view_csv)
        csv_row.addWidget(self.btn_export_csv)
        csv_row.addStretch()
        action_layout.addLayout(csv_row)

        # --- Analiz Grubu (ayrı sekme, kart tabanlı pano yerleşimi) ---
        analysis_group, analysis_layout = groupbox("🔬 Analiz")

        def make_card(icon: str, title: str, description: str, slot):
            """
            Tek bir pano "kartı" oluşturur: üstte simge+başlık, altında kısa bir açıklama
            bulunan, her yerine tıklanabilen büyük bir düğme. Böylece her analiz
            aracının amacı yalnızca düğme etiketine bakarak tahmin edilmeden anlaşılır.
            """
            card = QPushButton()
            card.setCursor(Qt.PointingHandCursor)
            card.clicked.connect(slot)
            card.setMinimumSize(190, 110)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            card.setStyleSheet(
                "QPushButton { text-align:left; background:#ffffff; border:1px solid #D7E0DE; "
                "border-radius:8px; padding:10px 12px; }"
                "QPushButton:hover { background:#E3F3EE; border-color:#0F8B6D; }"
                "QPushButton:pressed { background:#CFE6E2; }"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(4, 4, 4, 4)
            card_layout.setSpacing(4)

            title_lbl = QLabel(f"{icon}  {title}")
            title_lbl.setStyleSheet("font-weight:bold; font-size:16px; border:none; background:transparent;")
            title_lbl.setWordWrap(True)
            card_layout.addWidget(title_lbl)

            desc_lbl = QLabel(description)
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet("color:#666; font-size:13.5px; border:none; background:transparent;")
            card_layout.addWidget(desc_lbl)
            card_layout.addStretch()

            return card

        # Alt grup 1: Veri Seti Analizi: elinizdeki veri setini keşfetme/anlama
        dataset_analysis_box = QGroupBox("📊 Veri Seti Analizi")
        dataset_analysis_row = QHBoxLayout(dataset_analysis_box)
        dataset_analysis_row.setSpacing(10)

        self.btn_dataset_summary = make_card(
            "📈", "Veri Seti Özeti",
            "Kayıt sayılarını tür, bölge ve habitata göre gösterir."
            ,
            self.show_dataset_summary,
        )
        self.btn_habitat_info = make_card(
            "🌿", "Habitat Bilgileri",
            "Tanımlı noktaları ve habitat tiplerini, hangilerinin etiketlendiğini gösterir."
            ,
            self.show_habitat_info,
        )
        self.btn_species_richness = make_card(
            "🧮", "Tür Zenginliği Belirleme",
            "Noktalara göre tür sayısını ve tür birikim eğrisini gösterir."
            ,
            self.show_species_richness,
        )
        self.btn_location_map = make_card(
            "🗺️", "Konum Haritası",
            "Kayıt noktalarını harita üzerinde gösterir."
            ,
            self.show_location_map,
        )

        self.btn_classification_diagnosis = make_card(
            "🔬", "3. Başarı Sınırı Analizi",
            "Kayıtlı sonuçlara bakar, eğitim yapmaz. Başarıyı model mi, veri mi sınırlıyor, gösterir."
            ,
            self.show_classification_diagnosis,
        )

        self.btn_csv_viewer = make_card(
            "📋", "Veri Seti (CSV)",
            "Kayıtlı segmentleri tabloda gösterir; arama ve sıralama yapılabilir."
            ,
            self.show_csv_viewer,
        )

        self.btn_data_quality = make_card(
            "🧹", "Veri Kalitesi Denetimi",
            "Yinelenen ve eksik kayıtları bulur; yinelenenleri iki onayla siler."
            ,
            self.show_data_quality,
        )

        dataset_analysis_row.addWidget(self.btn_dataset_summary)
        dataset_analysis_row.addWidget(self.btn_habitat_info)
        dataset_analysis_row.addWidget(self.btn_species_richness)
        dataset_analysis_row.addWidget(self.btn_location_map)
        dataset_analysis_row.addStretch()

        # Alt grup: Veri Seti Yönetimi: veriyi görme, denetleme ve tanım verilerini düzenleme (menüde "Veri Seti" menüsü)
        self.btn_catalog_manager = make_card(
            "🗂", "Tanım Verileri Yöneticisi",
            "Tür listesini, habitat noktalarını ve açılır listeleri düzenler."
            ,
            self.show_catalog_manager,
        )
        dataset_manage_box = QGroupBox("🗃 Veri Seti Yönetimi")
        dataset_manage_row = QHBoxLayout(dataset_manage_box)
        dataset_manage_row.setSpacing(10)
        dataset_manage_row.addWidget(self.btn_csv_viewer)
        dataset_manage_row.addWidget(self.btn_data_quality)
        dataset_manage_row.addStretch()

        # "Ön Hazırlık" sekmesi (ilk sekme): etiketlemeye başlamadan önce tanım verilerini hazırlama
        prep_group = QGroupBox("🧰 Ön Hazırlık")
        prep_layout = QVBoxLayout(prep_group)
        prep_note = QLabel("Etiketlemeye başlamadan önce gerekli ayarları buradan yapın.")
        prep_note.setWordWrap(True)
        prep_note.setStyleSheet("color:#555;")
        prep_layout.addWidget(prep_note)
        self.btn_restore_backup = make_card(
            "♻", "Yedekten Geri Yükle",
            "Tanım dosyalarını önceki bir yedeğe döndürür.",
            self.show_backup_restore,
        )
        from .prep_panel import PrepPanel
        _root = csv_project_root()
        _csv = self.csv_file if os.path.isabs(self.csv_file) else os.path.join(_root, self.csv_file)
        _segs = list(dict.fromkeys(os.path.abspath(d) for d in (
            os.path.join(src_root_dir(), "segments"), os.path.join(_root, "segments"),
            os.path.join(os.path.dirname(_csv), "segments"))))
        self.prep_panel = PrepPanel(self, src_root_dir(), _csv, _segs,
                                    cards=(self.btn_catalog_manager, self.btn_restore_backup))
        for _card in (self.btn_catalog_manager, self.btn_restore_backup):
            _card.setMinimumHeight(64)
            _card.setMaximumHeight(70)
        _scroll = QScrollArea()
        _scroll.setWidgetResizable(True)
        _scroll.setFrameShape(QFrame.NoFrame)
        _scroll.setWidget(self.prep_panel)
        prep_layout.addWidget(_scroll, 1)
        self.apply_label_defaults()
        self._prep_group = prep_group

        # Alt grup 2: Tür Sınıflandırma: ardışık iş akışı (önce klasörleri oluştur, sonra eğit)
        classifier_box = QGroupBox("🧠 Tür Sınıflandırma")
        classifier_row = QHBoxLayout(classifier_box)
        classifier_row.setSpacing(10)

        self.btn_classification_folders = make_card(
            "🗂️", "1. Sınıflandırma Klasörleri",
            "Etiketli segmentleri tür klasörlerine ayırır. Önce bunu yapın."
            ,
            self.show_classification_folders,
        )
        self.btn_train_classifier = make_card(
            "🎓", "2. Tür Sınıflandırma Eğitimi",
            "Model seçin (BirdNET 2.4, Perch 2.0, BirdNET + Perch); başarıyı çapraz doğrulamayla ölçer."
            ,
            self.show_model_picker,
        )

        classifier_row.addWidget(self.btn_classification_folders)
        classifier_row.addWidget(self.btn_train_classifier)
        classifier_row.addWidget(self.btn_classification_diagnosis)
        classifier_row.addStretch()

        analysis_layout.addWidget(dataset_manage_box)
        analysis_layout.addWidget(dataset_analysis_box)
        analysis_layout.addWidget(classifier_box)
        analysis_layout.addStretch()

        self.main_tabs = QTabWidget()
        self.main_tabs.addTab(self._prep_group, "🧰 Ön Hazırlık")
        self.main_tabs.addTab(file_group, "🎵 Ses Dosyası")
        self.main_tabs.addTab(view_group, "👁️ Görünüm")
        self.main_tabs.addTab(playproc_group, "🎵 Oynatma ve İşleme")
        self.main_tabs.addTab(zoom_group, "🔍 Yakınlaştırma")
        self.main_tabs.addTab(selection_group, "⏰ Seçim")
        self.main_tabs.addTab(detection_group, "🔍 Ötüş Tespiti")
        label_scroll = QScrollArea()
        label_scroll.setWidgetResizable(True)
        label_scroll.setFrameShape(QFrame.NoFrame)
        label_scroll.setWidget(action_group)
        self.label_tab = label_scroll
        self.main_tabs.addTab(label_scroll, "🏷️ Etiketleme")
        self.main_tabs.addTab(analysis_group, "🔬 Analiz")
        self.main_tabs.setCurrentWidget(file_group)   # açılış: Ses Dosyası (Ön Hazırlık ilk sekme ama her seferinde gerekmez)
        # Sekme etiketleri hiçbir zaman "..." ile kısaltılmaz; hepsi doğal genişlikte
        # sığmıyorsa kaydırma okları gösterilir (etiketin tamamı okunabilir kalsın).
        self.main_tabs.tabBar().setElideMode(Qt.ElideNone)
        self.main_tabs.tabBar().setExpanding(False)
        self.main_tabs.setUsesScrollButtons(True)

        left_layout.addWidget(self.main_tabs)
        self._footer_hidden_tabs = (self.label_tab, detection_group)
        self.main_tabs.currentChanged.connect(self._update_footer_visibility)
        self._update_footer_visibility()

        # Matplotlib paneli
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        self.figure = plt.figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        top_layout.addWidget(self.canvas)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top_panel)
        splitter.addWidget(self.controls_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([int(h * 0.45), int(h * 0.55)])
        main_layout.addWidget(splitter)

        self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_scroll = self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.player = QMediaPlayer()
        self.player.positionChanged.connect(self.update_position)
        self.player.durationChanged.connect(self.update_duration)

        self.selection_player = QMediaPlayer()
        self.selection_player.stateChanged.connect(self.on_selection_player_state_changed)
        self.selection_player.positionChanged.connect(self.update_selection_position)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)

        self.btn_run_process.clicked.connect(self.run_processing_pipeline)
        self.btn_save_full.clicked.connect(self.save_full_audio_to_wav)

        self.setFocusPolicy(Qt.StrongFocus)
        self.apply_compact_mode(True)

        if nr is None:
            self.chk_denoise.setEnabled(False)
            self.chk_denoise.setChecked(False)
        if pyln is None:
            self.chk_lufs.setEnabled(False)
            self.chk_lufs.setChecked(False)
        if librosa is None:
            self.btn_detect_bird_calls.setEnabled(False)

        self.create_context_menu()
        self.show()
        self.update_background_widgets_state()

    def update_background_widgets_state(self):
        """Tür Sayısı seçimine göre arka plan tür bilgileri bileşenlerini etkinleştirir/devre dışı bırakır."""
        current_type = self.cmb_type_count.currentData() or self.cmb_type_count.currentText()
        primary_selected = bool(self.cmb_turkish_name.currentText().strip())

        background_species_widgets = [
            self.cmb_background_turkish_name,
            self.txt_background_scientific_name,
            self.txt_background_family,
            self.txt_background_order,
            self.txt_other_species
        ]

        background_noise_widgets = [
            self.cmb_background_noise
        ]

        if current_type == "Single" or not primary_selected:
            for widget in background_species_widgets:
                widget.setEnabled(False)
                if widget == self.cmb_background_turkish_name:
                    widget.setCurrentText("")
                    self.txt_background_scientific_name.clear()
                    self.txt_background_family.clear()
                    self.txt_background_order.clear()
                elif widget == self.txt_other_species:
                    widget.clear()

            for widget in background_noise_widgets:
                widget.setEnabled(True)
                if widget == self.cmb_background_noise:
                    widget.setCurrentText("")
        else:
            for widget in background_species_widgets:
                widget.setEnabled(True)
            for widget in background_noise_widgets:
                widget.setEnabled(True)