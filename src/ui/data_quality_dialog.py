"""Veri Kalitesi Denetimi penceresi. Denetim salt-okunurdur; CSV/dosya değişikliği yalnızca iki onayla yapılır."""
import os
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QPushButton,
                             QFileDialog, QMessageBox)

from ..data import data_quality
from . import theme


class DataQualityDialog(QDialog):
    def __init__(self, csv_path, search_dirs=None, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.search_dirs = search_dirs or []
        self.report = None
        self.setWindowTitle("🧹 Veri Kalitesi Denetimi")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.resize(960, 760)
        lay = QVBoxLayout(self)
        theme.add_hero(lay, self.windowTitle(), "Veri setinde tutarsız ve şüpheli kayıtların denetimi (yinelenenler iki onayla temizlenir)")
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        lay.addWidget(self.text, 1)
        row = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 Yenile")
        self.btn_export = QPushButton("📤 Şüpheli Satırları CSV'ye Aktar")
        self.btn_fix = QPushButton("🗑 Yinelenenleri Sil / Düzelt")
        self.btn_fix.setToolTip("Yinelenen (fazlalık) satırları CSV'den ve ses dosyalarını segment klasöründen kaldırır. "
                                "İki kez onay istenir; CSV'nin yedeği alınır, dosyalar yedek klasöre taşınır.")
        self.btn_close = QPushButton("Kapat")
        row.addWidget(self.btn_refresh)
        row.addWidget(self.btn_export)
        row.addWidget(self.btn_fix)
        row.addStretch()
        row.addWidget(self.btn_close)
        lay.addLayout(row)
        self.btn_refresh.clicked.connect(self.run)
        self.btn_export.clicked.connect(self._export)
        self.btn_fix.clicked.connect(self._fix_duplicates)
        self.btn_close.clicked.connect(self.close)
        self.run()

    def run(self):
        df = data_quality.read_text(self.csv_path)
        if df is None:
            self.report = None
            self.text.setPlainText(f"CSV dosyası bulunamadı veya boş:\n{self.csv_path}")
            self.btn_export.setEnabled(False)
            self.btn_fix.setEnabled(False)
            return
        self.report = data_quality.audit(df, search_dirs=self.search_dirs)
        mtime = datetime.fromtimestamp(os.path.getmtime(self.csv_path)).strftime("%d.%m.%Y %H:%M")
        self.text.setPlainText(
            f"Dosya: {self.csv_path}\nSon değişiklik: {mtime} (rapor, bu andaki dosyadan hesaplanır; "
            "veri değişince 🔄 Yenile'ye basın)\n\n"
            + self.report.format_text()
            + "\n\nNot: Denetim yalnızca okur. CSV ve segment dosyalarında değişiklik, yalnızca "
              "'Yinelenenleri Sil / Düzelt' ile ve iki onaydan sonra yapılır."
        )
        self.btn_export.setEnabled(len(self.report.suspicious_rows()) > 0)
        self.btn_fix.setEnabled(self.report.n_duplicate_extra > 0)

    def _export(self):
        if self.report is None:
            return
        sus = self.report.suspicious_rows()
        path, _ = QFileDialog.getSaveFileName(self, "Şüpheli satırları kaydet",
                                              "supheli_satirlar.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            sus.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
        except OSError as e:
            QMessageBox.warning(self, "Hata", f"Kaydedilemedi: {e}")
            return
        QMessageBox.information(self, "Bilgi", f"{len(sus)} satır kaydedildi:\n{path}")

    def _fix_duplicates(self):
        """Yinelenen satırları iki ayrı onaydan sonra siler (CSV yedeklenir, dosyalar yedek klasöre taşınır)."""
        plan = data_quality.plan_duplicate_removal(self.csv_path, self.search_dirs)
        if plan.n_rows == 0:
            QMessageBox.information(self, "Bilgi", "Silinecek yinelenen satır bulunamadı.")
            return
        warn = ""
        if self.report is not None:
            real = [c for c in self.report.dup_differing_cols if c not in data_quality.TRIVIAL_DIFF_COLS]
            if real:
                warn = ("\n\nDİKKAT: Yinelenen gruplarda şu sütunların değeri farklı: " + ", ".join(real)
                        + ". Bu, bilinçli ikinci bir etiket olabilir; yalnızca ilk satır korunur.")
        shared = (f"\n{len(plan.files_kept_shared)} dosya, kalan bir satır tarafından da kullanıldığı için korunur."
                  if plan.files_kept_shared else "")
        Q = QMessageBox
        first = Q.question(
            self, "Yinelenenleri Sil / Düzelt (1/2)",
            f"{plan.n_groups} yinelenen gruptan {plan.n_rows} fazlalık satır CSV'den silinecek.\n"
            f"{len(plan.files_to_move)} ses dosyası segment klasöründen kaldırılacak.{shared}\n\n"
            "Her grupta ilk satır korunur, sonrakiler silinir." + warn + "\n\nDevam edilsin mi?",
            Q.Yes | Q.No, Q.No)
        if first != Q.Yes:
            return
        second = Q.question(
            self, "Yinelenenleri Sil / Düzelt (2/2) — SON ONAY",
            f"Bu işlem CSV dosyasını DEĞİŞTİRİR:\n{self.csv_path}\n\n"
            f"• {plan.n_rows} satır silinecek, {len(plan.files_to_move)} ses dosyası kaldırılacak.\n"
            "• İşlemden önce CSV'nin yedeği alınır; ses dosyaları kalıcı silinmez, "
            "'_silinen_yedek_…' klasörüne taşınır.\n\nGerçekten silmek istiyor musunuz?",
            Q.Yes | Q.No, Q.No)
        if second != Q.Yes:
            return
        try:
            res = data_quality.apply_duplicate_removal(self.csv_path, self.search_dirs)
        except OSError as exc:
            QMessageBox.warning(self, "Hata", f"İşlem yapılamadı: {exc}")
            self.run()
            return
        msg = (f"{res.rows_removed} satır CSV'den silindi, {len(res.files_moved)} ses dosyası kaldırıldı.\n\n"
               f"CSV yedeği: {res.backup_csv}")
        if res.backup_dir:
            msg += f"\nSilinen dosyalar (geri alınabilir): {res.backup_dir}"
        if res.errors:
            msg += "\n\nTaşınamayan dosyalar:\n" + "\n".join(res.errors[:10])
        QMessageBox.information(self, "Tamamlandı", msg)
        self.run()
