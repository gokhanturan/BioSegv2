"""BioSeg Labeling Studio ana giriş noktası."""

import sys
import os
import faulthandler
import traceback
from datetime import datetime
import matplotlib
# PyQt5'i içe aktarmadan önce arka ucu (backend) ayarla
matplotlib.use('Qt5Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from src.ui.main_window import BirdSoundApp


_LOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_fault_file = None


def _install_error_handlers():
    """
    Hata yakalayıcılar. PyQt5, bir Qt slotu içinde yakalanmamış Python hatası olunca
    sys.excepthook varsayılan değilse programı SESSİZCE sonlandırır (Windows'ta çıkış kodu
    0xC0000409). Kendi excepthook'umuzu kurunca program kapanmaz; hata günlüğe yazılır ve
    kullanıcıya gösterilir. faulthandler ise gerçek (yerel kütüphane) çökmelerinde Python
    yığınını bioseg_cokme.log dosyasına yazar.
    """
    global _fault_file
    try:
        _fault_file = open(os.path.join(_LOG_DIR, "bioseg_cokme.log"), "a", encoding="utf-8")
        faulthandler.enable(file=_fault_file, all_threads=True)
    except Exception:
        pass

    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(os.path.join(_LOG_DIR, "bioseg_hata.log"), "a", encoding="utf-8") as f:
                f.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} ===\n{text}")
        except Exception:
            pass
        sys.stderr.write(text)
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox
            if QApplication.instance() is not None:
                box = QMessageBox(QMessageBox.Warning, "Beklenmeyen hata",
                                  f"{exc_type.__name__}: {exc}\n\nProgram açık kalır. Ayrıntı: bioseg_hata.log")
                box.setDetailedText(text)
                box.exec_()
        except Exception:
            pass

    sys.excepthook = _hook


def main():
    """BioSeg uygulamasını başlatır ve çalıştırır."""
    _install_error_handlers()
    # Modern monitörlerde daha iyi görüntü için yüksek DPI ölçeklemeyi etkinleştir
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # "Konum Haritası" özelliğinin kullandığı QWebEngineView, bu özniteliğin
    # QApplication oluşturulmadan ÖNCE ayarlanmasını gerektirir — aksi
    # hâlde (PyQtWebEngine kurulu olsa bile) ImportError verir ve harita
    # sessizce statik moda döner (dünya çapında kaydırma yapılamaz).
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)

    app = QApplication(sys.argv)
    window = BirdSoundApp()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()