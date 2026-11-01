"""BioSeg marka teması: ana pencere ve tüm formlarda ortak renk geçişi (mavi -> yeşil) ve başlık bandı."""
import re

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

BRAND_BLUE = "#0b3d6b"
BRAND_GREEN = "#0f8b6d"
BRAND_BLUE_HOVER = "#0e4f85"
BRAND_GREEN_HOVER = "#12a07c"
BRAND_BLUE_PRESSED = "#082f54"
BRAND_GREEN_PRESSED = "#0b6f57"


def gradient(c1=BRAND_BLUE, c2=BRAND_GREEN) -> str:
    """Soldan sağa renk geçişi (Qt stil sayfası değeri)."""
    return f"qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {c1}, stop:1 {c2})"


def plain_title(window_title: str) -> str:
    """Pencere başlığındaki baştaki simgeleri (emoji) atar."""
    return re.sub(r"^[^\w]+", "", window_title or "").strip()


def make_hero(title: str, subtitle: str = "", compact: bool = False) -> QFrame:
    """Gradyanlı başlık bandı. compact=True: ana pencere ve küçük formlar için ince sürüm."""
    hero = QFrame()
    hero.setObjectName("hero")
    hero.setStyleSheet(f"QFrame#hero {{ background: {gradient()}; border-radius: {10 if compact else 14}px; }}")
    lay = QHBoxLayout(hero)
    lay.setContentsMargins(20, 8 if compact else 16, 20, 8 if compact else 16)
    col = QVBoxLayout()
    col.setSpacing(2)
    t = QLabel(title)
    t.setStyleSheet(f"color:white; background:transparent; font-size:{19 if compact else 26}px; font-weight:800;")
    col.addWidget(t)
    if subtitle:
        st = QLabel(subtitle)
        st.setStyleSheet(f"color:#dbeafe; background:transparent; font-size:{12 if compact else 13}px;")
        col.addWidget(st)
    lay.addLayout(col, 1)
    return hero


def add_hero(layout, title: str, subtitle: str = "", compact: bool = True) -> QFrame:
    hero = make_hero(plain_title(title), subtitle, compact)
    layout.addWidget(hero)
    return hero
