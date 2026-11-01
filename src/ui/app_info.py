"""Uygulama kimliği metinleri (tez kapak sayfası)."""
import os

LOGO_FILE = "sdu-logo.png"

UNIVERSITY = "T.C. SÜLEYMAN DEMİREL ÜNİVERSİTESİ"
INSTITUTE = "FEN BİLİMLERİ ENSTİTÜSÜ"
DEPARTMENT = "BİLGİSAYAR MÜHENDİSLİĞİ ANABİLİM DALI"
THESIS_TITLE = "YAPAY ZEKA İLE KUŞ SESLERİNİN AKUSTİK ANALİZİ VE TÜR ZENGİNLİĞİNİN BELİRLENMESİ"
THESIS_TYPE = "DOKTORA TEZİ"
AUTHOR = "Gökhan TURAN"
ADVISOR = "Prof. Dr. Ecir UĞUR KÜÇÜKSİLLE"
SECOND_ADVISOR = "Doç. Dr. Halil SÜEL"
PLACE_YEAR = "ISPARTA-2026"


def footer_lines():
    return (
        f"{UNIVERSITY} · {INSTITUTE} · {DEPARTMENT}",
        THESIS_TITLE,
        f"{THESIS_TYPE} · {AUTHOR} · {PLACE_YEAR}",
        f"Danışman: {ADVISOR} · İkinci Danışman: {SECOND_ADVISOR}",
    )


def find_logo(dirs):
    for d in dirs:
        p = os.path.join(d, LOGO_FILE)
        if d and os.path.isfile(p):
            return p
    return None
