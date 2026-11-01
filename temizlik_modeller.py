#!/usr/bin/env python3
"""
temizlik_modeller.py
====================
BioSeg v2.0'dan kullanılmayan modelleri toplu olarak kaldırır.

Projede KALAN modeller: BirdNET 2.4, Perch 2.0, BirdNET + Perch (birleşik).
KALDIRILAN modeller:    Kendi CNN, EfficientNetV2B0, Klasik SVM, Klasik RF, YAMNet, VGGish
                        (+ kayıt etkisi tanı betiği ve bunların testleri).

KULLANIM (proje kök klasöründe):
    python temizlik_modeller.py              # KURU ÇALIŞTIRMA: yalnızca neyin kaldırılacağını listeler
    python temizlik_modeller.py --uygula     # dosyaları _kaldirilan_yedek_<tarih>/ altına TAŞIR (geri alınabilir)
    python temizlik_modeller.py --uygula --kalici   # TAŞIMAZ, KALICI SİLER (geri alınamaz)

GÜVENLİK:
  * Önce güncellenmiş pipeline.py / model_registry.py projede olmalıdır; hâlâ kaldırılacak modülleri
    kullanıyorlarsa betik durur (zorla: --zorla).
  * modeller/perch, modeller/birlesik_birdnet_perch, kök klasördeki BirdNET sonuçları, trained_classifier,
    CSV ve segment dosyalarına DOKUNULMAZ.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime

KALDIRILACAK_DOSYALAR = [
    "src/ml/handcrafted.py",
    "src/ml/classical_models.py",
    "src/ml/own_models.py",
    "src/ml/logmel.py",
    "src/ml/audioset_embeddings.py",
    "src/ml/recording_effect.py",
    "kayit_etkisi_tanisi.py",
    "tests/test_classical_models.py",
    "tests/test_own_models.py",
    "tests/test_audioset_embeddings.py",
    "tests/test_recording_effect.py",
]

# Eğitilmiş sonuçlar / önbellekler (modeller/ altında)
KALDIRILACAK_KLASORLER = [
    "modeller/kendi_cnn",
    "modeller/efficientnet",
    "modeller/klasik_svm",
    "modeller/klasik_rf",
    "modeller/yamnet",
    "modeller/vggish",
]

KALDIRILAN_MODUL_ADLARI = ("handcrafted", "classical_models", "own_models", "logmel",
                           "audioset_embeddings", "recording_effect")
KONTROL_DOSYALARI = ("src/ml/pipeline.py", "src/ml/model_registry.py", "src/ml/train_classifier.py")
ESKI_MODEL_KIMLIKLERI = ('"kendi_cnn"', '"efficientnet"', '"klasik_svm"', '"klasik_rf"', '"yamnet"', '"vggish"')
ESKI_TEST_ADLARI = ("test_classical_models", "test_own_models", "test_audioset_embeddings", "test_recording_effect")


def _boyut(yol: str) -> int:
    if os.path.isfile(yol):
        return os.path.getsize(yol)
    toplam = 0
    for kok, _d, dosyalar in os.walk(yol):
        for f in dosyalar:
            try:
                toplam += os.path.getsize(os.path.join(kok, f))
            except OSError:
                pass
    return toplam


def _mb(b: int) -> str:
    return f"{b / 1e6:.1f} MB"


def proje_koku_mu(kok: str) -> bool:
    return os.path.isfile(os.path.join(kok, "src", "ml", "pipeline.py"))


def hala_kullaniliyor(kok: str) -> list:
    """Güncellenmiş kod dosyaları kaldırılacak modüllere/modellere hâlâ başvuruyor mu?"""
    sorunlar = []
    for rel in KONTROL_DOSYALARI:
        yol = os.path.join(kok, *rel.split("/"))
        if not os.path.isfile(yol):
            continue
        with open(yol, encoding="utf-8", errors="replace") as f:
            metin = f.read()
        for ad in KALDIRILAN_MODUL_ADLARI:
            if f"from .{ad} import" in metin or f"from . import {ad}" in metin or f"import {ad}" in metin:
                sorunlar.append(f"{rel}: '{ad}' modülünü hâlâ içe aktarıyor")
        if rel.endswith("model_registry.py"):
            for kim in ESKI_MODEL_KIMLIKLERI:
                if f"id={kim}" in metin:
                    sorunlar.append(f"{rel}: {kim} modeli hâlâ kayıtlı")
    return sorunlar


def eski_pyc_listesi(kok: str) -> list:
    """Kaldırılan modüllerin artık işe yaramayan .pyc önbellekleri."""
    adlar = KALDIRILAN_MODUL_ADLARI + ("kayit_etkisi_tanisi",) + ESKI_TEST_ADLARI
    bulunan = []
    for dizin, _alt, dosyalar in os.walk(kok):
        if os.path.basename(dizin) != "__pycache__" or "_kaldirilan_yedek_" in dizin:
            continue
        for f in dosyalar:
            if f.endswith(".pyc") and f.split(".")[0] in adlar:
                bulunan.append(os.path.relpath(os.path.join(dizin, f), kok))
    return sorted(bulunan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="BioSeg: kullanılmayan modelleri projeden kaldır.")
    ap.add_argument("--uygula", action="store_true", help="Gerçekten uygula (varsayılan: yalnızca listele).")
    ap.add_argument("--kalici", action="store_true", help="Yedeğe taşımak yerine KALICI sil (--uygula ile).")
    ap.add_argument("--zorla", action="store_true", help="Kod dosyaları güncellenmemiş olsa da devam et.")
    ap.add_argument("--kok", default=os.getcwd(), help="Proje kök klasörü (varsayılan: bulunulan klasör).")
    a = ap.parse_args(argv)
    kok = os.path.abspath(a.kok)

    if not proje_koku_mu(kok):
        print(f"HATA: {kok} bir BioSeg proje kökü değil (src/ml/pipeline.py yok).\n"
              "Betiği proje kök klasöründe çalıştırın ya da --kok verin.")
        return 2

    sorunlar = hala_kullaniliyor(kok)
    if sorunlar and not a.zorla:
        print("DURDU: kod dosyaları henüz güncellenmemiş; şimdi silmek uygulamayı bozar.\n")
        for s in sorunlar:
            print("  -", s)
        print("\nÖnce güncellenmiş dosyaları (pipeline.py, model_registry.py, model_significance.py, "
              "train_classifier.py, main_window.py, testler) projeye kopyalayın, sonra tekrar çalıştırın.")
        return 3

    hedefler = []
    for rel in KALDIRILACAK_DOSYALAR + KALDIRILACAK_KLASORLER:
        yol = os.path.join(kok, *rel.split("/"))
        if os.path.exists(yol):
            hedefler.append((rel, yol))
    for rel in eski_pyc_listesi(kok):
        hedefler.append((rel, os.path.join(kok, rel)))

    if not hedefler:
        print("Kaldırılacak bir şey bulunamadı (zaten temiz).")
        return 0

    toplam = sum(_boyut(y) for _r, y in hedefler)
    baslik = "UYGULANIYOR" if a.uygula else "KURU ÇALIŞTIRMA (hiçbir şey değişmez)"
    print(f"{baslik} — {len(hedefler)} öğe, {_mb(toplam)}\n")
    for rel, yol in hedefler:
        print(f"  {'[klasör]' if os.path.isdir(yol) else '[dosya] '} {rel}   ({_mb(_boyut(yol))})")

    if not a.uygula:
        print("\nUygulamak için:  python temizlik_modeller.py --uygula"
              "\n(Dosyalar silinmez; _kaldirilan_yedek_<tarih>/ klasörüne taşınır, geri alınabilir.)")
        return 0

    if a.kalici:
        for _rel, yol in hedefler:
            if os.path.isdir(yol):
                shutil.rmtree(yol)
            else:
                os.remove(yol)
        print("\nKALICI olarak silindi.")
        return 0

    yedek = os.path.join(kok, "_kaldirilan_yedek_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    for rel, yol in hedefler:
        hedef = os.path.join(yedek, *rel.split("/"))
        os.makedirs(os.path.dirname(hedef), exist_ok=True)
        shutil.move(yol, hedef)
    print(f"\nTaşındı: {yedek}\nHer şey çalışıyorsa bu klasörü kendiniz silebilirsiniz; "
          "geri almak için içindekileri aynı göreli yollara geri taşıyın.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
