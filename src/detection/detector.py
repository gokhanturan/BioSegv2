"""
src/detection/detector.py
================
BioSeg v2.0 — BirdNET-Analyzer tabanlı kuş ötüşü tespit iş parçacığı.

NOT: "Ötüş Tespiti" YALNIZCA BirdNET 2.4 kullanır. Perch 2.0 tespitte kullanılmaz;
ancak sınıflandırıcı eğitiminde (ml/model_registry.py: BirdNET 2.4, Perch 2.0 ve birleşik
BirdNET+Perch) dondurulmuş gömme çıkarıcı olarak arayüzden kullanılır. Perch sarmalayıcısı
(`perch_classifier.py`) ayrıca araştırma betiklerinde (gomme_karsilastirma.py,
girdi_bicimi_kontrolu.py, artirma_deneyi.py) de kullanılır.
"""

from __future__ import annotations

import traceback
from typing import List, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from .perch_classifier import PerchClassifier
from . import birdnet_classifier
from .species_merge import UnifiedDetection, merge_across_sources

# Perch 2.0 modeli, "Tür Sınıflandırma Eğitimi" özelliği (embedding çıkarma)
# için hâlâ paylaşımlı/önbellekli tutulur — bu fonksiyon
# `src/ml/pipeline.py` tarafından çağrılır ve Ötüş Tespiti ile ilgisi yoktur.
_shared_perch_classifier: Optional[PerchClassifier] = None


def get_shared_perch_classifier() -> PerchClassifier:
    global _shared_perch_classifier
    if _shared_perch_classifier is None:
        _shared_perch_classifier = PerchClassifier(use_cpu_variant=True)
    return _shared_perch_classifier


class SpeciesDetectionThread(QThread):
    """
    BirdNET-Analyzer ile kuş ötüşü tespiti ve tür sınıflandırması yapan
    QThread. Perch 2.0'ı, eski enerji tabanlı bölge bulucuyu veya
    UMAP/HDBSCAN kümelemesini İÇERMEZ.
    """

    progress = pyqtSignal(int, str)  # yüzde, durum mesajı
    finished = pyqtSignal(list, list, list)  # unified_rows, perch_raw (her zaman boş), birdnet_raw
    error = pyqtSignal(str)

    def __init__(
        self,
        y,
        sr: int,
        analysis_filepath: str,
        # BirdNET parametreleri
        birdnet_min_conf: float = 0.1,
        birdnet_lat: Optional[float] = None,
        birdnet_lon: Optional[float] = None,
        parent=None,
    ):
        """
        `analysis_filepath`: BirdNET'in analiz edeceği ses dosyasının yolu.
        Bu, özgün (diskteki) dosya OLABİLİR ya da (Gürültü Azaltma / LUFS
        Normalizasyonu ön işlemi uygulandıysa) işlenmiş sesin yazıldığı
        GEÇİCİ bir WAV dosyası olabilir — hangisinin kullanılacağına
        çağıran kod (main_window.py) karar verir.
        """
        super().__init__(parent)
        self.y = y
        self.sr = sr
        self.analysis_filepath = analysis_filepath

        self.birdnet_min_conf = birdnet_min_conf
        self.birdnet_lat = birdnet_lat
        self.birdnet_lon = birdnet_lon

    def run(self):
        birdnet_raw: List = []

        try:
            self.progress.emit(10, "BirdNET-Analyzer çalıştırılıyor...")
            try:
                birdnet_raw = self._run_birdnet()
            except birdnet_classifier.BirdNETLoadError as exc:
                self.error.emit(f"BirdNET yüklenemedi: {exc}")

            self.progress.emit(95, "Sonuçlar hazırlanıyor...")
            # perch_raw her zaman boş bir listedir — merge_across_sources
            # yine de çağrılır; böylece UnifiedDetection/tablo mantığı
            # değişmez (istenirse Perch'in ileride kolayca geri eklenebilmesi
            # için altyapı korunmuştur).
            unified = merge_across_sources([], birdnet_raw)

            self.progress.emit(100, "Tamamlandı.")
            self.finished.emit(unified, [], birdnet_raw)

        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Tespit sırasında genel hata: {str(e)}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    def _run_birdnet(self):
        return birdnet_classifier.classify_recording(
            self.analysis_filepath,
            min_conf=self.birdnet_min_conf,
            lat=self.birdnet_lat,
            lon=self.birdnet_lon,
        )