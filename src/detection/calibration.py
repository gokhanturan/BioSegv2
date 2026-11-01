"""
src/detection/calibration.py
================
Wood & Kahl (2024, Journal of Ornithology) yönteminden esinlenen, hafif
bir kalibrasyon aracı: kullanıcının kendi tespit örneklemini "doğru" /
"yanlış" olarak işaretlemesini, bu etiketleri kalıcı bir günlüğe
kaydetmesini ve biriken etiketlerle ham güven skorunu GERÇEK isabet
olasılığına çeviren basit bir lojistik regresyon uydurmasını sağlar.

Bu, bugün BioSegv2 (Tkinter prototip) için tasarlanan aynı mantığın,
asıl BioSeg v2.0 (bu PyQt5 projesi) için taşınmış hâlidir.

METODOLOJİ: Bu, tam bir ROC/AUC analizinin yerini tutmaz; tek değişkenli
(güven skoru -> doğru/yanlış olasılığı) bir lojistik regresyondur. Amaç,
"bu modelin %70 dediği tespitler gerçekte yaklaşık %X oranında doğru
çıkıyor" gibi kaba ama veriye dayalı bir kalibrasyon eğrisi sunmaktır.
Güvenilir bir sonuç için makul sayıda (en az ~20-30, ideal olarak 50+)
işaretlenmiş örnek gerekir; az sayıda örnekle uydurulan bir eğri
aşırı-uyuma (overfitting) çok açıktır — bu modül az örnekte kullanıcıyı
bu konuda uyarır.

Kalıcı depolama: proje köküne (`get_project_root()`'a göre) yazılan basit
bir CSV günlüğü (`calibration_log.csv`), kaynak (Perch2/BirdNET) bazında
ayrı satırlar. Harici bir bağımlılık (scikit-learn) gerektirmemek için
lojistik regresyon, numpy ile basit bir gradyan inişiyle (gradient
descent) sıfırdan uydurulur.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from ..data.csv_handler import get_project_root as get_csv_root

DEFAULT_LOG_FILENAME = "calibration_log.csv"
MIN_SAMPLES_FOR_RELIABLE_FIT = 30


def _default_log_path() -> str:
    return os.path.join(get_csv_root(), DEFAULT_LOG_FILENAME)


@dataclass
class CalibrationRecord:
    timestamp: str
    source: str  # "Perch2" ya da "BirdNET"
    species: str
    confidence: float  # 0-100
    is_correct: bool


def append_record(
    species: str, source: str, confidence: float, is_correct: bool, log_path: Optional[str] = None
) -> None:
    """Bir doğrulama kaydını CSV günlüğüne ekler (dosya yoksa oluşturur)."""
    log_path = log_path or _default_log_path()
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "source", "species", "confidence", "is_correct"])
        writer.writerow(
            [datetime.now().isoformat(timespec="seconds"), source, species, f"{confidence:.2f}", int(is_correct)]
        )


def load_records(log_path: Optional[str] = None) -> List[CalibrationRecord]:
    log_path = log_path or _default_log_path()
    if not os.path.exists(log_path):
        return []
    records = []
    with open(log_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append(
                    CalibrationRecord(
                        timestamp=row["timestamp"],
                        source=row["source"],
                        species=row["species"],
                        confidence=float(row["confidence"]),
                        is_correct=bool(int(row["is_correct"])),
                    )
                )
            except (KeyError, ValueError):
                continue  # bozuk satırı sessizce atla
    return records


@dataclass
class CalibrationResult:
    n_samples: int
    n_correct: int
    weight: float
    bias: float
    reliable: bool

    def predict_probability(self, raw_confidence_pct: float) -> float:
        """Ham güven yüzdesini (0-100), uydurulan eğriye göre tahmini
        GERÇEK doğru-olma olasılığına (0-1) çevirir."""
        x = raw_confidence_pct / 100.0
        z = self.weight * x + self.bias
        return float(1.0 / (1.0 + np.exp(-z)))

    def format_text(self) -> str:
        lines = [
            f"Kalibrasyon {'GÜVENİLİR' if self.reliable else 'ÖN İZLEME (az örnek)'}: "
            f"{self.n_samples} örnek ({self.n_correct} doğru, {self.n_samples - self.n_correct} yanlış)",
        ]
        if not self.reliable:
            lines.append(
                f"Not: Güvenilir bir eğri için en az {MIN_SAMPLES_FOR_RELIABLE_FIT} "
                "işaretlenmiş örnek önerilir."
            )
        for pct in (10, 30, 50, 70, 90, 100):
            est = self.predict_probability(pct) * 100
            lines.append(f"  Ham güven %{pct:>3d} -> tahmini gerçek isabet %{est:.1f}")
        return "\n".join(lines)


def _fit_logistic_regression(
    x: np.ndarray, y: np.ndarray, lr: float = 0.5, n_iter: int = 2000
) -> Tuple[float, float]:
    """Tek değişkenli lojistik regresyonu basit gradyan inişiyle uydurur."""
    w, b = 0.0, 0.0
    for _ in range(n_iter):
        z = w * x + b
        pred = 1.0 / (1.0 + np.exp(-z))
        error = pred - y
        grad_w = np.mean(error * x)
        grad_b = np.mean(error)
        w -= lr * grad_w
        b -= lr * grad_b
    return float(w), float(b)


def compute_calibration(
    source: Optional[str] = None, log_path: Optional[str] = None
) -> Optional[CalibrationResult]:
    """
    Günlükteki kayıtlardan (isteğe bağlı olarak tek bir kaynağa —
    Perch2/BirdNET — filtrelenmiş) bir lojistik regresyon kalibrasyonu
    uydurur. Hiç kayıt yoksa None döner.
    """
    records = load_records(log_path)
    if source is not None:
        records = [r for r in records if r.source == source]
    if not records:
        return None

    x = np.array([r.confidence / 100.0 for r in records])
    y = np.array([1.0 if r.is_correct else 0.0 for r in records])

    if len(set(y.tolist())) == 1:
        p = 0.9 if y[0] == 1.0 else 0.1
        bias = float(np.log(p / (1 - p)))
        return CalibrationResult(
            n_samples=len(records), n_correct=int(y.sum()),
            weight=0.0, bias=bias,
            reliable=len(records) >= MIN_SAMPLES_FOR_RELIABLE_FIT,
        )

    w, b = _fit_logistic_regression(x, y)
    return CalibrationResult(
        n_samples=len(records), n_correct=int(y.sum()),
        weight=w, bias=b,
        reliable=len(records) >= MIN_SAMPLES_FOR_RELIABLE_FIT,
    )
