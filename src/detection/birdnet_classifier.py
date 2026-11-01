"""
src/detection/birdnet_classifier.py
================
BirdNET-Analyzer'ı (Cornell K. Lisa Yang Center for Conservation
Bioacoustics) `birdnetlib` üzerinden çağıran sarmalayıcı.

LİSANS NOTU: BirdNET-Analyzer modelleri CC BY-NC-SA 4.0 lisanslıdır;
BirdNET-Team'in kendi belgelerine göre eğitim/araştırma kullanımı
"ticari olmayan kullanım" kapsamında serbestçe izinlidir. Kullanırken
şu makaleye atıf verin: Kahl, S., Wood, C.M., Eibl, M., Klinck, H.
(2021). BirdNET: A deep learning solution for avian diversity
monitoring. Ecological Informatics, 61, 101236.

BirdNET'in son katmanındaki sigmoid eğrisi kasıtlı olarak
düzleştirilmiştir (duyarlılık=0.85), bu da Perch 2.0'a kıyasla daha
kademeli güven skorları üretir — yine de gerçek olasılık değildir.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

BIRDNET_NON_SPECIES_LABELS = {
    "Power tools", "Human non-vocal", "Human vocal", "Human whistle",
    "Dog", "Engine", "Fireworks", "Gun", "Siren", "Environmental", "Noise", "Silence",
}


class BirdNETLoadError(RuntimeError):
    """BirdNET modeli/kütüphanesi yüklenemediğinde fırlatılır."""


_analyzer = None  # modül seviyesinde önbelleklenir


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        try:
            from birdnetlib.analyzer import Analyzer
        except ImportError as exc:
            raise BirdNETLoadError(
                "birdnetlib kurulu değil. Kurmak için:\n"
                "  pip install birdnetlib\n"
                "Ayrıca sisteminizde ffmpeg kurulu olmalıdır."
            ) from exc
        _analyzer = Analyzer()
    return _analyzer


def is_birdnet_species_label(label: str) -> bool:
    return label not in BIRDNET_NON_SPECIES_LABELS


def classify_recording(
    filepath: str,
    min_conf: float = 0.1,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    date: Optional[datetime] = None,
) -> List[Tuple[float, float, str, float]]:
    """
    BirdNET-Analyzer ile bir ses dosyasını analiz eder.
    Dönüş: [(başlangıç_sn, bitiş_sn, bilimsel_ad, güven_yüzde), ...]
    """
    try:
        from birdnetlib import Recording
    except ImportError as exc:
        raise BirdNETLoadError(
            "birdnetlib kurulu değil. Kurmak için:\n"
            "  pip install birdnetlib\n"
            "Ayrıca sisteminizde ffmpeg kurulu olmalıdır."
        ) from exc

    analyzer = _get_analyzer()
    kwargs = {"min_conf": min_conf}
    if lat is not None and lon is not None:
        kwargs["lat"] = lat
        kwargs["lon"] = lon
    if date is not None:
        kwargs["date"] = date

    recording = Recording(analyzer, filepath, **kwargs)
    recording.analyze()

    results: List[Tuple[float, float, str, float]] = []
    for det in recording.detections:
        results.append(
            (
                float(det["start_time"]),
                float(det["end_time"]),
                det["scientific_name"],
                float(det["confidence"]) * 100.0,
            )
        )
    return results
