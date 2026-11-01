"""
src/detection/species_merge.py
================
Perch 2.0 ve BirdNET'in BAĞIMSIZ ürettiği tespit listelerini, tek bir
tabloda "ayrı sütunlar" halinde göstermek üzere birleştirir (union).

TASARIM GEREKÇESİ: Enerji-tabanlı eski bölge bulucu her zaman güvenilir
çalışmıyordu ve tek başına ne Perch ne de BirdNET her şeyi
yakalayabiliyordu. Bu modül, iki modelin kendi dahili tarama
mantıklarını (Perch: kayan 5 s pencere; BirdNET: kendi iç parçalama)
TÜM kayıt üzerinde BAĞIMSIZ çalıştırdıktan sonra sonuçları birleştirir:

1. Perch tespitleri (kendi içinde zaten tür bazında birleştirilmiş) satır
   iskeletini oluşturur.
2. Her Perch satırı için, zaman aralığı örtüşen bir BirdNET tespiti
   aranır; bulunursa o satırın BirdNET sütunları doldurulur.
3. Hiçbir Perch satırıyla eşleşmeyen BirdNET tespitleri, sadece BirdNET
   sütunları dolu YENİ satırlar olarak eklenir.

Sonuç: bir model bir bölgeyi kaçırsa bile diğeri yakaladıysa o bölge
tabloda kayıp gitmez — "bazı yerleri tespit edemiyordu" sorununu
doğrudan hedefler.

Güven skorları TOPLANMAZ/ORTALANMAZ: Perch ve BirdNET'in güven ölçekleri
birbiriyle doğrudan karşılaştırılabilir değildir (Perch'in son katmanı
kalibre edilmemiştir; BirdNET'in sigmoid eğrisi kasıtlı düzleştirilmiştir,
Kahl ve ark. 2021). Bu yüzden her kaynağın güveni ayrı ayrı gösterilir.

NOT (v2.0): "Ötüş Tespiti" sekmesi artık yalnızca BirdNET kullanır;
`detector.py` bu fonksiyonu `perch_detections=[]` (boş liste) ile çağırır.
Perch ile ilgili mantık, ileride Perch'in geri eklenebilmesi için
altyapı olarak korunmuştur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import turkish_lookup


@dataclass
class UnifiedDetection:
    """Tablodaki tek bir satırı temsil eder."""

    row_id: int
    start_s: float
    end_s: float

    perch_species: Optional[str] = None
    perch_turkish: Optional[str] = None
    perch_confidence: Optional[float] = None
    perch_is_species: bool = True  # False ise FSD50K genel ses kategorisi

    birdnet_species: Optional[str] = None
    birdnet_turkish: Optional[str] = None
    birdnet_confidence: Optional[float] = None

    @property
    def has_perch(self) -> bool:
        return self.perch_species is not None

    @property
    def has_birdnet(self) -> bool:
        return self.birdnet_species is not None

    @property
    def is_cross_validated(self) -> bool:
        """İki model de AYNI türü mü buldu?"""
        if not (self.has_perch and self.has_birdnet):
            return False
        return self.perch_species.strip().lower() == self.birdnet_species.strip().lower()

    @property
    def best_species_for_labeling(self) -> Optional[str]:
        """
        Etiketleme formunu otomatik doldurmak için en iyi tür önerisini
        döndürür. Öncelik: çapraz doğrulanmış > Perch (tür ise) > BirdNET.
        """
        if self.is_cross_validated:
            return self.perch_species
        if self.has_perch and self.perch_is_species:
            return self.perch_species
        if self.has_birdnet:
            return self.birdnet_species
        return None


def _ranges_overlap(a_start: float, a_end: float, b_start: float, b_end: float, tolerance_s: float) -> bool:
    return max(a_start, b_start) - tolerance_s <= min(a_end, b_end) + tolerance_s


def merge_across_sources(
    perch_detections: List[Tuple[float, float, str, float]],
    birdnet_detections: List[Tuple[float, float, str, float]],
    tolerance_s: float = 2.0,
) -> List[UnifiedDetection]:
    """
    Perch ve BirdNET tespit listelerini (start_s, end_s, tür_adı,
    güven_yüzde) birleştirip UnifiedDetection satırları üretir.
    """
    from .perch_classifier import is_probable_species
    from .birdnet_classifier import is_birdnet_species_label

    rows: List[UnifiedDetection] = []
    consumed_birdnet_idx = set()

    for p_start, p_end, p_species, p_conf in sorted(perch_detections, key=lambda d: d[0]):
        row = UnifiedDetection(
            row_id=len(rows),
            start_s=p_start,
            end_s=p_end,
            perch_species=p_species,
            perch_turkish=turkish_lookup.get_turkish_name(p_species) if is_probable_species(p_species) else None,
            perch_confidence=p_conf,
            perch_is_species=is_probable_species(p_species),
        )

        # Zaman + tür örtüşen ilk (en yüksek güvenli) BirdNET tespitini bul
        best_match_idx = None
        best_match_conf = -1.0
        for idx, (b_start, b_end, b_species, b_conf) in enumerate(birdnet_detections):
            if idx in consumed_birdnet_idx:
                continue
            if _ranges_overlap(p_start, p_end, b_start, b_end, tolerance_s) and b_conf > best_match_conf:
                best_match_idx = idx
                best_match_conf = b_conf

        if best_match_idx is not None:
            b_start, b_end, b_species, b_conf = birdnet_detections[best_match_idx]
            row.birdnet_species = b_species
            row.birdnet_turkish = (
                turkish_lookup.get_turkish_name(b_species) if is_birdnet_species_label(b_species) else None
            )
            row.birdnet_confidence = b_conf
            row.end_s = max(row.end_s, b_end)
            consumed_birdnet_idx.add(best_match_idx)

        rows.append(row)

    # Perch'le eşleşmeyen kalan BirdNET tespitlerini ayrı satır olarak ekle
    for idx, (b_start, b_end, b_species, b_conf) in enumerate(birdnet_detections):
        if idx in consumed_birdnet_idx:
            continue
        rows.append(
            UnifiedDetection(
                row_id=len(rows),
                start_s=b_start,
                end_s=b_end,
                birdnet_species=b_species,
                birdnet_turkish=(
                    turkish_lookup.get_turkish_name(b_species) if is_birdnet_species_label(b_species) else None
                ),
                birdnet_confidence=b_conf,
            )
        )

    rows.sort(key=lambda r: r.start_s)
    for i, row in enumerate(rows):
        row.row_id = i
    return rows
