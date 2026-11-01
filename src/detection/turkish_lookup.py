"""
src/detection/turkish_lookup.py
================
Perch 2.0 / BirdNET'in ürettiği bilimsel (Latince) tür adını, projenin
KENDİ mevcut `bird_species_data.json` veritabanı (518 tür, Türkçe isim +
bilimsel isim + familya + takım) üzerinden Türkçe ortak isme ve
taksonomik bilgiye çevirir.

Not: Bu, BioSegv2 prototipinde elle derlenmiş ~150 kayıtlık bir sözlük
yerine, projenin v1.0'dan beri sahip olduğu, çok daha kapsamlı ve
kürate edilmiş gerçek veri kaynağını kullanır — tekerleği yeniden icat
etmiyoruz.

Perch 2.0 (~14.597 tür) ve BirdNET (~6.000+ tür), `bird_species_data.json`
(518 tür, Türkiye/Batı Palearktik odaklı) içeriğinden çok daha geniş bir
taksonomik kapsama sahiptir. Bu nedenle bir tahmin bu veritabanında
bulunamayabilir — bu durumda `lookup()` None döner; çağıran kod bilimsel
ismi göstermeye devam etmeli, "bulunamadı" diye kullanıcıyı yanıltmamalıdır.
"""

from __future__ import annotations

from typing import Dict, Optional, TypedDict

from ..data.config import BIRD_SPECIES_DATA


class SpeciesInfo(TypedDict):
    turkish_name: str
    scientific_name: str
    family: str
    order: str


_INDEX: Optional[Dict[str, SpeciesInfo]] = None


def _build_index() -> Dict[str, SpeciesInfo]:
    index: Dict[str, SpeciesInfo] = {}
    for entry in BIRD_SPECIES_DATA:
        sci = str(entry.get("scientific_name", "")).strip().lower()
        if not sci:
            continue
        index[sci] = {
            "turkish_name": entry.get("turkish_name", ""),
            "scientific_name": entry.get("scientific_name", ""),
            "family": entry.get("family", ""),
            "order": entry.get("order", ""),
        }
    return index


def lookup(scientific_name: str) -> Optional[SpeciesInfo]:
    """
    Bilimsel isme karşılık gelen tür bilgisini (Türkçe isim, familya,
    takım) döndürür. `bird_species_data.json` içinde bulunamazsa None
    döner.
    """
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_index()
    return _INDEX.get(scientific_name.strip().lower())


def get_turkish_name(scientific_name: str) -> Optional[str]:
    info = lookup(scientific_name)
    return info["turkish_name"] if info else None


def display_name(scientific_name: str) -> str:
    """Arayüzde gösterilecek birleşik metni üretir: 'Bilimsel isim (Türkçe)'."""
    tr = get_turkish_name(scientific_name)
    return f"{scientific_name} ({tr})" if tr else scientific_name


def chart_label(scientific_name: str, with_latin: bool = False) -> str:
    """Grafik etiketi: Türkçe ad; bulunamazsa bilimsel ad. with_latin=True ise 'Türkçe (Bilimsel)'."""
    tr = get_turkish_name(scientific_name)
    if not tr:
        return scientific_name
    return f"{tr} ({scientific_name})" if with_latin else tr


def refresh_index() -> None:
    """Test veya çalışma zamanı veri güncellemesi sonrası indeksi yeniler."""
    global _INDEX
    _INDEX = _build_index()
