"""
src/ml/dataset_builder.py
================
`bird_sound_dataset.csv` ve `segments/classification/<tür>/` klasör
yapısından, sınıflandırıcı eğitimi için kullanılacak örnek listesini
oluşturur.

KRİTİK TASARIM NOKTASI — veri sızıntısını önleme: Aynı orijinal kayıttan
kesilmiş birden fazla segment, train/val/test arasında BÖLÜNMEMELİDİR
(aksi halde model, aynı kaydın arka plan gürültüsünü/karakteristiğini
"ezberleyip" yapay olarak yüksek doğruluk gösterebilir — gerçek
genelleme performansını ölçmez). Bu yüzden her örneğe, hangi orijinal
kayıttan geldiğini belirten bir `recording_id` atanır (CSV'deki
`Original_Recording_Name` alanından) ve bölme işlemi bu kimliğe göre
GRUP-FARKINDA (group-aware) yapılır — bkz. `train_classifier.py`daki
`group_aware_split`.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from ..data.classification_folders import _sanitize_folder_name, CLASSIFICATION_FOLDER_NAME
from ..data.config import get_project_root as get_src_root
from ..data.csv_handler import get_project_root as get_csv_root
from ..detection import turkish_lookup

# Sınıflandırmaya alınacak türler için varsayılan en az segment sayısı (tüm eğitim/doğrulama yolları bunu kullanır)
DEFAULT_MIN_SEGMENTS = 30


@dataclass
class TrainingSample:
    filepath: str
    species: str
    recording_id: str  # grup-farkında bölme için — aynı kayıttan gelen segmentler asla farklı setlere düşmemeli


def build_training_samples(
    csv_path: Optional[str] = None,
    classification_dir: Optional[str] = None,
    min_samples_per_class: int = DEFAULT_MIN_SEGMENTS,
    min_recordings_per_class: int = 2,
    restrict_to_known_bird_species: bool = False,
) -> Tuple[List[TrainingSample], List[str], Dict[str, int]]:
    """
    Eğitim için uygun örnekleri döndürür.

    Dönüş: (kullanılacak örnekler, DIŞLANAN türlerin listesi, TÜM
    türlerin ham SEGMENT sayısı — dışlananlar dahil).

    Bir tür şu şartlardan HERHANGİ BİRİNİ karşılamazsa dışlanır (hata
    fırlatılmaz):
    1. `min_samples_per_class`'tan az segmenti varsa (çok az örnekle
       eğitim, güvenilir olmayan/aşırı-uyumlu bir sınıflandırıcı üretir).
    2. `min_recordings_per_class`'tan az FARKLI orijinal kaydı varsa —
       KRİTİK: bir tür sadece 1 kayıttan geliyorsa, o kayıt ya train'e ya
       test'e gidebilir ama İKİSİNE BİRDEN GİDEMEZ (aksi veri sızıntısı
       olur) — yani böyle bir tür asla düzgün eğitilip test edilemez.
       Varsayılan eşik 2 (en az 1 kayıt train'e, en az 1 kayıt test'e).
    3. `restrict_to_known_bird_species=True` VE tür, projenin 518 türlük
       kürate edilmiş kuş veritabanında (`bird_species_data.json`) yoksa
       — Perch 2.0/BirdNET çok-taksonlu modeller olduğu için (kurbağa,
       böcek, memeli de tanıyabilirler), bu VARSAYILAN OLARAK KAPALI
       (False) bir güvenlik filtresidir; açıldığında eğitim setinin
       KESİNLİKLE sadece bilinen kuş türlerinden oluşmasını garanti eder.
    """
    if csv_path is None:
        csv_path = os.path.join(get_csv_root(), "bird_sound_dataset.csv")
    if classification_dir is None:
        classification_dir = os.path.join(get_src_root(), "segments", CLASSIFICATION_FOLDER_NAME)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Veri seti CSV dosyası bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    # Aynı kaydın iki kanalı/sürümü (…L / …M, aynı nokta, aynı süre) tek kayıt kimliği alır; aksi hâlde
    # biri eğitime, biri teste düşüp sızıntı yaratabilir (bkz. src/data/recording_groups.py).
    from ..data.recording_groups import canonical_recording_ids
    df["_recording_group"] = canonical_recording_ids(df)

    all_samples: List[TrainingSample] = []
    for _, row in df.iterrows():
        species = str(row.get("Scientific_Name", "")).strip()
        filename = str(row.get("Filename", "")).strip()
        recording_id = str(row.get("_recording_group", "")).strip()

        if not species or species.lower() == "nan":
            continue
        if not filename or filename.lower() == "nan":
            continue
        if not recording_id or recording_id.lower() == "nan":
            # Orijinal kayıt bilgisi yoksa, dosya adının kendisini grup
            # kimliği olarak kullan (en azından AYNI dosyadan gelen
            # yinelenen satırlar gruplanır; ideal değil ama güvenli).
            recording_id = filename

        folder_name = _sanitize_folder_name(species)
        filepath = os.path.join(classification_dir, folder_name, filename)
        if not os.path.exists(filepath):
            continue

        all_samples.append(TrainingSample(filepath=filepath, species=species, recording_id=recording_id))

    raw_counts = Counter(s.species for s in all_samples)

    species_recordings: Dict[str, Set[str]] = {}
    for s in all_samples:
        species_recordings.setdefault(s.species, set()).add(s.recording_id)

    kept_species: Set[str] = {
        sp for sp, c in raw_counts.items()
        if c >= min_samples_per_class and len(species_recordings.get(sp, set())) >= min_recordings_per_class
    }

    if restrict_to_known_bird_species:
        # bird_species_data.json'da (518 kuş türü) BULUNMAYAN türleri
        # eğitim setinden çıkar — Perch 2.0/BirdNET kurbağa/böcek/memeli
        # de önerebildiği için, bu KESİN bir "sadece kuş" garantisi verir.
        kept_species = {sp for sp in kept_species if turkish_lookup.lookup(sp) is not None}

    excluded_species = sorted(sp for sp in raw_counts if sp not in kept_species)

    filtered_samples = [s for s in all_samples if s.species in kept_species]

    return filtered_samples, excluded_species, dict(raw_counts)
