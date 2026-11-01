"""
src/data/classification_folders.py
================
`segments/` klasöründeki (zaten kesilmiş) ses dosyalarını, veri setindeki
(`bird_sound_dataset.csv`) `Scientific_Name` alanına göre
`segments/classification/<Scientific_Name>/` altında tür-başına-klasör
yapısına KOPYALAR — bir sonraki aşamada (sınıflandırıcı eğitimi) klasör
yapısının hazır olması için.

TASARIM KARARLARI:
- `segments/` klasöründeki ORİJİNAL dosyalara DOKUNULMAZ (silinmez,
  taşınmaz) — sadece kopyalanır. Kullanıcı bunu birden çok kez
  çalıştırabilir (örn. yeni segmentler eklendikçe); zaten kopyalanmış ve
  değişmemiş dosyalar tekrar kopyalanmaz (dosya boyutu karşılaştırılarak
  atlanır).
- `Scientific_Name` boş olan (henüz tam etiketlenmemiş) ya da kaynak ses
  dosyası diskte bulunamayan kayıtlar sessizce ATLANIR — hata fırlatılmaz,
  sadece sayılır ve kullanıcıya özet halinde bildirilir.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from .config import get_project_root as get_src_root
from .csv_handler import get_project_root as get_csv_root

CLASSIFICATION_FOLDER_NAME = "classification"


def _sanitize_folder_name(name: str) -> str:
    """Tür adını, Windows/Linux'ta geçerli bir klasör adına çevirir."""
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    return name or "Unlabeled"


@dataclass
class BuildResult:
    classification_dir: str
    species_folders_created: int
    files_copied: int
    files_already_present: int
    skipped_missing_scientific_name: int
    skipped_missing_source_file: int
    species_counts: Dict[str, int] = field(default_factory=dict)

    def format_summary(self) -> str:
        lines = [
            f"{self.species_folders_created} tür klasörü hazır ({self.classification_dir}).",
            f"{self.files_copied} yeni dosya kopyalandı.",
            f"{self.files_already_present} dosya zaten mevcuttu (değişmemiş, atlandı).",
        ]
        if self.skipped_missing_scientific_name:
            lines.append(
                f"⚠ {self.skipped_missing_scientific_name} kayıt, 'Scientific_Name' alanı boş "
                "olduğu için atlandı (önce bu segmentleri etiketleyin)."
            )
        if self.skipped_missing_source_file:
            lines.append(
                f"⚠ {self.skipped_missing_source_file} kayıt, kaynak ses dosyası "
                "'segments/' klasöründe bulunamadığı için atlandı."
            )
        return "\n".join(lines)


def build_classification_folders(
    csv_path: Optional[str] = None,
    segments_dir: Optional[str] = None,
    classification_dir: Optional[str] = None,
) -> BuildResult:
    """
    Veri setindeki her segmenti, Scientific_Name'ine göre
    `segments/classification/<Scientific_Name>/` altına kopyalar.
    Yollar verilmezse, uygulamanın gerçek konumlarına göre otomatik
    çözümlenir (main_window.py'nin kullandığı aynı mantıkla tutarlı).
    """
    if csv_path is None:
        csv_path = os.path.join(get_csv_root(), "bird_sound_dataset.csv")
    if segments_dir is None:
        segments_dir = os.path.join(get_src_root(), "segments")
    if classification_dir is None:
        classification_dir = os.path.join(segments_dir, CLASSIFICATION_FOLDER_NAME)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Veri seti CSV dosyası bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    os.makedirs(classification_dir, exist_ok=True)

    species_folders = set()
    files_copied = 0
    files_already_present = 0
    skipped_missing_scientific_name = 0
    skipped_missing_source_file = 0
    species_counts: Dict[str, int] = {}

    for _, row in df.iterrows():
        scientific_name = str(row.get("Scientific_Name", "")).strip()
        filename = str(row.get("Filename", "")).strip()

        if not scientific_name or scientific_name.lower() == "nan":
            skipped_missing_scientific_name += 1
            continue
        if not filename or filename.lower() == "nan":
            skipped_missing_source_file += 1
            continue

        src_path = os.path.join(segments_dir, filename)
        if not os.path.exists(src_path):
            skipped_missing_source_file += 1
            continue

        folder_name = _sanitize_folder_name(scientific_name)
        dest_folder = os.path.join(classification_dir, folder_name)
        if folder_name not in species_folders:
            os.makedirs(dest_folder, exist_ok=True)
            species_folders.add(folder_name)

        dest_path = os.path.join(dest_folder, filename)

        if os.path.exists(dest_path) and os.path.getsize(dest_path) == os.path.getsize(src_path):
            files_already_present += 1
        else:
            shutil.copy2(src_path, dest_path)
            files_copied += 1

        species_counts[folder_name] = species_counts.get(folder_name, 0) + 1

    return BuildResult(
        classification_dir=classification_dir,
        species_folders_created=len(species_folders),
        files_copied=files_copied,
        files_already_present=files_already_present,
        skipped_missing_scientific_name=skipped_missing_scientific_name,
        skipped_missing_source_file=skipped_missing_source_file,
        species_counts=species_counts,
    )