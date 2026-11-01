"""
Tür Sınıflandırma Eğitimi penceresinin SON sonucunu diske kaydeder ve geri yükler.

Amaç: pencere her açıldığında eğitimi (dakikalar süren gömme çıkarımı + çapraz doğrulama) yeniden yapmadan
en son sonucu, ne zaman üretildiğiyle birlikte göstermek. Kayıt tek bir JSON dosyasıdır; model ağırlıkları
zaten `trained_classifier/` klasöründe durur, burada yalnızca GÖSTERİLEN sonuç saklanır.

Kayıtlı sonuç, veri seti sonradan değişmişse bunu anlayabilsin diye eğitim anındaki veri izini
(`classification_diagnosis.data_fingerprint`) de taşır.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

FILE_NAME = "son_siniflandirma_sonucu.json"
FORMAT_VERSION = 1


@dataclass
class StoredResult:
    result: object                      # pipeline.PipelineResult (history=None)
    details: str                        # "Ayrıntılar" sekmesi metni
    saved_at: _dt.datetime
    settings: dict = field(default_factory=dict)
    fingerprint: Optional[dict] = None


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"JSON'a çevrilemeyen tür: {type(o).__name__}")


def format_time(dt: _dt.datetime) -> str:
    return dt.strftime("%d.%m.%Y %H:%M:%S")


def save_last_result(folder: str, pipeline_result, details: str, settings: Optional[dict] = None,
                     fingerprint: Optional[dict] = None, when: Optional[_dt.datetime] = None) -> str:
    """Sonucu `folder/son_siniflandirma_sonucu.json` olarak yazar (önce geçici dosya, sonra atomik değiştirme)."""
    tr = pipeline_result.training_result
    when = when or _dt.datetime.now()
    data = {
        "surum": FORMAT_VERSION,
        "zaman": when.isoformat(timespec="seconds"),
        "ayarlar": settings or {},
        "veri_izi": fingerprint,
        "ayrintilar": details,
        "save_dir": pipeline_result.save_dir,
        "raw_species_counts": pipeline_result.raw_species_counts or {},
        "egitim": {
            "test_accuracy": tr.test_accuracy, "test_macro_f1": tr.test_macro_f1,
            "per_class_report": tr.per_class_report,
            "confusion_matrix": np.asarray(tr.confusion_matrix).tolist(),
            "class_names": list(tr.class_names),
            "n_train": tr.n_train, "n_val": tr.n_val, "n_test": tr.n_test,
            "n_train_recordings": tr.n_train_recordings, "n_val_recordings": tr.n_val_recordings,
            "n_test_recordings": tr.n_test_recordings,
            "excluded_species": list(tr.excluded_species or []),
            "test_predictions": tr.test_predictions,
            "cv_summary": tr.cv_summary,
            "ogrenme_egrisi": tr.learning_curve,
            "zenginlik_uyumu": tr.zenginlik_uyumu,
        },
    }
    path = os.path.join(folder, FILE_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=_default)
    os.replace(tmp, path)
    return path


def load_last_result(folder: str) -> Optional[StoredResult]:
    """Kayıtlı sonucu okur; dosya yok / bozuk / sürüm uyumsuzsa None (pencere boş açılır, hata vermez)."""
    path = os.path.join(folder, FILE_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("surum") != FORMAT_VERSION:
            return None
        from .pipeline import PipelineResult
        from .train_classifier import TrainingResult
        e = d["egitim"]
        tr = TrainingResult(
            test_accuracy=float(e["test_accuracy"]), test_macro_f1=float(e["test_macro_f1"]),
            per_class_report=e["per_class_report"], confusion_matrix=np.asarray(e["confusion_matrix"]),
            class_names=list(e["class_names"]),
            n_train=int(e["n_train"]), n_val=int(e["n_val"]), n_test=int(e["n_test"]),
            n_train_recordings=int(e["n_train_recordings"]), n_val_recordings=int(e["n_val_recordings"]),
            n_test_recordings=int(e["n_test_recordings"]),
            excluded_species=list(e.get("excluded_species") or []), history=None,
            test_predictions=e.get("test_predictions"), cv_summary=e.get("cv_summary"),
            learning_curve=e.get("ogrenme_egrisi"), zenginlik_uyumu=e.get("zenginlik_uyumu"))
        res = PipelineResult(training_result=tr, save_dir=d.get("save_dir", ""),
                             raw_species_counts=d.get("raw_species_counts") or {})
        return StoredResult(result=res, details=d.get("ayrintilar", ""),
                            saved_at=_dt.datetime.fromisoformat(d["zaman"]),
                            settings=d.get("ayarlar") or {}, fingerprint=d.get("veri_izi"))
    except (OSError, ValueError, KeyError, TypeError):
        return None


CLEARABLE_FILES = (
    FILE_NAME,
    "son_egitim_sonucu.json",
    "test_segment_tahminleri.csv",
    "test_kayit_bazli.csv",
    "test_tur_varligi.csv",
    "ogrenme_egrisi.csv",
    "ogrenme_egrisi_ozet.csv",
    "zenginlik_ozet.csv",
    "zenginlik_nokta.csv",
)


def clear_saved_results(folder: str) -> list:
    """Son eğitim sonucuna ait kayıt ve rapor dosyalarını siler; silinen dosya adlarını döndürür.
    Eğitilmiş model klasörüne, gömme önbelleğine ve veri setine dokunulmaz."""
    removed = []
    for name in CLEARABLE_FILES:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            os.remove(path)
            removed.append(name)
    return removed
