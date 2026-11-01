"""
Sınıflandırma modelleri kayıt defteri.

Her model kendi sonuç klasörüne yazar; böylece bir modelin eğitimi ötekilerin kayıtlı sonucunu, eğitilmiş
modelini ve rapor dosyalarını ezmez. BirdNET mevcut dosya konumlarını korur (geriye uyumluluk); diğer
modeller `modeller/<id>/` altında tutulur.

Yeni bir model eklemek için `MODELS` sözlüğüne bir `ModelSpec` eklemek yeterlidir.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import last_result_store

MODELS_DIR_NAME = "modeller"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    ad: str                       # arayüzde görünen ad
    backend: str                  # pipeline.EMBEDDING_BACKENDS içinden
    gomme_boyutu: int
    girdi_notu: str               # pencerenin bilgi kutusunda gösterilen girdi/yöntem notu
    alt_klasor: Optional[str] = None      # None: proje kökü (BirdNET, geriye uyumluluk)
    gomme_onbellegi: Optional[str] = None  # None: önbellek kullanılmaz (mevcut BirdNET davranışı)
    gomme_okuma_onbellekleri: tuple = ()   # proje köküne göre, yalnızca OKUNAN eski önbellekler (kaynak etiketi eşleşirse)
    egitici: str = "mlp"                  # yalnızca "mlp": dondurulmuş gömme + MLP başlığı
    altyazi: Optional[str] = None         # pencere başlığının altındaki satır (None: "<ad> gömmeleri üzerinde ...")


MODELS: Dict[str, ModelSpec] = {
    "birdnet": ModelSpec(
        id="birdnet", ad="BirdNET 2.4", backend="birdnet", gomme_boyutu=1024,
        girdi_notu="BirdNET 2.4 (dondurulmuş) ile her segmentin 1024 boyutlu akustik gömmesi çıkarılır "
                   "(3 sn pencere, 48 kHz).",
    ),
    "perch": ModelSpec(
        id="perch", ad="Perch 2.0", backend="perch", gomme_boyutu=1536,
        girdi_notu="Perch 2.0 (dondurulmuş) ile her segmentin 1536 boyutlu akustik gömmesi çıkarılır (32 kHz). "
                   "Perch 5 sn pencere bekler; 3 sn'lik segmentler <b>sıfırla 5 sn'ye tamamlanır</b> "
                   "(modelin belirtimi dışı kullanım; sonuçlar bu notla birlikte yorumlanmalıdır).",
        alt_klasor=f"{MODELS_DIR_NAME}/perch", gomme_onbellegi="perch_gomme_onbellegi.npz",
        gomme_okuma_onbellekleri=("embedding_onbellek.npz",),
    ),
    "birlesik_birdnet_perch": ModelSpec(
        id="birlesik_birdnet_perch", ad="BirdNET + Perch (birleşik)", backend="combined", gomme_boyutu=1024 + 1536,
        girdi_notu="Her segment için BirdNET 2.4 (1024) ve Perch 2.0 (1536) gömmeleri <b>yan yana eklenir</b> "
                   "(2560 boyut; erken birleştirme) ve tek-model sonuçlarıyla AYNI küçük MLP başlığı eğitilir. Ham "
                   "gömmeler eklenir, veri kümesinden öğrenilen hiçbir ölçek kullanılmaz; tek-model sonuçlarından "
                   "tek fark birleştirmedir. Her iki gömme de gerekir; ikisinin önbelleği varsa model yüklenmez. "
                   "Perch'in 3 sn segmentlerde sıfır dolguyla çalışması bu modele de yansır. "
                   "<b>Hipotez (sonuçlara bakmadan önce):</b> birleşik gömme, tek gömmeden daha yüksek makro F1 verir; "
                   "eşleştirilmiş testle sınanır, fark çıkmayabilir.",
        alt_klasor=f"{MODELS_DIR_NAME}/birlesik_birdnet_perch", gomme_onbellegi=None,
        altyazi="BirdNET 2.4 ve Perch 2.0 gömmelerinin birleşimi üzerinde sınıflandırıcı eğitimi",
    ),
}

DEFAULT_MODEL_ID = "birdnet"


def get_model(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError:
        raise ValueError(f"Bilinmeyen model: {model_id!r} (geçerli: {list(MODELS)})") from None


def result_dir(model_id: str, root: str) -> str:
    """Modelin kayıtlı sonuç ve rapor dosyalarının klasörü (yoksa oluşturulur)."""
    spec = get_model(model_id)
    folder = root if spec.alt_klasor is None else os.path.join(root, *spec.alt_klasor.split("/"))
    os.makedirs(folder, exist_ok=True)
    return folder


def model_save_dir(model_id: str, root: str) -> Optional[str]:
    """Eğitilmiş model klasörü; None = pipeline'ın varsayılanı (yalnızca BirdNET)."""
    spec = get_model(model_id)
    if spec.alt_klasor is None:
        return None
    return os.path.join(result_dir(model_id, root), "model")


def embedding_cache_path(model_id: str, root: str) -> Optional[str]:
    spec = get_model(model_id)
    if spec.gomme_onbellegi is None:
        return None
    return os.path.join(result_dir(model_id, root), spec.gomme_onbellegi)


def embedding_read_caches(model_id: str, root: str) -> tuple:
    return tuple(os.path.join(root, f) for f in get_model(model_id).gomme_okuma_onbellekleri)


def component_cache_paths(model_id: str, root: str) -> dict:
    """Birleşik modelin bileşen önbellekleri: {parça: (yazılacak_önbellek, yalnızca_okunur_önbellekler)}.
    Perch bileşeni Perch modelinin KENDİ önbelleğini (ve eski araştırma önbelleğini) paylaşır; BirdNET'in
    kayıt defterinde önbelleği olmadığından kendi klasöründe bir önbellek tutulur. Birleşik olmayan modeller: {}."""
    if get_model(model_id).backend != "combined":
        return {}
    return {
        "birdnet": (os.path.join(result_dir(model_id, root), "birdnet_gomme_onbellegi.npz"), ()),
        "perch": (embedding_cache_path("perch", root), embedding_read_caches("perch", root)),
    }


def stored_results(root: str) -> List[tuple]:
    """[(ModelSpec, StoredResult)] — kayıtlı sonucu olan modeller, kayıt sırasıyla."""
    out = []
    for spec in MODELS.values():
        folder = root if spec.alt_klasor is None else os.path.join(root, *spec.alt_klasor.split("/"))
        stored = last_result_store.load_last_result(folder)
        if stored is not None:
            out.append((spec, stored))
    return out


def download_cache_dirs(model_id: str) -> List[str]:
    """Modelin ilk indirmede yazıldığı önbellek klasörleri (yalnızca Perch; BirdNET paketi kendi klasörünü yönetir)."""
    if model_id != "perch":
        return []
    dirs = [os.environ.get("TFHUB_CACHE_DIR", ""),
            os.path.join(tempfile.gettempdir(), "tfhub_modules"),
            os.path.expanduser("~/.cache/kagglehub"),
            os.path.expanduser("~/.cache/tfhub_modules")]
    seen, out = set(), []
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def dir_size_mb(dirs: List[str]) -> float:
    """Klasörlerin toplam dosya boyutu (MB). Okunamayan dosyalar yok sayılır."""
    total = 0
    for base in dirs:
        for dirpath, _dirs, files in os.walk(base):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
    return total / (1024 * 1024)
