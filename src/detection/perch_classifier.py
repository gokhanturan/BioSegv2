"""
src/detection/perch_classifier.py
================
Google DeepMind Perch 2.0 biyoakustik modeli için çıkarım (inference)
sarmalayıcısı.

Bu modül, 2026 Temmuz ayında BioSegv2 (Tkinter tabanlı ayrı bir prototip)
üzerinde gerçek saha kayıtlarıyla doğrulanmış bir dizi düzeltmeyi içerir:

1. **Etiket dosyası başlık satırı düzeltmesi (KRİTİK).** Perch 2.0'ın
   `labels.csv` dosyasının 0. satırı gerçek bir tür adı değil,
   `inat2024_fsd50k` adlı bir BAŞLIK hücresidir. Dosya `header=None` ile
   okunursa bu başlık veri sanılır, etiket listesi modelin gerçek sınıf
   sayısından bir fazla çıkar ve her tür adı modelin ürettiği sınıf
   indeksinden BİR KAYDIRILMIŞ okunur (off-by-one hatası) — yüksek
   güvenle ama sistematik olarak YANLIŞ tür adları raporlanır. Bu modül
   dosyayı pandas'ın varsayılan başlık algılamasıyla okur.
2. **Model çözümlenmiş klasöründen etiket arama.** Etiketler, TF-Hub
   önbelleğinin tamamı yerine `hub.resolve()` ile bulunan, o an yüklenen
   modele özgü klasörden alınır — aksi halde önbellekte birikmiş başka
   bir modele ait bir `labels.csv` yanlışlıkla kullanılabilir.
3. **FSD50K tür-olmayan kategori filtresi.** ~14.795 sınıfın ~198'i
   FSD50K'den gelen genel ses olayı kategorisidir (Music, Dog, vb.).
   Bunlar, FSD50K'nin çok-kelimeli sınıf adlarında hep alt çizgi
   kullanma kuralına dayanarak ayırt edilir.
4. **Kısa kayıtlarda otomatik pencere-tekrarı gevşetmesi.** Kayıt 5 s'lik
   pencereden kısaysa yalnızca 1 pencere üretilir; "min. pencere tekrarı"
   gereksinimi bu durumda otomatik olarak pencere sayısını aşmayacak
   şekilde düşürülür (aksi halde kısa-ama-geçerli kayıtlar için sonuç
   her zaman sıfır çıkardı).

ÖNEMLİ BİLİMSEL UYARI: Perch 2.0'ın çıkışları kalibre edilmemiştir
(model kartı, Google DeepMind, 2025). Güven skorları sık sık %100'e
doygun (saturate) olabilir; bu, modelin eğitim hedefinin ("bir
pencerede bulunan HER sesi yüksek güvenle işaretle") doğal bir
sonucudur, bir hata değildir.
"""

from __future__ import annotations

import glob
import logging
import os
import tempfile
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("perch_classifier")

PERCH_SAMPLE_RATE = 32_000
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(PERCH_SAMPLE_RATE * WINDOW_SECONDS)
EMBEDDING_DIM = 1536  # Perch 2.0'ın ara temsil (embedding) boyutu

KAGGLE_MODEL_URL_CPU = (
    "https://www.kaggle.com/models/google/bird-vocalization-classifier/"
    "frameworks/TensorFlow2/variations/perch_v2_cpu/versions/1"
)
KAGGLE_MODEL_URL_GPU = (
    "https://www.kaggle.com/models/google/bird-vocalization-classifier/"
    "frameworks/TensorFlow2/variations/perch_v2/versions/1"
)


class PerchModelLoadError(RuntimeError):
    """Model indirilemediğinde ya da yüklenemediğinde fırlatılır."""


LOCAL_MODEL_ENV = "PERCH_MODEL_DIR"
LOCAL_MODEL_FOLDER = "perch_model"   # <uygulama klasörü>/perch_model/ varsa ve içinde model varsa o kullanılır


def _is_saved_model_dir(path: str) -> bool:
    return os.path.isdir(path) and any(
        os.path.isfile(os.path.join(path, n)) for n in ("saved_model.pb", "saved_model.pbtxt"))


KAGGLE_HANDLE_CPU = "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"
KAGGLE_HANDLE_GPU = "google/bird-vocalization-classifier/tensorFlow2/perch_v2/1"
HANDLE_ENV = "PERCH_MODEL_HANDLE"   # örn. google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/2


def _find_saved_model_dir(path: str) -> Optional[str]:
    if _is_saved_model_dir(path):
        return path
    for dirpath, _dirs, files in os.walk(path):
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            return dirpath
    return None


def resolve_model_location(use_cpu_variant: bool = True) -> str:
    """Modelin diskteki klasörü. Sırayla: elle konan klasör (PERCH_MODEL_DIR / perch_model/), yoksa `kagglehub` ile
    Kaggle'dan indirme (ilerleme çubuğu konsolda görünür; önbelleğe alınır, sonraki çalıştırmalarda indirme olmaz).
    `kagglehub` kurulu değilse ya da indirme başarısızsa, sessizce takılmak yerine açık bir hata verir."""
    local = find_local_model_dir()
    if local:
        return local
    handle = os.environ.get(HANDLE_ENV, "").strip() or (KAGGLE_HANDLE_CPU if use_cpu_variant else KAGGLE_HANDLE_GPU)
    try:
        import kagglehub
    except ModuleNotFoundError as exc:
        raise PerchModelLoadError(
            "Perch 2.0'ı otomatik indirmek için 'kagglehub' paketi gerekli. Kurmak için:\n"
            "  pip install kagglehub\n"
            f"Orijinal hata: {exc}") from exc
    logger.info("Perch 2.0 modeli kagglehub ile indiriliyor/çözümleniyor: %s", handle)
    try:
        path = kagglehub.model_download(handle)
    except Exception as exc:  # noqa: BLE001
        raise PerchModelLoadError(
            f"Model indirilemedi ({handle}).\n"
            "İnternet bağlantınızı kontrol edin. Gerekirse Kaggle kimlik bilgilerini tanımlayın "
            "(KAGGLE_USERNAME / KAGGLE_KEY ya da ~/.kaggle/kaggle.json). Ağ kısıtlıysa modeli tarayıcıdan indirip "
            f"{LOCAL_MODEL_FOLDER}/ klasörüne açabilirsiniz.\n"
            f"Orijinal hata: {exc}") from exc
    found = _find_saved_model_dir(str(path))
    if not found:
        raise PerchModelLoadError(f"İndirilen klasörde saved_model.pb bulunamadı: {path}")
    return found


def find_local_model_dir() -> Optional[str]:
    """
    İnternetten indirmeye alternatif: modeli elle indirip açtığınız KLASÖR (içinde saved_model.pb olan).
    Sırayla: PERCH_MODEL_DIR ortam değişkeni, sonra `<uygulama klasörü>/perch_model/`.
    Ortam değişkeni verilmiş ama klasör geçersizse sessizce internete düşmek yerine hata verir.
    """
    env = os.environ.get(LOCAL_MODEL_ENV, "").strip()
    if env:
        if not _is_saved_model_dir(env):
            raise PerchModelLoadError(
                f"{LOCAL_MODEL_ENV}={env} geçerli bir model klasörü değil: içinde saved_model.pb bulunamadı. "
                "Arşivi açtıktan sonra saved_model.pb dosyasının bulunduğu klasörü gösterin.")
        return env
    from ..data.config import get_project_root
    default = os.path.join(get_project_root(), LOCAL_MODEL_FOLDER)
    return default if _is_saved_model_dir(default) else None


def is_probable_species(label: str) -> bool:
    """
    Bir etiketin gerçek bir tür (bilimsel binom isim) mi, yoksa Perch
    2.0'ın eğitim verisine dahil edilen FSD50K genel ses olayları
    kategorisinden mi geldiğini ayırt eder.

    Gerekçe: FSD50K resmi belgelerinde çok kelimeli sınıf adlarında
    boşluk yerine alt çizgi kullandığını belirtir. iNaturalist taksonomi
    tabanlı tür adları ise her zaman boşluklu, iki (veya üç) kelimelik
    bilimsel binom formatındadır.
    """
    return (" " in label) and ("_" not in label)


class PerchClassifier:
    """Perch 2.0 modeli için çıkarım sarmalayıcısı (tembel/lazy yükleme)."""

    def __init__(self, use_cpu_variant: bool = True):
        self._use_cpu_variant = use_cpu_variant
        self._module = None
        self._tf = None
        self._signature = None
        self._input_key: Optional[str] = None
        self._output_key: Optional[str] = None
        self.labels: Optional[Sequence[str]] = None

    # ------------------------------------------------------------------
    def ensure_loaded(self) -> None:
        """Model henüz yüklenmemişse yükler. Zaten yüklüyse hiçbir şey yapmaz."""
        if self._module is not None:
            return

        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ModuleNotFoundError as exc:
            raise PerchModelLoadError(
                "tensorflow / tensorflow_hub kurulu değil. Kurmak için:\n"
                "  pip install tensorflow tensorflow-hub \"setuptools<82\"\n"
                "(TensorFlow >= 2.20.0rc0 gereklidir).\n"
                f"Orijinal hata: {exc}"
            ) from exc
        except ImportError as exc:
            # ÖNEMLİ: tensorflow KURULU olabilir ama native DLL'i
            # yüklenemiyor olabilir (örn. Windows'ta msvcp140.dll
            # çakışması). Bu durumu "kurulu değil" diye yanlış
            # etiketlememek için gerçek alt hatayı olduğu gibi gösteriyoruz.
            raise PerchModelLoadError(
                "tensorflow / tensorflow_hub İÇE AKTARILAMADI (kurulu olabilir "
                "ama native DLL yüklenemiyor olabilir — örn. Windows'ta eski/"
                "çakışan bir msvcp140.dll). Sistem PATH'inizde Anaconda gibi "
                "başka bir Python dağıtımının klasörleri varsa, bu bir DLL "
                "çakışmasına yol açabilir.\n\n"
                f"Orijinal hata: {exc}"
            ) from exc

        self._tf = tf
        model_url = resolve_model_location(self._use_cpu_variant)
        logger.info("Perch 2.0 modeli yükleniyor: %s", model_url)
        try:
            self._module = hub.load(model_url)
        except Exception as exc:  # noqa: BLE001
            raise PerchModelLoadError(
                f"Model yüklenemedi ({model_url}).\n"
                "İnternet bağlantınızı ve Kaggle kimlik bilgilerinizi "
                "(KAGGLE_USERNAME / KAGGLE_KEY ya da ~/.kaggle/kaggle.json "
                "— 'Legacy API Key' formatında) kontrol edin.\n"
                f"Orijinal hata: {exc}"
            ) from exc

        self._signature = self._module.signatures.get(
            "serving_default", next(iter(self._module.signatures.values()))
        )
        self._input_key = list(self._signature.structured_input_signature[1].keys())[0]
        self.labels = self._load_labels(model_url)

        if self.labels is None:
            logger.warning(
                "Tür etiket listesi (labels.csv) otomatik bulunamadı; "
                "sonuçlar sınıf indeksi olarak raporlanacak."
            )
        else:
            n = len(self.labels)
            if not (10_000 <= n <= 20_000):
                logger.warning(
                    "DİKKAT: Etiket dosyası %d satır içeriyor; Perch 2.0 için "
                    "beklenen aralık (~10.000-20.000) dışında. Yanlış bir "
                    "labels.csv yüklenmiş olabilir.",
                    n,
                )

    # ------------------------------------------------------------------
    @staticmethod
    def _read_labels_csv(path: str) -> List[str]:
        import pandas as pd

        # KRİTİK: header=None DEĞİL. Bkz. modül docstring'i (madde 1).
        df = pd.read_csv(path)
        return df.iloc[:, 0].astype(str).tolist()

    def _load_labels(self, model_url: str) -> Optional[List[str]]:
        try:
            import tensorflow_hub as hub

            resolved_dir = hub.resolve(model_url)
            logger.info("Model çözümlenmiş klasörü: %s", resolved_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model klasörü çözümlenemedi (%s); genel arama yapılacak.", exc)
            resolved_dir = None

        candidates: List[str] = []
        if resolved_dir and os.path.isdir(resolved_dir):
            candidates.extend(glob.glob(os.path.join(resolved_dir, "**", "labels.csv"), recursive=True))

        if candidates:
            logger.info("Etiket dosyası (modele özgü) bulundu: %s", candidates[0])
            return self._read_labels_csv(candidates[0])

        # Yedek: geniş önbellek araması (doğruluğu garanti edilmez)
        search_bases = [
            os.environ.get("TFHUB_CACHE_DIR", ""),
            os.path.join(tempfile.gettempdir(), "tfhub_modules"),
            os.path.expanduser("~/.cache/kagglehub"),
            os.path.expanduser("~/.cache/tfhub_modules"),
        ]
        for base in search_bases:
            if base and os.path.isdir(base):
                candidates.extend(glob.glob(os.path.join(base, "**", "labels.csv"), recursive=True))
        if candidates:
            logger.info("Etiket dosyası bulundu: %s", candidates[0])
            return self._read_labels_csv(candidates[0])

        return None

    # ------------------------------------------------------------------
    def predict_window(self, waveform_5s: np.ndarray) -> np.ndarray:
        """
        5 saniyelik (WINDOW_SAMPLES örnek), 32 kHz, mono, float32 dalga formu
        için her sınıf için bağımsız SIGMOID olasılığını döndürür.
        """
        self.ensure_loaded()

        if waveform_5s.shape[0] != WINDOW_SAMPLES:
            raise ValueError(
                f"Beklenen pencere uzunluğu {WINDOW_SAMPLES} örnek "
                f"(5 s @ {PERCH_SAMPLE_RATE} Hz), alınan: {waveform_5s.shape[0]}"
            )

        x = self._tf.convert_to_tensor(waveform_5s[np.newaxis, :].astype(np.float32))
        out = self._signature(**{self._input_key: x})

        if self._output_key is None:
            n_labels = len(self.labels) if self.labels else None
            best_key = None
            if "label" in out:
                best_key = "label"
            else:
                for k, v in out.items():
                    if n_labels is not None and v.shape[-1] == n_labels:
                        best_key = k
                        break
            self._output_key = best_key or next(iter(out.keys()))

            output_dim = out[self._output_key].shape[-1]
            if n_labels is not None and n_labels != output_dim:
                logger.error(
                    "UYUŞMAZLIK: Etiket sayısı (%d) modelin çıkış boyutuyla (%d) "
                    "eşleşmiyor! Tür isimleri yanlış hizalanmış olabilir.",
                    n_labels, output_dim,
                )

        logits = out[self._output_key].numpy()[0]
        return 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    def predict_window_embedding(self, waveform_5s: np.ndarray) -> np.ndarray:
        """
        5 saniyelik bir pencere için, tür sınıflandırma başlığını (label)
        DEĞİL, modelin ~1536 boyutlu ARA TEMSİLİNİ (embedding) döndürür.

        Bu, kendi özel sınıflandırıcınızı eğitmek için kullanılır (bkz.
        `src/ml/`) — Perch'in hazır 14.795 sınıflık çıkışını değil,
        modelin öğrendiği genel akustik temsili temel alır. Önceki
        çalışmanızda (embedding + kendi MLP'niz) kullandığınız aynı
        yaklaşım.
        """
        self.ensure_loaded()

        if waveform_5s.shape[0] != WINDOW_SAMPLES:
            raise ValueError(
                f"Beklenen pencere uzunluğu {WINDOW_SAMPLES} örnek "
                f"(5 s @ {PERCH_SAMPLE_RATE} Hz), alınan: {waveform_5s.shape[0]}"
            )

        x = self._tf.convert_to_tensor(waveform_5s[np.newaxis, :].astype(np.float32))
        out = self._signature(**{self._input_key: x})

        if "embedding" not in out:
            raise RuntimeError(
                "Model çıktısında 'embedding' anahtarı bulunamadı. "
                f"Mevcut anahtarlar: {list(out.keys())}"
            )

        return out["embedding"].numpy()[0]

    def label_name(self, class_index: int) -> str:
        if self.labels is not None and 0 <= class_index < len(self.labels):
            return self.labels[class_index]
        return f"sinif_{class_index}"


# ------------------------------------------------------------------
# Pencereleme ve segment birleştirme (audio_utils benzeri yardımcılar)
# ------------------------------------------------------------------
def generate_windows(
    y: np.ndarray, sr: int, window_seconds: float = WINDOW_SECONDS, hop_seconds: float = 2.5
) -> List[Tuple[float, float, np.ndarray]]:
    """Ses sinyalini sabit uzunlukta, kısmen örtüşen pencerelere böler."""
    window_samples = int(sr * window_seconds)
    hop_samples = max(1, int(sr * hop_seconds))
    n = len(y)
    if n == 0:
        return []

    starts = list(range(0, max(n - window_samples, 0) + 1, hop_samples))
    if not starts or starts[-1] + window_samples < n:
        last_start = max(n - window_samples, 0)
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

    windows = []
    for s in starts:
        e = s + window_samples
        chunk = y[s:e]
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))
        windows.append((s / sr, e / sr, chunk))
    return windows


def merge_detections(
    detections: List[Tuple[float, float, str, float]],
    max_gap_seconds: float = 1.0,
    min_window_hits: int = 1,
) -> List[Tuple[float, float, str, float]]:
    """
    Aynı türe ait, aralarında max_gap_seconds'tan az boşluk bulunan ardışık
    pencere tespitlerini tek bir zaman segmentinde birleştirir. Bir türün
    raporlanması için en az min_window_hits bağımsız pencerede görünmüş
    olması gerekir (izole yanlış pozitifleri elemek için).
    """
    by_species: dict = {}
    for start, end, species, conf in detections:
        by_species.setdefault(species, []).append((start, end, conf))

    merged: List[Tuple[float, float, str, float]] = []
    for species, segs in by_species.items():
        segs.sort(key=lambda t: t[0])
        cur_start, cur_end, confs = segs[0][0], segs[0][1], [segs[0][2]]
        for s, e, c in segs[1:]:
            if s - cur_end <= max_gap_seconds:
                cur_end = max(cur_end, e)
                confs.append(c)
            else:
                if len(confs) >= min_window_hits:
                    merged.append((cur_start, cur_end, species, float(np.mean(confs))))
                cur_start, cur_end, confs = s, e, [c]
        if len(confs) >= min_window_hits:
            merged.append((cur_start, cur_end, species, float(np.mean(confs))))

    merged.sort(key=lambda x: x[0])
    return merged


def classify_recording(
    classifier: PerchClassifier,
    y: np.ndarray,
    sr: int,
    threshold: float = 0.5,
    hop_seconds: float = 2.5,
    top_k: int = 1,
    min_window_hits: int = 2,
    progress_callback=None,
) -> List[Tuple[float, float, str, float]]:
    """
    Bir kaydın tamamını Perch 2.0 ile tarar ve birleştirilmiş
    (start_s, end_s, tür_adı, güven_yüzde) listesini döndürür.

    Kısa kayıt düzeltmesi: toplam pencere sayısı min_window_hits'ten azsa,
    etkin eşik otomatik olarak pencere sayısını aşmayacak şekilde düşürülür
    (aksi halde kısa-ama-geçerli kayıtlar için sonuç her zaman sıfır çıkar).
    """
    windows = generate_windows(y, sr, hop_seconds=hop_seconds)
    n_windows = len(windows)
    effective_min_hits = min(min_window_hits, n_windows) if n_windows > 0 else min_window_hits

    raw_detections: List[Tuple[float, float, str, float]] = []
    for i, (start_s, end_s, chunk) in enumerate(windows):
        probs = classifier.predict_window(chunk)
        top_idx = np.argsort(probs)[::-1][:top_k]
        for idx in top_idx:
            conf = float(probs[idx])
            if conf >= threshold:
                species = classifier.label_name(int(idx))
                raw_detections.append((start_s, end_s, species, conf * 100.0))
        if progress_callback is not None:
            progress_callback(i + 1, n_windows)

    return merge_detections(raw_detections, max_gap_seconds=hop_seconds, min_window_hits=effective_min_hits)