"""
src/ml/birdnet_embedding.py
===========================
BirdNET 2.4 (`birdnet` paketi) ile 1024 boyutlu gömme çıkarma. Uygulamanın sınıflandırıcı eğitimi,
çapraz doğrulama ve tür zenginliği taraması bu modülü kullanır.

Segmentler BirdNET'in 3 sn'lik, örtüşmeyen pencereleridir; her segment tek pencere olarak gömülür
(3 sn'den kısa olanlar sıfırla tamamlanır). Uzun kayıtlar 3 sn'lik pencerelere bölünerek taranır.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Callable, List, Optional, Tuple

import numpy as np

from .dataset_builder import TrainingSample
from .embedding_comparison import BIRDNET_SR, BIRDNET_WINDOW_S, pool_birdnet_result

logger = logging.getLogger("birdnet_embedding")

BACKEND = "birdnet"
EMBEDDING_DIM = 1024
DEFAULT_LIBRARY = "litert"


class BirdNETModelLoadError(RuntimeError):
    """BirdNET gömme modeli yüklenemediğinde fırlatılır."""


_models: dict = {}
ENV_APP_DATA = "BIRDNET_APP_DATA"


def _is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _default_birdnet_dir() -> str:
    base = os.environ.get("APPDATA") if os.name == "nt" else None
    return os.path.join(base or os.path.expanduser("~/.local/share"), "birdnet")


def configure_model_dir() -> Optional[str]:
    """TFLite/LiteRT, Windows'ta ASCII olmayan karakter içeren yollardaki (ör. "C:\\Users\\GökhanTURAN\\...")
    model dosyalarını açamaz ("Could not open ... model allocation is null/empty"). Varsayılan BirdNET veri
    klasörü böyle bir yola düşüyorsa, BirdNET'in model klasörü ASCII bir yola yönlendirilir
    (BIRDNET_APP_DATA); alt süreçler ortam değişkenini devralır. Kullanıcı kendisi ayarladıysa dokunulmaz.
    Döndürür: yönlendirilen klasör ya da None."""
    if os.environ.get(ENV_APP_DATA):
        return None
    if _is_ascii(_default_birdnet_dir()):
        return None
    candidates = [os.path.join(_project_root(), "birdnet_models")]
    if os.name == "nt":
        candidates += [os.path.join(os.environ.get("PUBLIC", r"C:\Users\Public"), "BioSeg", "birdnet"),
                       os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"), "BioSeg", "birdnet")]
    for c in candidates:
        if not _is_ascii(c):
            continue
        try:
            os.makedirs(c, exist_ok=True)
        except OSError:
            continue
        os.environ[ENV_APP_DATA] = c
        logger.info("BirdNET model klasörü ASCII yola yönlendirildi: %s", c)
        return c
    return None


def get_birdnet_model(library: str = DEFAULT_LIBRARY):
    """BirdNET 2.4 modelini (ilk çağrıda) yükler ve bellekte tutar."""
    if library not in _models:
        configure_model_dir()
        try:
            import birdnet
        except ImportError as exc:
            raise BirdNETModelLoadError(
                "BirdNET gömmeleri için 'birdnet' paketi gerekir:\n  pip install birdnet") from exc
        try:
            _models[library] = birdnet.load("acoustic", "2.4", "tf", library=library)
        except Exception as exc:  # noqa: BLE001
            raise BirdNETModelLoadError(str(exc)) from exc
    return _models[library]


def _with_log_tail(message: str, n_lines: int = 12) -> str:
    """BirdNET hata iletisi bir oturum günlüğüne işaret ediyorsa (".../birdnet_session_xxx.log"),
    günlüğün son satırlarını iletiye ekler; gerçek neden orada yazar."""
    import re
    m = re.search(r"logs:\s*(.+?\.log)", message.replace("\n", " "))
    if not m:
        return message
    try:
        with open(m.group(1).strip(), "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f.readlines() if ln.strip()]
        return message + "\n--- günlüğün son satırları ---\n" + "\n".join(lines[-n_lines:])
    except OSError:
        return message


def _fit_window(y: np.ndarray) -> np.ndarray:
    min_len = int(BIRDNET_SR * BIRDNET_WINDOW_S)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))
    return y.astype(np.float32)


def extract_embeddings_for_samples(
    samples: List[TrainingSample],
    model=None,
    library: str = DEFAULT_LIBRARY,
    chunk_size: int = 32,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    stage_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[np.ndarray, List[TrainingSample], int, Optional[str]]:
    """Örnek listesinden (N, 1024) gömme matrisi ve başarıyla işlenen örnekler.
    Okunamayan/boş dosyalar atlanır; dönen `kept_samples`, X satırlarıyla birebir eşleşir.
    Dönüş: (X, kept_samples, n_failures, last_error).

    `model` verilmezse çıkarma AYRI bir Python sürecinde (birdnet_worker) yapılır: `birdnet` kendi alt
    süreçlerini başlatır ve bunlar ana modülü yeniden içe aktarır; arayüz uygulamasında ana modül tüm
    arayüzü yüklediği için alt süreçler çöker. `stage_callback("model", a, b)` model yükleme aşamasını bildirir."""
    if model is None:
        return _extract_in_subprocess(samples, library, chunk_size, progress_callback, stage_callback)
    import librosa

    chunk_size = max(1, int(chunk_size))
    total = len(samples)
    X: List[np.ndarray] = []
    kept: List[TrainingSample] = []
    n_failures, last_error = 0, None

    for start in range(0, total, chunk_size):
        loaded = []
        for s in samples[start:start + chunk_size]:
            try:
                y, _ = librosa.load(s.filepath, sr=BIRDNET_SR, mono=True)
                if len(y) == 0:
                    raise ValueError(f"Ses dosyası boş: {s.filepath}")
                loaded.append((s, _fit_window(y)))
            except Exception as exc:  # noqa: BLE001
                n_failures += 1
                last_error = str(exc)
                logger.warning("Ses okunamadı (%s): %s", s.filepath, exc)
        if loaded:
            try:
                res = model.encode_arrays([(y, BIRDNET_SR) for _, y in loaded])
                for i, (s, _) in enumerate(loaded):
                    X.append(pool_birdnet_result(res, index=i))
                    kept.append(s)
            except Exception as exc:  # noqa: BLE001
                n_failures += len(loaded)
                last_error = _with_log_tail(str(exc))
                logger.warning("BirdNET gömmesi çıkarılamadı (%d dosya): %s", len(loaded), exc)
        if progress_callback is not None:
            progress_callback(min(start + chunk_size, total), total)

    X_arr = np.vstack(X).astype(np.float32) if X else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return X_arr, kept, n_failures, last_error


class BirdNETEmbedder:
    """Uzun bir kaydı BirdNET'in 3 sn'lik pencerelerine bölüp her pencerenin gömmesini verir
    (tür zenginliği taraması için; `richness_inference.scan_recording` bu arayüzü kullanır)."""

    cache_tag = BACKEND
    default_hop = BIRDNET_WINDOW_S

    def __init__(self, model=None, library: str = DEFAULT_LIBRARY, batch_size: int = 64):
        self._model = model
        self.library = library
        self.batch_size = max(1, int(batch_size))

    @property
    def model(self):
        if self._model is None:
            self._model = get_birdnet_model(self.library)
        return self._model

    def embed_recording(self, filepath: str, hop: float = BIRDNET_WINDOW_S) -> Tuple[np.ndarray, np.ndarray]:
        import librosa

        y, _ = librosa.load(filepath, sr=BIRDNET_SR, mono=True)
        if len(y) == 0:
            raise ValueError(f"Ses dosyası boş: {filepath}")
        win = int(BIRDNET_SR * BIRDNET_WINDOW_S)
        step = max(1, int(round(BIRDNET_SR * float(hop))))
        starts_s = list(range(0, max(len(y) - win, 0) + 1, step))
        if not starts_s:
            starts_s = [0]
        chunks = [_fit_window(y[s:s + win]) for s in starts_s]
        rows = []
        for b in range(0, len(chunks), self.batch_size):
            part = chunks[b:b + self.batch_size]
            res = self.model.encode_arrays([(c, BIRDNET_SR) for c in part])
            rows.extend(pool_birdnet_result(res, index=i) for i in range(len(part)))
        return (np.array(starts_s, dtype=float) / BIRDNET_SR), np.vstack(rows).astype(np.float32)


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _extract_in_subprocess(samples, library, chunk_size, progress_callback, stage_callback):
    import json
    import subprocess
    import sys
    import tempfile

    if getattr(sys, "frozen", False):          # paketlenmiş uygulamada ayrı yorumlayıcı yok
        model = get_birdnet_model(library)
        if stage_callback:
            stage_callback("model", 1, 1)
        return extract_embeddings_for_samples(samples, model=model, chunk_size=chunk_size,
                                              progress_callback=progress_callback)

    with tempfile.TemporaryDirectory(prefix="bioseg_birdnet_", ignore_cleanup_errors=True) as tmp:
        in_json, out_npz = os.path.join(tmp, "girdi.json"), os.path.join(tmp, "cikti.npz")
        with open(in_json, "w", encoding="utf-8") as f:
            json.dump([s.filepath for s in samples], f)
        root = _project_root()
        configure_model_dir()
        env = dict(os.environ)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.ml.birdnet_worker", in_json, out_npz, library, str(int(chunk_size))],
            cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
        err, tail = None, []
        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith("@@PROG ") and progress_callback:
                _, c, t = line.split()
                progress_callback(int(c), int(t))
            elif line.startswith("@@MODEL ") and stage_callback:
                _, c, t = line.split()
                stage_callback("model", int(c), int(t))
            elif line.startswith("@@ERR "):
                err = line[6:]
            elif line:
                tail.append(line)
                tail = tail[-8:]
        code = proc.wait()
        if code != 0 or not os.path.isfile(out_npz):
            if err:
                raise BirdNETModelLoadError(err)
            raise BirdNETModelLoadError(
                f"BirdNET işlemi beklenmedik biçimde sonlandı (çıkış kodu {code}).\n" + "\n".join(tail[-5:]))
        with np.load(out_npz, allow_pickle=False) as data:      # Windows: açık dosya silinemez
            X, ok = data["X"].astype(np.float32), data["ok"].copy()
            n_fail, last_error = int(data["n_fail"]), str(data["last_error"])
        kept = [s for s, k in zip(samples, ok) if k]
        return X, kept, n_fail, last_error or None
