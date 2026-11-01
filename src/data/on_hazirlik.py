"""Ön Hazırlık sekmesinin arka planı (arayüzden bağımsız, test edilebilir).

- Varsayılan etiket bilgileri: Kaydeden / Ornitolog / Mikrofon / Kaydedici (etiket_varsayilanlari.json)
- Yedekten geri yükleme: Tanım Verileri Yöneticisi'nin `yedek_json/` yedekleri
- Model durumu: BirdNET 2.4 ve Perch 2.0 dosyaları diskte var mı (model YÜKLENMEZ, yalnızca dosyalara bakılır)
- Veri klasörleri: CSV ve segment klasörü, kayıt sayıları
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import shutil
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

DEFAULTS_FILE = "etiket_varsayilanlari.json"
# form alanı (CSV sütunu) -> yapılandırma listesi
DEFAULT_FIELDS: Dict[str, str] = {
    "Recordist": "recordists",
    "Ornithologist": "ornithologists",
    "Microphone": "microphones",
    "Recorder": "recorders",
}
BACKUP_DIR = "yedek_json"
BACKUP_NAMES = {"bird_sound_config": "bird_sound_config.json",
                "bird_species_data": "bird_species_data.json",
                "habitat_data": "habitat_data.json"}


# ------------------------------------------------------------------ varsayılan etiket bilgileri
def load_label_defaults(root: str) -> Dict[str, str]:
    p = os.path.join(root, DEFAULTS_FILE)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: str(data[k]) for k in DEFAULT_FIELDS if isinstance(data.get(k), str) and data[k].strip()}


def save_label_defaults(root: str, values: Dict[str, str]) -> None:
    clean = {k: str(values[k]).strip() for k in DEFAULT_FIELDS if str(values.get(k, "")).strip()}
    fd, tmp = tempfile.mkstemp(prefix=DEFAULTS_FILE + ".", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, os.path.join(root, DEFAULTS_FILE))
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ------------------------------------------------------------------ yedekler
def list_backups(root: str) -> List[dict]:
    """Yedekleri yeniden eskiye sıralı döndürür: {file, path, target, when, size}."""
    folder = os.path.join(root, BACKUP_DIR)
    out = []
    if not os.path.isdir(folder):
        return out
    for f in os.listdir(folder):
        stem, ext = os.path.splitext(f)
        for prefix, target in BACKUP_NAMES.items():
            if f.startswith(prefix + "_") and ext == ".json":
                stamp = stem[len(prefix) + 1:]
                try:
                    when = datetime.strptime(stamp, "%Y%m%d_%H%M%S_%f")
                except ValueError:
                    continue
                path = os.path.join(folder, f)
                out.append({"file": f, "path": path, "target": target, "when": when,
                            "size": os.path.getsize(path)})
                break
    out.sort(key=lambda d: d["when"], reverse=True)
    return out


def restore_backup(root: str, backup_path: str) -> str:
    """Yedeği asıl dosyanın yerine koyar. Önce geçerli dosyanın yedeği alınır (geri alınabilsin).
    Geçersiz JSON ise hiçbir şey değiştirilmez. Döndürür: geri yüklenen dosya adı."""
    chosen = next((b for b in list_backups(root) if os.path.abspath(b["path"]) == os.path.abspath(backup_path)), None)
    if chosen is None:
        raise ValueError("Bu dosya bir Tanım Verileri Yöneticisi yedeği değil.")
    with open(backup_path, "r", encoding="utf-8") as f:
        json.load(f)  # bozuksa burada hata verir, asıl dosyaya dokunulmaz
    target = os.path.join(root, chosen["target"])
    if os.path.exists(target):
        stem, ext = os.path.splitext(chosen["target"])
        os.makedirs(os.path.join(root, BACKUP_DIR), exist_ok=True)
        shutil.copy2(target, os.path.join(root, BACKUP_DIR, f"{stem}_{datetime.now():%Y%m%d_%H%M%S_%f}{ext}"))
    fd, tmp = tempfile.mkstemp(prefix=chosen["target"] + ".", suffix=".tmp", dir=root)
    os.close(fd)
    try:
        shutil.copyfile(backup_path, tmp)
        os.replace(tmp, target)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return chosen["target"]


# ------------------------------------------------------------------ model durumu
def _any_file(base: str, patterns: Tuple[str, ...]) -> Optional[str]:
    if not base or not os.path.isdir(base):
        return None
    for pat in patterns:
        hits = glob.glob(os.path.join(base, "**", pat), recursive=True)
        if hits:
            return hits[0]
    return None


def perch_status() -> Tuple[bool, str]:
    try:
        from ..detection.perch_classifier import find_local_model_dir
        local = find_local_model_dir()
    except Exception:  # noqa: BLE001  (TensorFlow vb. yoksa bile durum sorgusu çökmesin)
        local = None
    if local:
        return True, f"Model klasörü bulundu: {local}"
    for base in (os.environ.get("TFHUB_CACHE_DIR", ""),
                 os.path.expanduser("~/.cache/kagglehub"),
                 os.path.join(tempfile.gettempdir(), "tfhub_modules"),
                 os.path.expanduser("~/.cache/tfhub_modules")):
        hit = _any_file(base, ("saved_model.pb",))
        if hit:
            return True, f"İndirilmiş model bulundu: {os.path.dirname(hit)}"
    if importlib.util.find_spec("kagglehub") is None:
        return False, "Model yok ve 'kagglehub' kurulu değil; ilk kullanımda indirilemez."
    return False, "Model henüz indirilmemiş; ilk eğitimde internetten indirilir."


def birdnet_status() -> Tuple[bool, str]:
    if importlib.util.find_spec("birdnet") is None:
        return False, "'birdnet' paketi kurulu değil (pip install birdnet)."
    try:
        from ..ml.birdnet_embedding import ENV_APP_DATA, _default_birdnet_dir
        base = os.environ.get(ENV_APP_DATA) or _default_birdnet_dir()
    except Exception:  # noqa: BLE001
        base = os.path.join(os.path.expanduser("~/.local/share"), "birdnet")
    hit = _any_file(base, ("*.tflite", "*.keras", "*.pb", "*.h5"))
    if hit:
        return True, f"Paket kurulu, model dosyası bulundu: {os.path.dirname(hit)}"
    return False, "Paket kurulu; model dosyası henüz indirilmemiş (ilk kullanımda indirilir)."


def model_status() -> List[Tuple[str, bool, str]]:
    """[(ad, hazır_mı, açıklama)] — BirdNET + Perch, ikisi de hazırsa hazırdır."""
    b_ok, b_msg = birdnet_status()
    p_ok, p_msg = perch_status()
    both_ok = b_ok and p_ok
    return [("BirdNET 2.4", b_ok, b_msg),
            ("Perch 2.0", p_ok, p_msg),
            ("BirdNET + Perch", both_ok,
             "İki model de hazır." if both_ok else "İki modelin de hazır olması gerekir.")]


# ------------------------------------------------------------------ veri klasörleri
def data_overview(csv_path: str, segments_dirs: List[str]) -> dict:
    """CSV yolu/kayıt sayısı ve segment klasörü/dosya sayısı. Dosya yoksa sayılar 0."""
    rows = 0
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
                rows = max(sum(1 for _ in f) - 1, 0)
        except OSError:
            rows = 0
    seg_dir = next((d for d in segments_dirs if os.path.isdir(d)), segments_dirs[0] if segments_dirs else "")
    n_seg = 0
    if seg_dir and os.path.isdir(seg_dir):
        n_seg = sum(1 for e in os.scandir(seg_dir) if e.is_file() and e.name.lower().endswith((".wav", ".flac", ".mp3")))
    return {"csv_path": csv_path, "csv_exists": os.path.isfile(csv_path), "csv_rows": rows,
            "segments_dir": seg_dir, "segments_exists": bool(seg_dir) and os.path.isdir(seg_dir),
            "segment_files": n_seg}
