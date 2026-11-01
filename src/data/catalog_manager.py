"""Tanım verilerinin yönetimi: tür veri tabanı, habitat noktaları ve seçim listeleri.

Üç JSON dosyası: bird_species_data.json (tür), habitat_data.json (habitat noktaları) ve
bird_sound_config.json (açılır listeler). Bu modül arayüzden bağımsızdır; ekle, güncelle, sil, listele,
doğrulama ve güvenli kaydetme işlerini yapar. Arayüz (src/ui/catalog_dialog.py) yalnızca bunu çağırır.

Güvenlik ilkeleri:
- Kayıt atomiktir (geçici dosyaya yaz, sonra yer değiştir); yarım yazılmış dosya oluşmaz.
- Her kayıttan önce eski dosyanın tarih damgalı yedeği `yedek_json/` klasörüne alınır (son 10 yedek tutulur).
- Yapılandırma dosyasında yalnızca düzenlenen anahtarlar değişir; bilinmeyen anahtarlar ve sıra korunur.
- Silme/yeniden adlandırma CSV'yi DEĞİŞTİRMEZ; yalnızca CSV'de kaç satırın etkileneceği raporlanır.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

SPECIES_FILE = "bird_species_data.json"
HABITAT_FILE = "habitat_data.json"
CONFIG_FILE = "bird_sound_config.json"
BACKUP_DIR = "yedek_json"
KEEP_BACKUPS = 10

SPECIES_FIELDS = ("order", "family", "scientific_name", "turkish_name")
HABITAT_FIELDS = ("habitat_ID", "habitat_tipi", "kodu", "utm_zone", "utm_easting", "utm_northing",
                  "lat", "lon", "google_maps", "konum")

# Yapılandırma listeleri: anahtar -> (görünen ad, CSV sütunu, arayüzde kullanılıyor mu)
CONFIG_LISTS: Dict[str, Tuple[str, Optional[str], bool]] = {
    "call_types": ("Ötüş türleri", "Call_Type", True),
    "background_noise": ("Arka plan gürültüsü", "Background_Noise_Type", True),
    "recordists": ("Kaydeden kişiler", "Recordist", True),
    "ornithologists": ("Ornitologlar", "Ornithologist", True),
    "verification_statuses": ("Doğrulama durumları", "Verification_Status", True),
    "microphones": ("Mikrofonlar", "Microphone", True),
    "recorders": ("Kaydediciler", "Recorder", True),
}
CONFIG_TEXT_FIELDS = {"project": "Proje adı"}

_BINOMIAL = re.compile(r"^[A-Z][a-z]+(-[a-z]+)?( [a-z]+(-[a-z]+)?){1,2}$")


def _norm(s) -> str:
    return " ".join(str(s or "").split()).casefold()


def _clean(s) -> str:
    return " ".join(str(s or "").split())


class CatalogError(ValueError):
    """Doğrulama hatası (kayıt eklenmez/güncellenmez)."""


class CatalogStore:
    def __init__(self, root: Optional[str] = None):
        if root is None:
            from .config import get_project_root
            root = get_project_root()
        self.root = root
        self.species: List[dict] = []
        self.habitats: List[dict] = []
        self.config: dict = {}
        self._dirty = {"species": False, "habitat": False, "config": False}
        self.load()

    # ---------------------------------------------------------------- yükleme / kaydetme
    def _path(self, name: str) -> str:
        return os.path.join(self.root, name)

    def _read(self, name: str, default):
        p = self._path(name)
        if not os.path.exists(p):
            return default
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def load(self) -> None:
        self.species = self._read(SPECIES_FILE, [])
        self.habitats = self._read(HABITAT_FILE, [])
        from .config import migrate_legacy_config
        self.config = migrate_legacy_config(self._read(CONFIG_FILE, {}))
        self._dirty = {k: False for k in self._dirty}

    @property
    def dirty(self) -> bool:
        return any(self._dirty.values())

    def dirty_kinds(self) -> List[str]:
        return [k for k, v in self._dirty.items() if v]

    def _backup(self, name: str) -> Optional[str]:
        src = self._path(name)
        if not os.path.exists(src):
            return None
        folder = self._path(BACKUP_DIR)
        os.makedirs(folder, exist_ok=True)
        stem, ext = os.path.splitext(name)
        dst = os.path.join(folder, f"{stem}_{datetime.now():%Y%m%d_%H%M%S_%f}{ext}")
        shutil.copy2(src, dst)
        olds = sorted(f for f in os.listdir(folder) if f.startswith(stem + "_") and f.endswith(ext))
        for f in olds[:-KEEP_BACKUPS]:
            try:
                os.remove(os.path.join(folder, f))
            except OSError:
                pass
        return dst

    def _write_atomic(self, name: str, data) -> None:
        folder = self.root
        fd, tmp = tempfile.mkstemp(prefix=name + ".", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp, self._path(name))
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def save(self) -> List[str]:
        """Değişen dosyaları (yedekleyerek) kaydeder. Döndürür: yazılan dosya adları."""
        written = []
        for kind, name, data in (("species", SPECIES_FILE, self.species),
                                 ("habitat", HABITAT_FILE, self.habitats),
                                 ("config", CONFIG_FILE, self.config)):
            if self._dirty[kind]:
                self._backup(name)
                self._write_atomic(name, data)
                self._dirty[kind] = False
                written.append(name)
        return written

    def revert(self) -> None:
        """Kaydedilmemiş tüm değişiklikleri diskteki hale döndürür."""
        self.load()

    # ---------------------------------------------------------------- türler
    def validate_species(self, rec: dict, index: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """(hatalar, uyarılar). Hata varsa kayıt yapılamaz."""
        errs, warns = [], []
        order, fam = _clean(rec.get("order")), _clean(rec.get("family"))
        sci, tr = _clean(rec.get("scientific_name")), _clean(rec.get("turkish_name"))
        if not order:
            errs.append("Takım (order) boş olamaz.")
        if not fam:
            errs.append("Familya (family) boş olamaz.")
        if not sci:
            errs.append("Bilimsel ad boş olamaz.")
        elif not _BINOMIAL.match(sci):
            errs.append("Bilimsel ad 'Cins tür' biçiminde olmalı (ör. Turdus merula; ilk harf büyük, diğerleri küçük).")
        if not tr:
            errs.append("Türkçe ad boş olamaz.")
        others = [(i, s) for i, s in enumerate(self.species) if i != index]
        if sci and any(_norm(s.get("scientific_name")) == _norm(sci) for _, s in others):
            errs.append(f"'{sci}' bilimsel adı zaten kayıtlı.")
        if tr and any(_norm(s.get("turkish_name")) == _norm(tr) for _, s in others):
            warns.append(f"'{tr}' Türkçe adı başka bir türde de var. Etiketlemede Türkçe ad bilimsel ada çevrilirken "
                         "ilk bulunan tür kullanılır; karışıklık olabilir.")
        return errs, warns

    @staticmethod
    def _species_rec(rec: dict) -> dict:
        return {k: _clean(rec.get(k)) for k in SPECIES_FIELDS}

    def add_species(self, rec: dict) -> int:
        errs, _ = self.validate_species(rec)
        if errs:
            raise CatalogError("\n".join(errs))
        self.species.append(self._species_rec(rec))
        self._dirty["species"] = True
        return len(self.species) - 1

    def update_species(self, index: int, rec: dict) -> None:
        errs, _ = self.validate_species(rec, index)
        if errs:
            raise CatalogError("\n".join(errs))
        self.species[index] = self._species_rec(rec)
        self._dirty["species"] = True

    def delete_species(self, index: int) -> dict:
        self._dirty["species"] = True
        return self.species.pop(index)

    def species_problems(self) -> List[str]:
        """Veri tabanındaki mevcut tutarsızlıklar (yinelenen bilimsel/Türkçe ad, bozuk biçim)."""
        out, seen_sci, seen_tr = [], {}, {}
        for i, s in enumerate(self.species):
            sci, tr = _clean(s.get("scientific_name")), _clean(s.get("turkish_name"))
            if _norm(sci) in seen_sci:
                out.append(f"Yinelenen bilimsel ad: {sci} (satır {seen_sci[_norm(sci)] + 1} ve {i + 1})")
            seen_sci.setdefault(_norm(sci), i)
            if _norm(tr) in seen_tr:
                other = self.species[seen_tr[_norm(tr)]].get("scientific_name")
                out.append(f"Aynı Türkçe ad iki türde: {tr} ({other} ve {sci})")
            seen_tr.setdefault(_norm(tr), i)
            if sci and not _BINOMIAL.match(sci):
                out.append(f"Bilimsel ad biçimi şüpheli: '{sci}'")
        return out

    # ---------------------------------------------------------------- habitat noktaları
    def next_habitat_id(self) -> int:
        ids = [int(h.get("habitat_ID", 0)) for h in self.habitats if str(h.get("habitat_ID", "")).lstrip("-").isdigit()]
        return (max(ids) if ids else 0) + 1

    @staticmethod
    def maps_link(lat: float, lon: float) -> str:
        return f"https://www.google.com/maps?q={lat},{lon}"

    def validate_habitat(self, rec: dict, index: Optional[int] = None) -> Tuple[List[str], List[str]]:
        errs, warns = [], []
        try:
            hid = int(str(rec.get("habitat_ID")).strip())
            if hid <= 0:
                errs.append("Habitat ID pozitif bir tam sayı olmalı.")
        except (TypeError, ValueError):
            hid = None
            errs.append("Habitat ID tam sayı olmalı.")
        if not _clean(rec.get("habitat_tipi")):
            errs.append("Habitat tipi boş olamaz.")
        kod = _clean(rec.get("kodu"))
        if not kod:
            errs.append("Kod boş olamaz.")
        if not _clean(rec.get("konum")):
            errs.append("Konum (il) boş olamaz.")
        for key, lo, hi, ad in (("lat", -90, 90, "Enlem (lat)"), ("lon", -180, 180, "Boylam (lon)")):
            try:
                v = float(str(rec.get(key)).replace(",", "."))
                if not lo <= v <= hi:
                    errs.append(f"{ad} {lo} ile {hi} arasında olmalı.")
            except (TypeError, ValueError):
                errs.append(f"{ad} sayı olmalı.")
        for key, ad in (("utm_easting", "UTM doğu"), ("utm_northing", "UTM kuzey")):
            v = str(rec.get(key, "")).strip()
            if v:
                try:
                    float(v.replace(",", "."))
                except ValueError:
                    errs.append(f"{ad} sayı olmalı (boş bırakılabilir).")
        others = [h for i, h in enumerate(self.habitats) if i != index]
        if hid is not None and any(str(h.get("habitat_ID")) == str(hid) for h in others):
            errs.append(f"Habitat ID {hid} zaten kayıtlı.")
        if kod and any(_norm(h.get("kodu")) == _norm(kod) for h in others):
            errs.append(f"'{kod}' kodu zaten kayıtlı.")
        return errs, warns

    def _habitat_rec(self, rec: dict) -> dict:
        def num(key):
            v = str(rec.get(key, "")).strip().replace(",", ".")
            return float(v) if v else None
        lat, lon = num("lat"), num("lon")
        out = {
            "habitat_ID": int(str(rec["habitat_ID"]).strip()),
            "habitat_tipi": _clean(rec.get("habitat_tipi")),
            "kodu": _clean(rec.get("kodu")),
            "utm_zone": _clean(rec.get("utm_zone")),
            "utm_easting": num("utm_easting"),
            "utm_northing": num("utm_northing"),
            "lat": lat,
            "lon": lon,
            "google_maps": self.maps_link(lat, lon),
            "konum": _clean(rec.get("konum")),
        }
        return out

    def add_habitat(self, rec: dict) -> int:
        errs, _ = self.validate_habitat(rec)
        if errs:
            raise CatalogError("\n".join(errs))
        self.habitats.append(self._habitat_rec(rec))
        self._dirty["habitat"] = True
        return len(self.habitats) - 1

    def update_habitat(self, index: int, rec: dict) -> None:
        errs, _ = self.validate_habitat(rec, index)
        if errs:
            raise CatalogError("\n".join(errs))
        self.habitats[index] = self._habitat_rec(rec)
        self._dirty["habitat"] = True

    def delete_habitat(self, index: int) -> dict:
        self._dirty["habitat"] = True
        return self.habitats.pop(index)

    def habitat_types(self) -> List[str]:
        return sorted({_clean(h.get("habitat_tipi")) for h in self.habitats if h.get("habitat_tipi")})

    def habitat_locations(self) -> List[str]:
        return sorted({_clean(h.get("konum")) for h in self.habitats if h.get("konum")})

    # ---------------------------------------------------------------- yapılandırma listeleri
    def list_keys(self) -> List[str]:
        """Dosyada bulunan (metin listesi olan) anahtarlar; arayüzde kullanılanlar önce."""
        keys = [k for k, v in self.config.items() if isinstance(v, list) and all(isinstance(x, str) for x in v)]
        used = [k for k in CONFIG_LISTS if k in keys and CONFIG_LISTS[k][2]]
        rest = [k for k in keys if k not in used]
        return used + rest

    def list_title(self, key: str) -> str:
        return CONFIG_LISTS.get(key, (key, None, False))[0]

    def list_is_used(self, key: str) -> bool:
        return CONFIG_LISTS.get(key, (key, None, False))[2]

    def get_list(self, key: str) -> List[str]:
        return list(self.config.get(key, []))

    def _check_item(self, key: str, value: str, skip: Optional[int] = None) -> str:
        v = _clean(value)
        if not v:
            raise CatalogError("Değer boş olamaz.")
        items = self.config.get(key, [])
        if any(_norm(x) == _norm(v) for i, x in enumerate(items) if i != skip):
            raise CatalogError(f"'{v}' bu listede zaten var.")
        return v

    def add_list_item(self, key: str, value: str) -> None:
        v = self._check_item(key, value)
        self.config.setdefault(key, []).append(v)
        self._dirty["config"] = True

    def update_list_item(self, key: str, index: int, value: str) -> None:
        v = self._check_item(key, value, skip=index)
        self.config[key][index] = v
        self._dirty["config"] = True

    def delete_list_item(self, key: str, index: int) -> str:
        self._dirty["config"] = True
        return self.config[key].pop(index)

    def move_list_item(self, key: str, index: int, delta: int) -> int:
        items = self.config[key]
        j = index + delta
        if not 0 <= j < len(items):
            return index
        items[index], items[j] = items[j], items[index]
        self._dirty["config"] = True
        return j

    def get_text(self, key: str) -> str:
        return str(self.config.get(key, ""))

    def set_text(self, key: str, value: str) -> None:
        v = _clean(value)
        if not v:
            raise CatalogError("Proje adı boş olamaz.")
        self.config[key] = v
        self._dirty["config"] = True

    # ---------------------------------------------------------------- CSV kullanım sayıları
    def species_usage(self, df, rec: dict) -> int:
        """CSV'de bu türe ait segment sayısı (Scientific_Name veya Turkish_Name eşleşmesi)."""
        if df is None or not len(df):
            return 0
        n = 0
        mask = None
        for col, key in (("Scientific_Name", "scientific_name"), ("Turkish_Name", "turkish_name")):
            if col in df.columns:
                m = df[col].map(_norm) == _norm(rec.get(key))
                mask = m if mask is None else (mask | m)
        return int(mask.sum()) if mask is not None else n

    def habitat_usage(self, df, rec: dict) -> int:
        if df is None or not len(df) or "Habitat_ID" not in df.columns:
            return 0
        return int((df["Habitat_ID"].astype(str).str.strip() == str(rec.get("habitat_ID")).strip()).sum())

    def list_item_usage(self, df, key: str, value: str) -> Optional[int]:
        """CSV'de bu değerin kullanıldığı satır sayısı; listenin CSV sütunu yoksa None."""
        col = CONFIG_LISTS.get(key, (None, None, False))[1]
        if df is None or not len(df) or not col or col not in df.columns:
            return None
        return int((df[col].map(_norm) == _norm(value)).sum())
