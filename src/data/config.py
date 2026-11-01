# ========================
# VERİ YÜKLEME VE YAPILANDIRMA
# ========================

import os
import sys
import json


def get_project_root():
    """
    Proje kök dizinini döndürür (JSON dosyalarının bulunduğu ve
    kullanıcının yazabileceği yapılandırma dosyasının durması gereken yer).

    PyInstaller uyumludur: paketlenmiş bir .exe olarak çalışırken
    (`--onedir` veya `--onefile`), `__file__` tabanlı çözümleme paketin
    İÇ klasörünü gösterirdi — kullanıcının kolayca bulamayacağı ve kurulum
    konumuna bağlı olarak (örn. Program Files) yazılamayabilecek bir yer.
    Bu durumda bunun yerine .exe'nin bulunduğu klasörü kullanırız — burası
    aynı zamanda `build_exe.bat`'ın derlemeden hemen sonra
    `bird_species_data.json`/`habitat_data.json` dosyalarını KOPYALADIĞI
    yerdir (v1.0'ın düzenine uygun olarak, .exe'nin yanında görünür/
    düzenlenebilir olsunlar diye bilerek paketin DIŞINDA tutulur) — böylece
    hem başvuru verileri hem de kullanıcının yazabildiği yapılandırma/veri
    seti dosyaları aynı, kolay bulunabilir klasöre çözümlenir.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # src/data/config.py -> src/data -> src (dosyalar src/ içindedir)
    return os.path.dirname(current_dir)


def load_json_data(filename, default_data=None):
    """Proje kökündeki bir JSON dosyasından veri yükler."""
    try:
        filepath = os.path.join(get_project_root(), filename)
        if not os.path.exists(filepath):
            print(f"Dosya bulunamadı: {filepath}, varsayılan kullanılıyor")
            return default_data if default_data is not None else []
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"{filename} yüklenirken hata: {e}")
        return default_data if default_data is not None else []


# Habitat ve tür verilerini JSON dosyalarından yükle
HABITAT_DATA = load_json_data("habitat_data.json", [{
    "habitat_ID": 1,
    "habitat_type": "Forest clearing",
    "code": "oia001",
    "utm_zone": "36N",
    "utm_easting": 307408.0,
    "utm_northing": 4177163.0,
    "lat": 37.721497806152726,
    "lon": 30.814775682616677,
    "google_maps": "https://www.google.com/maps?q=37.721498,30.814776",
    "location": "Isparta"
}])

BIRD_SPECIES_DATA = load_json_data("bird_species_data.json", [{
    "order": "Passeriformes",
    "family": "Turdidae",
    "scientific_name": "Turdus merula",
    "turkish_name": "Karatavuk"
}])


# Etiketleme formundaki açılır listeler (yalnızca Türkçe arayüz). Anahtar adları CSV sütunlarıyla eşleşir.
DEFAULT_CONFIG = {
    "call_types": ["Kur", "Çağrı", "Ötüş", "Alarm", "İletişim", "Diğer"],
    "background_noise": ["Motor Sesi", "İnsan Sesi", "Rüzgar Sesi", "Telefon Sesi", "Yağmur Sesi", "Karışık Ses", "Diğer"],
    "recordists": ["Diğer"],
    "ornithologists": ["Diğer", "Doğrulanmadı"],
    "verification_statuses": ["Uzman (Ornitolog) Doğruladı", "Etiketçi", "Doğrulanmadı"],
    "microphones": ["Diğer"],
    "recorders": ["Diğer"],
    "project": "BioSeg Project 2026",
}
CONFIG_KEYS = tuple(DEFAULT_CONFIG)


def migrate_legacy_config(cfg):
    """Eski biçimi (…ENG anahtarları + İngilizce yedek listeler) yeni sade biçime çevirir; yeni biçimi olduğu gibi bırakır.

    Eski dosyalarda arayüzün gerçekten kullandığı liste `xxxENG` anahtarındaydı; ENG'siz anahtar ise İngilizce
    eski listeydi. Bu yüzden ENG anahtarı varsa onun değeri kullanılır. Kullanılmayan anahtarlar (locations,
    habitats, background_species, bird_species ...) bellekte atılır; dosya ancak kayıtta değişir."""
    out = {}
    for k in CONFIG_KEYS:
        v = cfg.get(k + "ENG", cfg.get(k))
        if v is not None:
            out[k] = v
    return out


def load_config(filepath="bird_sound_config.json"):
    """Yapılandırmayı JSON dosyasından yükler; dosya yoksa varsayılanı oluşturur."""
    default_config = dict(DEFAULT_CONFIG)

    try:
        # ÖNEMLİ DÜZELTME: Dosya, çalışma dizinine (CWD) göre değil, DAİMA
        # proje köküne (src/ klasörü) göre aranmalı — aksi halde uygulama
        # farklı bir dizinden (örn. proje kökünden `python src/main.py`
        # ile) çalıştırıldığında gerçek yapılandırma dosyası bulunamaz ve
        # sessizce üzerine genel varsayılan değerlerle YENİ bir dosya
        # yazılır (kullanıcının "Ornithologist"/"Verification Status" gibi
        # kendi girdileri görünmez hale gelir). save_config() zaten
        # get_project_root() kullanıyordu; load_config() de aynısını
        # kullanmalı.
        filepath_full = os.path.join(get_project_root(), filepath)
        if os.path.exists(filepath_full):
            with open(filepath_full, 'r', encoding='utf-8') as f:
                config_data = migrate_legacy_config(json.load(f))
                # Varsayılanlarla birleştir (eksik anahtarları ekle)
                for key, value in default_config.items():
                    if key not in config_data:
                        config_data[key] = value
                return config_data
        else:
            # Yapılandırma dosyası yoksa varsayılanını oluştur
            with open(filepath_full, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            return default_config
    except Exception as e:
        print(f"Yapılandırma yüklenirken hata: {e}")
        return default_config


def save_config(cfg, filepath="bird_sound_config.json"):
    """Yapılandırmayı JSON dosyasına kaydeder."""
    try:
        filepath_full = os.path.join(get_project_root(), filepath)
        with open(filepath_full, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Yapılandırma kaydedilirken hata: {e}")
        return False


def get_turkish_names(bird_data):
    """Benzersiz Türkçe kuş adlarının sıralı listesini çıkarıp döndürür."""
    names = sorted(set(b.get("turkish_name", "") for b in bird_data))
    return [n for n in names if n]


def get_locations(habitat_data):
    """Habitat verisindeki benzersiz konumların sıralı listesini çıkarıp döndürür."""
    locs = sorted(set(h.get("location", "") for h in habitat_data if h.get("location", "")))
    return locs
