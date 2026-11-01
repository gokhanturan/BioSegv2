# BioSeg v2.0 — Yeni Bir Bilgisayara Kurulum Kılavuzu

Bu kılavuz, PyCharm gibi bir geliştirme ortamı kurulu OLMAYAN bir
bilgisayara BioSeg v2.0'ı kurup çalıştırmak için gereken adımları
açıklar. Ekibinize teslim ederken bu dosyayı da birlikte gönderin.

---

## 1. Neden PyInstaller ile tek bir .exe yapmadık

Bilerek yapmadım — TensorFlow (Perch 2.0 için) ve PyQtWebEngine (Location
Map için) gibi çok ağır, native bağımlılıkları PyInstaller ile tek bir
.exe'ye paketlemek **son derece kırılgan** olur (eksik DLL'ler, devasa
dosya boyutu, sessiz çökmeler). Bunu buradan (Linux sandbox) test
edemediğim için, ekibinize güvenilirliği doğrulanmamış bir paket
vermek istemedim. Bunun yerine, bugüne kadar sizin bilgisayarınızda
karşılaştığımız TÜM sorunları (DLL çakışması, setuptools uyumsuzluğu
vb.) önceden çözen, **otomatik bir kurulum betiği** hazırladım — tek
gereksinim, hedef bilgisayarda Python'un kurulu olması (PyCharm değil).

---

## 2. Hedef bilgisayarda önceden kurulması gerekenler

### a) Python 3.12 (python.org'dan — ÖNEMLİ)

- https://www.python.org/downloads/ adresinden **Python 3.12.x**'i indirin.
- Kurulum sırasında **"Add Python to PATH"** kutucuğunu mutlaka işaretleyin.
- **Anaconda/Miniconda kullanmayın** — bugün bizim geliştirme
  sürecimizde de bu net şekilde ayrım gerektirdi; Anaconda'nın kendi
  paket yönetimi TensorFlow ile uyumsuzluklara yol açabiliyor.

### b) ffmpeg (BirdNET için gerekli)

- https://ffmpeg.org/download.html adresinden Windows sürümünü indirin.
- Çıkardığınız klasördeki `bin` klasörünü sistem PATH'ine ekleyin
  (Denetim Masası → Sistem → Gelişmiş Sistem Ayarları → Ortam
  Değişkenleri → Path → Yeni).
- Doğrulamak için yeni bir PowerShell penceresinde: `ffmpeg -version`

### c) İnternet bağlantısı (sadece ilk çalıştırmada)

BirdNET 2.4 modeli, ilk "Bird Call Detection" ya da "Train
Classifier" kullanımında otomatik indirilir. Sonraki çalıştırmalarda
yerel önbellekten kullanılır, internet gerekmez.

---

## 3. Projeyi hedef bilgisayara taşıma

**Taşımayın:**
- `.venv\` klasörü (platform-özel, yeniden oluşturulacak)
- `__pycache__` klasörleri

**Taşıyıp taşımamaya siz karar verin:**
- `bird_sound_dataset.csv`, `segments\`, `src\trained_classifier\` —
  eğer ekip **sıfırdan kendi verisini** toplayacaksa bunları
  ÇIKARIN (boş başlasınlar). Eğer sizin mevcut verinizin üzerine
  ekleme yapacaklarsa, DAHİL EDİN.

**En basit yöntem:** Proje klasörünü (`.venv` ve `__pycache__` hariç)
sağ tıklayıp "Gönder → Sıkıştırılmış (zip) klasör" yapıp, ekibinize
gönderin (USB, e-posta, paylaşımlı sürücü — internet üzerinden
göndermek gerekmiyor).

---

## 4. Kurulum adımları (hedef bilgisayarda)

1. Zip dosyasını açın (örn. `C:\Dev\BioSegv2.0` olarak).
2. `C:\Dev\BioSegv2.0` klasörüne gidin.
3. **`setup_bioseg.bat`** dosyasına çift tıklayın — bu betik:
   - Sanal ortam (`.venv`) oluşturur,
   - `setuptools`'u doğru sürümle (önce) kurar,
   - `requirements.txt`'teki tüm paketleri kurar.
   - Birkaç dakika sürebilir (TensorFlow büyük bir paket).
4. Kurulum bitince **`run_bioseg.bat`**'a çift tıklayarak programı açın.

---

## 5. Sık karşılaşılabilecek sorunlar (bugünkü deneyimimizden)

| Belirti | Çözüm |
|---|---|
| `ModuleNotFoundError: pkg_resources` | `setup_bioseg.bat`'ı tekrar çalıştırın — `setuptools<82` adımı atlanmış olabilir |
| PyQt5 açılışta çöküyor / DLL hatası | `.venv`'i silip `setup_bioseg.bat`'ı yeniden çalıştırın |
| "Tensor data is null" / Perch yüklenmiyor | `%TEMP%\tfhub_modules` klasörünü silip tekrar deneyin (bozuk önbellek) |
| BirdNET ses işleyemiyor | ffmpeg'in PATH'e eklendiğini `ffmpeg -version` ile doğrulayın |

---

## 6. Ekibin veri setini birleştirme

Her ekip üyesi kendi bilgisayarında ayrı bir `bird_sound_dataset.csv`
biriktirecekse, bunları daha sonra birleştirmek için daha önce
kullandığımız birleştirme yöntemini (aynı `Filename` tekrarlarını
otomatik eleyen) uygulayabilirsiniz — o komutu hatırlarsanız tekrar
paylaşabilirim.
