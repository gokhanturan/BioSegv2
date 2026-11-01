@echo off
REM ============================================================
REM BioSeg v2.0 - Otomatik Kurulum Betigi
REM ============================================================
REM Bu betik, BioSegv2.0 klasorunun ICINDE calistirilmalidir.
REM Python 3.12'nin (python.org'dan, Anaconda/Microsoft Store DEGIL)
REM onceden kurulu oldugunu ve PATH'e eklendigini varsayar.
REM ============================================================

echo.
echo === BioSeg v2.0 Kurulumu Basliyor ===
echo.

REM 1. Adim: Python surumunu kontrol et
python --version 2>NUL
if errorlevel 1 (
    echo HATA: Python bulunamadi. Once https://www.python.org adresinden
    echo Python 3.12'yi kurun ve "Add Python to PATH" secenegini isaretleyin.
    pause
    exit /b 1
)

echo.
echo NOT: Asagida gorunen Python surumunun 3.12.x oldugunu kontrol edin.
echo      Anaconda/Miniconda Python'u KULLANMAYIN - uyumsuzluklara yol acar.
python --version
echo.
pause

REM 2. Adim: Sanal ortam (venv) olustur
if exist ".venv" (
    echo .venv klasoru zaten var, atlaniyor.
) else (
    echo Sanal ortam olusturuluyor...
    python -m venv .venv
)

REM 3. Adim: Sanal ortami etkinlestir
call .venv\Scripts\activate.bat

REM 4. Adim: pip'i guncelle
echo.
echo pip guncelleniyor...
python -m pip install --upgrade pip

REM 5. Adim: setuptools'u ONCE, dogru surumle kur
REM (tensorflow-hub, setuptools>=82'de kaldirilan pkg_resources'a
REM  bagimli - bu satir olmadan TensorFlow-Hub calismaz)
echo.
echo setuptools uyumlu surumle kuruluyor...
python -m pip install "setuptools<82"

REM 6. Adim: requirements.txt'teki tum bagimliliklari kur
echo.
echo Bagimliliklar kuruluyor (bu birkac dakika surebilir)...
python -m pip install -r requirements.txt

echo.
echo ============================================================
echo KURULUM TAMAMLANDI
echo ============================================================
echo.
echo Bir sonraki adim:
echo   1. ffmpeg'in kurulu oldugundan emin olun (BirdNET icin gerekli):
echo      https://ffmpeg.org/download.html adresinden indirip PATH'e ekleyin.
echo   2. "run_bioseg.bat" dosyasina cift tiklayarak programi baslatin.
echo   3. Ilk "Detect Bird Calls" ya da "Train Classifier" calistirmasinda,
echo      BirdNET 2.4 modeli ilk kullanimda internetten indirilecektir (~birkac dakika,
echo      sadece ilk seferde).
echo.
pause
