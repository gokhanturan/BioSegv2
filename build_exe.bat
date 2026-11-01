@echo off
REM ============================================================
REM BioSeg v2.0 - .exe Insa Betigi (PyInstaller)
REM ============================================================
REM Bu betigi, calisan bir .venv kurulumunun (setup_bioseg.bat
REM daha once calistirilmis) OLDUGU C:\Dev\BioSegv2.0 klasorunde
REM calistirin.
REM
REM --onefile modu kullaniliyor (v1.0 ile ayni "tek .exe" deneyimi
REM icin, kullanicinin acik tercihiyle). BILINMESI GEREKEN:
REM TensorFlow gibi cok buyuk bagimliliklar nedeniyle, .exe HER
REM baslatmada kendini gecici bir klasore (%TEMP%\_MEIxxxxxx) acar -
REM bu yuzden ilk acilis (ve her acilis, --onedir'e kiyasla) birkac
REM saniye daha yavas olabilir. Ayrica bazi virus koruma yazilimlari,
REM PyInstaller'in tek-dosya modunu YANLIS POZITIF olarak
REM isaretleyebilir (bilinen, yaygin bir durum) - boyle bir uyari
REM alirsaniz dosyayi "guvenilir" olarak isaretlemeniz gerekebilir.
REM
REM NOT - v1.0 ile ayni klasor duzeni: bird_species_data.json,
REM habitat_data.json, bird_sound_config.json, bird_sound_dataset.csv
REM dosyalari .exe'nin YANINDA (icine gomulu DEGIL) duruyor - tipki
REM v1.0'da oldugu gibi. Bu yuzden bu dosyalari --add-data ile ICERI
REM GOMMUYORUZ; derleme bittikten sonra AYRI bir adimda .exe'nin
REM yanina KOPYALIYORUZ (asagida). Kod tarafinda bu, config.py ve
REM csv_handler.py'deki sys.frozen kontrolu sayesinde otomatik olarak
REM dogru calisir (get_project_root(), frozen modda .exe'nin
REM bulundugu klasoru dondurur - --onefile modunda bile sys.executable
REM gecici acilma klasorunu DEGIL, gercek .exe'nin kendi konumunu
REM gosterir, bu yuzden bu mantik degismeden calisir).
REM ============================================================

if not exist ".venv\Scripts\activate.bat" (
    echo HATA: Once setup_bioseg.bat calistirilmis olmali.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo PyInstaller kuruluyor...
python -m pip install pyinstaller

echo.
echo === .exe insa ediliyor - bu 10-20 dakika surebilir ===
echo.

pyinstaller ^
    --name BioSeg ^
    --onefile ^
    --windowed ^
    --noconfirm ^
    --collect-all tensorflow ^
    --collect-all tensorflow_hub ^
    --collect-all PyQt5 ^
    --collect-all birdnetlib ^
    --collect-data librosa ^
    --collect-data matplotlib ^
    --collect-all sklearn ^
    --collect-all resampy ^
    --hidden-import=scipy.signal ^
    --hidden-import=scipy.io.wavfile ^
    src\main.py

echo.
if exist "dist\BioSeg.exe" (
    echo === Yapilandirma dosyalari .exe'nin yanina kopyalaniyor ===
    copy /Y "src\bird_species_data.json" "dist\bird_species_data.json" >NUL
    copy /Y "src\habitat_data.json" "dist\habitat_data.json" >NUL
    if exist "bird_sound_config.json" (
        copy /Y "bird_sound_config.json" "dist\bird_sound_config.json" >NUL
        echo   - bird_sound_config.json kopyalandi (mevcut ayarlariniz dahil^)
    ) else (
        echo   - bird_sound_config.json bulunamadi, program ilk acilista varsayilanla olusturacak
    )
    echo   - bird_species_data.json kopyalandi
    echo   - habitat_data.json kopyalandi
    echo.
    echo ============================================================
    echo BASARILI: dist\BioSeg.exe olusturuldu.
    echo ============================================================
    echo.
    echo Baska bir bilgisayara tasimak icin "dist" klasorundeki
    echo su dosyalari birlikte kopyalayin:
    echo   - BioSeg.exe
    echo   - bird_species_data.json
    echo   - habitat_data.json
    echo   - bird_sound_config.json (varsa^)
    echo.
    echo bird_sound_dataset.csv dosyasini EKLEMEK isterseniz (ekip mevcut
    echo verinizin uzerine calisacaksa^), onu da elle "dist\" klasorune
    echo kopyalayin - program ilk kayitta onu zaten yoksa otomatik olusturur.
    echo.
    echo NOT: Ilk acilista Windows Defender/virus korumasi bir uyari
    echo gosterebilir (PyInstaller'in tek-dosya .exe'lerinde SIK GORULEN,
    echo bilinen bir yanlis pozitif) - "Yine de calistir/Guvenilir olarak
    echo isaretle" secenegini kullanmaniz gerekebilir.
) else (
    echo ============================================================
    echo HATA: .exe olusturulamadi. Yukaridaki hata mesajlarini
    echo kontrol edip paylasin, birlikte cozelim.
    echo ============================================================
)
echo.
pause
