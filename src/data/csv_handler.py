"""Veri seti dışa aktarımı için CSV dosyası işlemleri."""

import os
import sys
import csv


def get_project_root():
    """
    Proje kök dizinini (CSV dosyalarının saklandığı yer) döndürür.

    PyInstaller uyumludur — bkz. `config.get_project_root()` içindeki aynı not.
    Paketlenmiş (frozen) çalışmada CSV, .exe'nin yanında bulunur (kolay
    bulunur, yazılabilir); aksi hâlde normal kaynak ağacındaki proje köküdür.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # src/data/csv_handler.py -> src/data -> src -> proje kökü
    return os.path.dirname(os.path.dirname(current_dir))


def init_csv(filepath="bird_sound_dataset.csv"):
    """CSV dosyası yoksa başlık satırıyla birlikte oluşturur."""
    full_path = os.path.join(get_project_root(), filepath)

    if not os.path.exists(full_path):
        with open(full_path, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([
                "Original_Recording_Name", "Original_File_Path", "Original_Duration",
                "Channels", "File_Format", "Bit_Depth", "Recording_Date", "Recording_Time",
                "Filename", "File_Path", "Start_Time", "End_Time", "Duration",
                "Call_Type", "type_Count", "Order", "Family", "Scientific_Name", "Turkish_Name",
                "Background_Order", "Background_Family", "Background_Scientific_Name",
                "Background_Turkish_Name", "Background_Noise_Type",
                "Notes_For_Other_Species", "Location",
                "Recordist", "Ornithologist", "Verification_Status", "Confidence_Level",
                "Microphone", "Recorder",
                "Habitat_ID", "Habitat_Type", "Code", "UTM_Zone", "UTM_Easting", "UTM_Northing",
                "Lat", "Lon", "Google_Maps", "Notes", "Project"
            ])

    return full_path


def append_segment_to_csv(filepath, row_data):
    """Tek bir segment kaydını CSV dosyasının sonuna ekler."""
    full_path = os.path.join(get_project_root(), filepath)

    with open(full_path, mode='a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f, delimiter=';').writerow(row_data)


def get_csv_filepath():
    """Varsayılan CSV dosya adını döndürür."""
    return "bird_sound_dataset.csv"