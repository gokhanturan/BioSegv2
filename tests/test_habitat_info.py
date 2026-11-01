"""Habitat Bilgileri: JSON + CSV birleştirme, tutarlılık uyarıları, pencere ve kart."""
import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.data import habitat_info as hi


def _json():
    out = []
    for i in range(1, 9):
        out.append({"habitat_ID": i, "habitat_tipi": "A" if i <= 4 else "B", "kodu": f"k{i}",
                    "konum": "Isparta" if i % 2 else "Antalya", "lat": 37.0 + i / 100, "lon": 30.0 + i / 100,
                    "utm_zone": "36N", "utm_easting": 1.0, "utm_northing": 2.0, "google_maps": f"https://x/{i}"})
    return out


def _csv(extra=None):
    rows = []
    for pid, n in ((1, 10), (2, 3), (5, 7)):
        for k in range(n):
            rows.append({"Habitat_ID": pid, "Habitat_Type": "A" if pid <= 4 else "B", "Scientific_Name": f"sp{k % 3}",
                         "Original_Recording_Name": f"r{pid}", "Lat": 37.0 + pid / 100, "Lon": 30.0 + pid / 100})
    rows += extra or []
    return pd.DataFrame(rows)


def test_build_counts_and_coverage():
    r = hi.build(_json(), _csv())
    o = r["ozet"]
    assert (o["n_tanimli"], o["n_etiketli"], o["n_segment"]) == (8, 3, 20)
    assert o["n_tip"] == 2 and o["n_bolge"] == 2 and o["n_dusuk_cabali"] == 1       # nokta 2: 3 segment
    tp = r["tipler"].set_index("habitat_tipi")
    assert tp.loc["A", "tanimli"] == 4 and tp.loc["A", "etiketli"] == 2 and tp.loc["A", "segment"] == 13
    assert tp.loc["B", "etiketli"] == 1 and tp.loc["B", "kapsam"] == 25.0
    assert r["matris_tanimli"].to_numpy().sum() == 8 and r["matris_etiketli"].to_numpy().sum() == 3
    assert r["uyarilar"] == []


def test_segment_total_equals_csv_rows():
    df = _csv()
    assert hi.build(_json(), df)["ozet"]["n_segment"] == len(df)


def test_warnings_unknown_point_type_mismatch_and_missing_id():
    extra = [{"Habitat_ID": 99, "Habitat_Type": "A", "Scientific_Name": "z", "Original_Recording_Name": "r", "Lat": 1, "Lon": 1},
             {"Habitat_ID": None, "Habitat_Type": "A", "Scientific_Name": "z", "Original_Recording_Name": "r", "Lat": 1, "Lon": 1},
             {"Habitat_ID": 3, "Habitat_Type": "B", "Scientific_Name": "z", "Original_Recording_Name": "r", "Lat": 40, "Lon": 40}]
    w = " ".join(hi.build(_json(), _csv(extra))["uyarilar"])
    assert "99" in w and "Habitat_ID değeri yok" in w and "uyuşmuyor" in w and "ayrışıyor" in w


def test_english_keys_and_empty_inputs():
    d = [{"habitat_ID": 1, "habitat_type": "Forest", "code": "c", "location": "Isparta", "lat": 1, "lon": 2}]
    assert hi.points_frame(d)["habitat_tipi"].iloc[0] == "Forest"
    r = hi.build([], None)
    assert r["noktalar"].empty and r["uyarilar"]
    r = hi.build(_json(), None)            # CSV yok: yalnızca envanter
    assert r["ozet"]["n_etiketli"] == 0 and r["ozet"]["n_tanimli"] == 8


def test_insights_flag_unlabeled_region_and_low_points():
    txt = " ".join(t + x for _k, t, x in hi.insights(hi.build(_json(), _csv())))
    assert "Antalya" in txt and "Düşük çaba" in txt and "3'ten az" in txt


def test_dialog_and_card(tmp_path):
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from src.ui.main_window import HabitatInfoDialog
    d = HabitatInfoDialog(hi.build(_json(), _csv()))
    names = [d._tabs.tabText(i) for i in range(d._tabs.count())]
    assert names == ["Genel Bakış", "Habitat Tipleri", "Bölgeler", "Noktalar"]
    assert d._pts_tbl.rowCount() == 8
    d._cmb_durum.setCurrentText("Etiketli")
    assert d._pts_tbl.rowCount() == 3
    d._cmb_durum.setCurrentText("Etiketsiz"); d._cmb_konum.setCurrentIndex(1)
    assert d._pts_tbl.rowCount() == len(d.result["noktalar"][(~d.result["noktalar"]["etiketli"]) &
                                        (d.result["noktalar"]["konum"] == d._cmb_konum.currentText())])
    HabitatInfoDialog(hi.build([], None))      # boş veri çökmemeli
    src = open("src/ui/main_window.py", encoding="utf-8").read()
    assert "btn_habitat_info" in src and "show_habitat_info" in src
