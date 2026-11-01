"""coverage_report ve yapılandırmalı compute_summary testleri."""
import json

import pandas as pd

from src.data import coverage_report as C
from src.data import dataset_summary as D


def _df():
    rows = []
    for i in range(12):
        loc = "Isparta" if i < 9 else "Burdur"
        typ = "Orman" if i < 8 else "Makilik"
        for sp in (f"s{i % 5}", f"s{(i + 1) % 7}"):
            rows.append({"Habitat_ID": str(i), "Location": loc, "Habitat_Type": typ, "Turkish_Name": sp,
                         "Recording_Date": "2026-08-24", "Recording_Time": "12:00:00"})
    return pd.DataFrame(rows)


def test_missing_location_and_small_groups_warn():
    cfg = {"use_catalog_design": False, "check_time_fields": True, "expected_locations": ["Burdur", "Isparta", "Antalya"], "expected_points_total": 312, "min_points_per_group": 10}
    r = C.build_coverage_report(_df(), cfg)
    txt = r.format()
    assert "Antalya: hiç etiketli nokta yok" in txt
    assert "12 / tasarımdaki 312" in txt
    assert "Burdur: yalnızca 3 etiketli nokta" in txt
    assert "tek değerde" in txt                   # tarih/saat yığılması uyarısı
    assert r.problems


def test_no_config_makes_no_assumptions_about_regions():
    r = C.build_coverage_report(_df(), {"use_catalog_design": False})
    assert "hiç etiketli nokta yok" not in r.format() and "Tasarım" not in r.format()


def test_planned_points_counts_unlabeled(tmp_path):
    plan = tmp_path / "plan.csv"
    plan.write_text("Habitat_ID;Location;Habitat_Type\n" + "\n".join(
        f"{i};{'Antalya' if i >= 20 else 'Isparta'};Orman" for i in range(0, 25)), encoding="utf-8")
    r = C.build_coverage_report(_df(), {"planned_points_csv": str(plan)})
    assert "Tasarım: 25 nokta; etiketli 12 (%48), etiketsiz 13" in r.format()
    assert "Antalya=0/5" in r.format()


def test_config_loading_is_safe(tmp_path):
    assert C.load_config(str(tmp_path / "yok.json"))["min_points_per_group"] == 10
    bad = tmp_path / "bad.json"; bad.write_text("{not json", encoding="utf-8")
    assert C.load_config(str(bad))["expected_locations"] == []
    ok = tmp_path / "ok.json"; ok.write_text(json.dumps({"expected_locations": ["A"], "zzz": 1}), encoding="utf-8")
    cfg = C.load_config(str(ok)); assert cfg["expected_locations"] == ["A"] and "zzz" not in cfg


def test_summary_with_config_groups_habitats():
    cfg = dict(C.DEFAULT_CONFIG, habitat_groups={"Orman": "Ormanlık", "Makilik": "Açık alan"},
               expected_locations=["Antalya"])
    sm = D.compute_summary(_df(), config=cfg)
    assert {e.group for e in sm.type_richness} == {"Ormanlık", "Açık alan"}
    txt = sm.format_richness_summary()
    assert "Kapsam raporu" in txt and "Antalya" in txt
    assert "Ormanlık: 8 nokta" in txt and "Chao2" not in txt


def test_summary_without_config_unchanged():
    sm = D.compute_summary(_df())
    assert sm.coverage_lines == [] and {e.group for e in sm.type_richness} == {"Orman", "Makilik"}


def _design(n_loc=("A", "B"), per=6):
    rows = []
    k = 0
    for loc in n_loc:
        for _ in range(per):
            k += 1
            rows.append({"Habitat_ID": str(k), "Location": loc, "Habitat_Type": "T" + str(k % 2)})
    return pd.DataFrame(rows)


def test_design_is_generic_any_region_names_and_sizes():
    des = _design(("Kuzey", "Güney", "Doğu"), per=4)           # 12 noktalı, 3 bölgeli herhangi bir çalışma
    df = pd.DataFrame({"Habitat_ID": [1.0, 2.0, 5.0], "Location": ["Kuzey", "Kuzey", "Güney"],
                       "Habitat_Type": ["T1", "T0", "T1"], "Turkish_Name": ["a", "b", "c"]})   # float ID'ler
    r = C.build_coverage_report(df, {"use_catalog_design": False}, design=des)
    t = r.format()
    assert "Tasarım: 12 nokta; etiketli 3" in t
    assert "Doğu: tasarımdaki 4 noktadan hiçbiri" in t
    assert "Kuzey=2/4" in t and "Güney=1/4" in t


def test_unknown_ids_and_worklist():
    des = _design(("A",), per=5)
    df = pd.DataFrame({"Habitat_ID": [1, 2, 99], "Location": ["A"] * 3, "Habitat_Type": ["T0"] * 3, "Turkish_Name": list("xyz")})
    r = C.build_coverage_report(df, {"use_catalog_design": False}, design=des)
    assert "tasarım listesinde yok: 99" in r.format()
    todo = C.unlabeled_points(df, {"use_catalog_design": False}, design=des)
    assert list(todo["Habitat_ID"]) == ["3", "4", "5"]


def test_time_check_does_not_flag_legit_variation():
    df = _df()
    df["Recording_Date"] = [f"2026-0{4 + i % 3}-{10 + i % 9}" for i in range(len(df))]
    df["Recording_Time"] = [f"{5 + i % 6:02d}:00:00" for i in range(len(df))]
    r = C.build_coverage_report(df, {"use_catalog_design": False, "check_time_fields": True, "expected_months": [4, 5, 9], "expected_hours": [5, 12]})
    t = r.format()
    assert "tek değerde" not in t
    assert "9. ayda hiç kayıt yok" in t and "kaydı olmayan saatler: 11, 12" in t


def test_time_fields_silent_by_default():
    t = C.build_coverage_report(_df(), {"use_catalog_design": False}).format()
    assert "tarih" not in t.lower() and "saat" not in t.lower() and "tek değerde" not in t


def _verified_fraction(statuses):
    df = pd.DataFrame({"Verification_Status": statuses, "Habitat_ID": [str(i) for i in range(len(statuses))],
                       "Location": "Isparta", "Habitat_Type": "Orman", "Turkish_Name": "s"})
    r = C.build_coverage_report(df, {"use_catalog_design": False, "min_verified_fraction": 0.99})
    import re
    m = re.search(r"Doğrulanmış etiket oranı %(\d+)", r.format())
    return int(m.group(1)) if m else 100


def test_verified_fraction_understands_turkish_and_english_statuses():
    assert _verified_fraction(["Uzman (Ornitolog) Doğrulamalı"] + ["Doğrulanmamış"] * 3) == 25
    assert _verified_fraction(["Expert Verified"] + ["Not Verified"] * 3) == 25
    assert _verified_fraction(["Uzman (Ornitolog) Doğrulamalı", "Etiketleyen"] + ["Doğrulanmamış"] * 2) == 50
