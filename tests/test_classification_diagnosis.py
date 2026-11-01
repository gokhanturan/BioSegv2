"""Başarı sınırı analizi: veri setinden bağımsız (tür adı/sayı gömülü değil) davranış."""
import matplotlib
matplotlib.use("Agg")
import types
import numpy as np
import pandas as pd

from src.ml import classification_diagnosis as C
from src.visualization import classification_plots as P


def _df(counts, n_rec=3):
    rows = []
    for sp, n in counts.items():
        for i in range(n):
            rows.append({"Scientific_Name": sp, "Original_Recording_Name": f"{sp}_{i % n_rec}"})
    return pd.DataFrame(rows)


def _summary(accs, f1=0.7):
    return [{"yontem": k, "sinif_agirligi": "True", "n_kat": 15, "dogruluk_ort": a, "dogruluk_ss": 0.05,
             "makro_f1_ort": f1, "makro_f1_ss": 0.05} for k, a in accs.items()]


def _per_class(f1):
    return [{"sinif_agirligi": "True", "tur": s, "f1": v, "precision": v, "recall": v, "destek_segment": 1}
            for s, v in f1.items()]


def test_eligibility_and_no_cv():
    df = _df({"A": 40, "B": 32, "C": 4})
    d = C.diagnose(df)
    assert (d.n_species, d.n_eligible, d.n_segments_excluded) == (3, 2, 4)
    assert d.verdict == "cv_yok" and "capraz_dogrulama.py" in d.format()


def test_single_recording_species_excluded():
    df = pd.DataFrame({"Scientific_Name": ["X"] * 20, "Original_Recording_Name": ["r"] * 20})
    assert C.diagnose(df).n_eligible == 0


def test_data_limited_verdict():
    counts = {f"S{i}": 10 + 5 * i for i in range(10)}
    f1 = {f"S{i}": 0.3 + 0.06 * i for i in range(10)}          # destek arttıkça F1 artar
    d = C.diagnose(_df(counts), _summary({"MLP_BirdNET": .85, "LR_BirdNET": .85, "SifirAtis_Perch": .83}), _per_class(f1))
    assert d.plateau and d.rho > 0.9 and d.verdict == "veri"
    assert "veridir" in d.format() and d.weakest[0]["tur"] == "S0"


def test_model_limited_verdict_when_methods_differ():
    counts = {f"S{i}": 15 for i in range(8)}
    d = C.diagnose(_df(counts), _summary({"MLP_BirdNET": .9, "LR_BirdNET": .75, "SifirAtis_Perch": .6}),
                   _per_class({f"S{i}": .8 for i in range(8)}))
    assert d.plateau is False and d.verdict == "model"


def test_stale_cv_warning():
    d = C.diagnose(_df({"A": 20, "B": 20}), _summary({"MLP_BirdNET": .8, "LR_BirdNET": .8}),
                   _per_class({"A": .9, "Zzz": .5}))
    assert d.stale_note


def test_plots_render():
    counts = {f"S{i}": 8 + 3 * i for i in range(60)}
    f1 = {f"S{i}": min(1, 0.2 + 0.015 * i) for i in range(60)}
    d = C.diagnose(_df(counts), _summary({"MLP_BirdNET": .85, "LR_BirdNET": .85, "SifirAtis_Perch": .83, "LR_MFCC": .3}),
                   _per_class(f1))
    for fig in (P.data_sufficiency(d.species_table), P.methods_accuracy_f1(d.methods),
                P.support_vs_f1(d.species_table, d.rho)):
        assert fig.axes


def test_single_split_roundtrip_and_sources(tmp_path):
    tr = types.SimpleNamespace(
        test_accuracy=.838, test_macro_f1=.705, n_train=477, n_val=149, n_test=228, n_test_recordings=20,
        class_names=["A", "B"], per_class_report={"A": {"f1-score": 1.0, "support": 3}, "B": {"f1-score": .5, "support": 50}})
    C.save_single_split(str(tmp_path), tr)
    sg = C.load_single_split(str(tmp_path))
    assert sg["dogruluk"] == .838 and sg["tur"]["A"]["destek"] == 3
    d = C.diagnose(_df({"A": 20, "B": 20}), _summary({"MLP_BirdNET": .8, "LR_BirdNET": .8}), _per_class({"A": .9, "B": .8}),
                   single_split=sg, times={C.SUMMARY_FILE: 1000.0}, data_time=5000.0)
    txt = d.format()
    assert "Son eğitim (tek bölme)" in txt and "Kaynak: cv_yontem_ozeti.csv" in txt
    assert d.age_note and "yeni" in d.age_note


def test_no_single_split_file(tmp_path):
    assert C.load_single_split(str(tmp_path)) is None


def test_single_split_shown_without_cv():
    sg = {"zaman": "2026-10-11T20:00:00", "dogruluk": .8, "makro_f1": .7, "n_train": 10, "n_val": 2, "n_test": 5,
          "n_test_kayit": 2, "n_tur": 2, "tur": {"A": {"f1": 1.0, "destek": 2}}}
    assert "Son eğitim (tek bölme)" in C.diagnose(_df({"A": 20, "B": 20}), single_split=sg).format()


def test_paired_rows_deduplicated_and_makro_f1_only():
    mk = lambda m, v: {"karsilastirma": "MLP - LR_BirdNET", "metrik": m, "fark_ort": v, "fark_ss": .1,
                       "p_duzeltilmis": .05, "n_kat": 15, "mlp_kazanilan_kat": 9, "p_naif": .01}
    rows = [mk("dogruluk", 0.0), mk("makro_f1", .08), mk("dogruluk", 0.0), mk("makro_f1", .09)]
    d = C.diagnose(_df({"A": 20, "B": 20}), _summary({"MLP_BirdNET": .8, "LR_BirdNET": .8}),
                   _per_class({"A": .9, "B": .8}), rows)
    assert len(d.paired) == 1 and float(d.paired[0]["fark_ort"]) == .08


# ---- veri izi (sonuçların hangi veriyle üretildiği) ----
def _csv(path, rows):
    pd.DataFrame(rows).to_csv(path, sep=";", index=False, encoding="utf-8-sig")


_ROWS = [{"Original_Recording_Name": f"r{i % 3}", "Start_Time": str(3 * i), "End_Time": str(3 * i + 3),
          "Scientific_Name": "A" if i % 2 else "B", "Note": "x"} for i in range(12)]


def test_fingerprint_stable_to_copy_and_order_changes_on_label(tmp_path):
    import os, shutil, time
    a, b = str(tmp_path / "a.csv"), str(tmp_path / "b.csv")
    _csv(a, _ROWS)
    fa = C.data_fingerprint(a)
    shutil.copy(a, b); os.utime(b, (time.time() + 999, time.time() + 999))     # dosya zamanı farklı
    assert C.data_fingerprint(b)["hash"] == fa["hash"]
    _csv(b, list(reversed(_ROWS)))                                              # satır sırası farklı
    assert C.data_fingerprint(b)["hash"] == fa["hash"]
    rows = [dict(r) for r in _ROWS]; rows[0]["Scientific_Name"] = "C"           # bir etiket değişti
    _csv(b, rows)
    fb = C.data_fingerprint(b)
    assert fb["hash"] != fa["hash"] and fb["n_tur"] == 3 and fa["n_segment"] == 12
    rows = [dict(r, Note="y") for r in _ROWS]                                   # ilgisiz sütun değişti
    _csv(b, rows)
    assert C.data_fingerprint(b)["hash"] == fa["hash"]
    assert C.data_fingerprint(str(tmp_path / "yok.csv")) is None


def test_fingerprint_states_in_diagnosis(tmp_path):
    df = _df({"A": 30, "B": 30})
    summ, pc = _summary({"MLP_BirdNET": .85, "LR_BirdNET": .85}), _per_class({"A": .8, "B": .8})
    fp = {"n_segment": 60, "n_tur": 2, "n_kayit": 6, "hash": "abc"}
    ok = C.diagnose(df, summ, pc, fp_saved=fp, fp_now=dict(fp))
    assert ok.fp_state == "guncel" and "güncel veriyle" in ok.format() and not ok.age_note
    old = C.diagnose(df, summ, pc, fp_saved=fp, fp_now=dict(fp, hash="zzz", n_segment=70))
    assert old.fp_state == "eski" and "ESKİ veriyle" in old.format() and "70 segment" in old.format()
    none = C.diagnose(df, summ, pc, fp_saved=None, fp_now=fp, times={C.SUMMARY_FILE: 100.0}, data_time=500.0)
    assert none.fp_state == "yok" and "veri izi yok" in none.format() and none.age_note   # zamana geri döner
    # zaman daha yeni olsa bile iz eşleşiyorsa zaman uyarısı verilmez (kopyalama/geri yükleme yanlış alarmı)
    fine = C.diagnose(df, summ, pc, times={C.SUMMARY_FILE: 100.0}, data_time=500.0, fp_saved=fp, fp_now=dict(fp))
    assert fine.fp_state == "guncel" and not fine.age_note


def test_fingerprint_file_roundtrip(tmp_path):
    f = {"n_segment": 3, "n_tur": 1, "n_kayit": 1, "hash": "h"}
    C.write_fingerprint(str(tmp_path), f)
    assert C.load_fingerprint(str(tmp_path))["hash"] == "h"
    C.clear_fingerprint(str(tmp_path))
    assert C.load_fingerprint(str(tmp_path)) is None


def test_display_label_turkish_with_latin():
    df = pd.DataFrame([{"Scientific_Name": "Parus major", "Turkish_Name": "Büyük Baştankara",
                        "Original_Recording_Name": f"r{i % 3}"} for i in range(12)])
    sp = C.species_support(df)
    assert sp.loc[0, "etiket"] == "Büyük Baştankara (Parus major)" and sp.loc[0, "tur"] == "Parus major"
    d = C.diagnose(df, _summary({"MLP_BirdNET": .8, "LR_BirdNET": .8}), _per_class({"Parus major": .4, "Y": .9, "Z": .9,
                                                                               "W": .9, "V": .9, "U": .9}))
    assert any("Büyük Baştankara (Parus major)" in l for l in d.format().split("<br>"))
    # Türkçe ad sütunu yoksa Latince'ye geri döner
    assert C.species_support(_df({"A": 12})).loc[0, "etiket"] == "A"


def test_no_weak_species_gives_coverage_verdict_not_data_claim():
    """Süzgeçten geçen hiçbir tür zayıf değilse 'zayıf türler az segmentli' denmez; sınır kapsam olarak söylenir."""
    counts = {f"S{i}": 35 + 8 * i for i in range(10)}
    counts.update({f"R{i}": 3 for i in range(20)})                      # süzgeç dışı, nadir türler
    f1 = {f"S{i}": 0.85 + 0.01 * i for i in range(10)}
    d = C.diagnose(_df(counts), _summary({"MLP_BirdNET": .9, "LR_BirdNET": .9}), _per_class(f1))
    assert d.n_weak == 0 and d.verdict == "kapsam"
    t = d.format()
    assert "zayıf türler az segmentli" not in t and "veri kapsamıdır" in t
    assert "20 tür" in t and "tamamında F1 ≥ 0.7" in t
    assert d.rho_p is not None and f"p = {d.rho_p:.3f}" in t


def test_data_verdict_needs_significant_rho():
    counts = {f"S{i}": 40 + i for i in range(6)}
    f1 = {"S0": .6, "S1": .9, "S2": .5, "S3": .9, "S4": .95, "S5": .6}   # zayıf var, ilişki zayıf/anlamsız
    d = C.diagnose(_df(counts), _summary({"MLP_BirdNET": .85, "LR_BirdNET": .85}), _per_class(f1))
    assert d.n_weak > 0 and d.verdict == "belirsiz"


def test_scatter_title_has_n_and_p_and_plain_ticks():
    counts = {f"S{i}": 35 + 8 * i for i in range(11)}
    f1 = {f"S{i}": 0.85 + 0.01 * i for i in range(11)}
    d = C.diagnose(_df(counts), _summary({"MLP_BirdNET": .9, "LR_BirdNET": .9}), _per_class(f1))
    fig = P.support_vs_f1(d.species_table, d.rho, p=d.rho_p)
    ax = fig.axes[0]
    assert "n = 11" in ax.get_title() and "p = " in ax.get_title()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert labels and all("^" not in x and "×" not in x and "10" != x[:2] + "^" for x in labels)


# ---- üç modelin kayıtlı sonuçlarından (BirdNET 2.4 / Perch 2.0 / birleşik) ----
def _stored(model_id, ad, acc, f1, tur_f1, h="abc", cv=True, offset=0.0, phase=0):
    import datetime as dt
    rows = [{"tekrar": r, "kat": k, "dogruluk": acc + offset * ((r * 5 + k + phase) % 3 - 1), "makro_f1": f1 + offset * ((r * 5 + k + phase) % 3 - 1),
             "n_train": 80, "n_val": 10, "n_test": 30} for r in range(2) for k in range(5)]
    summ = ({"n_kat": 5, "n_tekrar": 2, "sinif_agirligi": True, "dogruluk": (acc, .03), "makro_f1": (f1, .04),
             "tur_f1": tur_f1, "kat_satirlari": rows, "bolme_izi": "iz1"} if cv else None)
    tr = types.SimpleNamespace(cv_summary=summ)
    spec = types.SimpleNamespace(id=model_id, ad=ad)
    st = types.SimpleNamespace(result=types.SimpleNamespace(training_result=tr), saved_at=dt.datetime(2026, 10, 21, 13, 50),
                               fingerprint={"hash": h, "n_segment": 663, "n_tur": 11, "n_kayit": 40})
    return spec, st


def _three(h=("abc", "abc", "abc")):
    sp = [f"S{i}" for i in range(11)]
    mk = lambda base: {s: base + 0.01 * i for i, s in enumerate(sp)}
    return [_stored("birdnet", "BirdNET 2.4", .95, .93, mk(.85), h[0], offset=.02, phase=0),
            _stored("perch", "Perch 2.0", .94, .92, mk(.84), h[1], offset=.02, phase=1),
            _stored("birlesik_birdnet_perch", "BirdNET + Perch (birleşik)", .955, .935, mk(.86), h[2], offset=.02, phase=2)]


def _df11():
    counts = {f"S{i}": 35 + 8 * i for i in range(11)}
    counts.update({f"R{i}": 3 for i in range(20)})
    return _df(counts)


def test_from_stored_three_models_no_mfcc_and_paired_tests():
    kw = C.from_stored(_three())
    d = C.diagnose(_df11(), fp_now={"hash": "abc", "n_segment": 663, "n_tur": 11, "n_kayit": 40}, **kw)
    assert d.model_mode and d.has_cv and set(d.methods) == set(C.MODEL_IDS)
    t = d.format()
    assert "Model karşılaştırması" in t and "BirdNET 2.4" in t and "Perch 2.0" in t and "MFCC" not in t
    assert "5×2 çapraz doğrulama" in t and "Nadeau–Bengio" in t
    assert d.fp_state == "guncel" and "Tüm modeller güncel" in t
    assert d.plateau is True and d.verdict == "kapsam"
    assert "gömme/model seçimi" in t and "MLP" not in d.verdict_text()
    assert {"f1_min", "f1_max"} <= set(d.species_table.columns)
    r = d.species_table.dropna(subset=["f1"]).iloc[0]
    assert r["f1_min"] <= r["f1"] <= r["f1_max"]


def test_stale_model_detected_per_model():
    d = C.diagnose(_df11(), fp_now={"hash": "abc", "n_segment": 663, "n_tur": 11, "n_kayit": 40},
                   **C.from_stored(_three(h=("abc", "OLD", "abc"))))
    assert d.fp_state == "eski" and "Perch 2.0" in d.fp_note and "BirdNET 2.4" not in d.fp_note.split("Şimdiki")[0]


def test_model_without_cv_is_noted_not_charted():
    st = _three()
    st[1] = _stored("perch", "Perch 2.0", .9, .9, {}, cv=False)
    kw = C.from_stored(st)
    d = C.diagnose(_df11(), **kw)
    assert "perch" not in d.methods and any("Perch 2.0" in n and "çapraz doğrulama sonucu yok" in n for n in d.model_notes)
    assert "Perch 2.0: çapraz doğrulama sonucu yok" in d.format()


def test_no_stored_models_message():
    d = C.diagnose(_df11(), **C.from_stored([]))
    assert d.model_mode and d.verdict == "cv_yok"
    assert "eğitim penceresinde" in d.format() and "capraz_dogrulama.py" not in d.format()


def test_model_plots_render_with_range():
    d = C.diagnose(_df11(), **C.from_stored(_three()))
    fig = P.methods_accuracy_f1(d.methods)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["BirdNET + Perch (birleşik)", "Perch 2.0", "BirdNET 2.4"]
    fig2 = P.support_vs_f1(d.species_table, d.rho, p=d.rho_p)
    assert "n = 11" in fig2.axes[0].get_title() and fig2.axes[0].get_legend() is not None


def test_thesis_figures_use_models_not_legacy_csv(tmp_path, monkeypatch):
    import importlib, sys, os
    sys.path.insert(0, os.getcwd())
    ts = importlib.import_module("tez_sekilleri")
    from src.ml import model_registry
    monkeypatch.setattr(model_registry, "stored_results", lambda root: _three())
    df = _df11().assign(Turkish_Name="x")
    csvp = tmp_path / "d.csv"
    df.to_csv(csvp, sep=";", index=False, encoding="utf-8-sig")
    out = ts.uret(str(csvp), None, None, str(tmp_path / "o"), 60, kok=str(tmp_path))
    assert {"sekil_05_veri_yeterliligi", "sekil_06_destek_f1", "sekil_07_cv_yontemler_makro_f1",
            "sekil_08_cv_yontemler_dogruluk", "sekil_09_cv_tur_f1"} <= set(out)


def test_model_mode_plateau_follows_paired_tests_not_accuracy_spread():
    """Doğruluk aralığı eşiğin biraz üstünde olsa bile eşleştirilmiş testte anlamlı fark yoksa 'model' denmez."""
    st = _three()
    st[1][1].result.training_result.cv_summary["dogruluk"] = (.915, .05)      # aralık > 0.03
    d = C.diagnose(_df11(), **C.from_stored(st))
    assert d.embedding_acc_spread > C.PLATEAU_ACC_SPREAD
    assert d.min_p_models is not None and d.min_p_models >= 0.05
    assert d.plateau is True and d.verdict == "kapsam"
    assert "anlamlı fark bulunamadı" in d.format()


def test_significant_model_difference_gives_model_verdict():
    st = _three()
    for r in st[1][1].result.training_result.cv_summary["kat_satirlari"]:    # Perch her katta belirgin kötü
        r["makro_f1"] -= 0.2
    d = C.diagnose(_df11(), **C.from_stored(st))
    assert d.plateau is False and d.verdict == "model" and "anlamlı fark var" in d.format()


def test_sufficiency_chart_shows_all_species():
    d = C.diagnose(_df11(), **C.from_stored(_three()))
    fig = P.data_sufficiency(d.species_table)
    assert len(fig.axes[0].patches) >= len(d.species_table) and "Gösterilmeyen" not in fig.axes[0].get_title()
