"""recording_level: kayıt bazlı hata tablosu ve (kayıt, tür) varlık metrikleri."""
import os

import numpy as np

from src.ml import recording_level as rl
from src.ml.dataset_builder import TrainingSample


def R(rec, gercek, tahmin, guven=0.9):
    return {"recording_id": rec, "dosya": "x.wav", "gercek": gercek, "tahmin": tahmin, "guven": guven}


def test_error_table_sorted_and_top_error():
    rows = ([R("r1", "A", "A")] * 5 + [R("r2", "B", "C")] * 4 + [R("r2", "B", "B")] * 1)
    t = rl.recording_error_table(rows)
    assert [e["kayit"] for e in t] == ["r2", "r1"]
    assert t[0]["n_hata"] == 4 and t[0]["en_sik_hata"] == "B -> C (4)"
    assert t[1]["dogruluk"] == 1.0 and t[1]["en_sik_hata"] == ""


def test_presence_perfect_and_multispecies_recording():
    rows = [R("r1", "A", "A"), R("r1", "A", "A"), R("r1", "B", "B")]   # r1 iki tür içerir
    m = rl.presence_metrics(rows)
    assert m["precision"] == m["recall"] == m["f1"] == 1.0 and m["tam_eslesen_kayit"] == 1.0
    assert m["n_cift_gercek"] == 2


def test_min_segment_removes_isolated_false_positive():
    rows = [R("r1", "A", "A")] * 6 + [R("r1", "A", "B")]    # tek segment yanlış B
    m1, m2 = rl.presence_metrics(rows, min_segment=1), rl.presence_metrics(rows, min_segment=2)
    assert m1["precision"] == 0.5 and m1["recall"] == 1.0
    assert m2["precision"] == 1.0 and m2["f1"] == 1.0


def test_min_conf_filters_low_confidence_predictions():
    rows = [R("r1", "A", "A", 0.9), R("r1", "A", "B", 0.3)]
    assert rl.presence_metrics(rows, min_conf=0.5)["precision"] == 1.0
    assert rl.presence_metrics(rows, min_conf=0.0)["precision"] == 0.5


def test_missed_species_counts_as_fn():
    rows = [R("r1", "A", "A"), R("r1", "B", "A")]            # B hiç tahmin edilmedi
    m = rl.presence_metrics(rows)
    assert m["recall"] == 0.5 and m["precision"] == 1.0
    per = {d["tur"]: d for d in m["tur_bazli"]}
    assert per["B"]["fn"] == 1 and per["A"]["tp"] == 1


def test_build_rows_and_reports(tmp_path):
    samples = [TrainingSample(filepath=f"/x/s{i}.wav", species="A", recording_id=f"r{i % 2}") for i in range(4)]
    proba = np.array([[.9, .1], [.2, .8], [.7, .3], [.4, .6]])
    rows = rl.build_prediction_rows(samples, [0, 1, 2, 3], np.array([0, 0, 0, 1]), proba, ["A", "B"])
    assert rows[1]["gercek"] == "A" and rows[1]["tahmin"] == "B" and abs(rows[1]["guven"] - 0.8) < 1e-9
    written = rl.write_reports(str(tmp_path), rows)
    assert set(written) == {"test_segment_tahminleri.csv", "test_kayit_bazli.csv", "test_tur_varligi.csv"}
    assert all(os.path.getsize(tmp_path / w) > 0 for w in written)
    assert "kesinlik" in rl.format_presence_summary(rows)


def test_presence_from_arrays_matches_rows():
    yt, yp, rc = [0, 0, 1, 1], [0, 1, 1, 1], ["r1", "r1", "r2", "r2"]
    m = rl.presence_from_arrays(yt, yp, rc, 2, min_segment=1)
    assert m["n_kayit"] == 2 and m["recall"] == 1.0 and m["precision"] == 2 / 3


def test_training_result_summary_includes_presence_line():
    from src.ml.train_classifier import TrainingResult
    tr = TrainingResult(test_accuracy=.9, test_macro_f1=.8, per_class_report={}, confusion_matrix=np.zeros((1, 1)),
                        class_names=["A"], n_train=1, n_val=1, n_test=1, n_train_recordings=1, n_val_recordings=1,
                        n_test_recordings=1, test_predictions=[R("r1", "A", "A")])
    assert "tür varlığı" in tr.format_summary()


def test_dominance_warning_triggers_on_single_dominant_recording():
    rows = ([R("big", "A", "A")] * 6 + [R("big", "B", "C")] * 9          # 15 segment, 9 hata
            + [R(f"r{i}", "A", "A") for i in range(10)] + [R("r_x", "A", "B")] * 1)
    w = rl.dominance_warning(rows)
    assert "big" in w and "çapraz doğrulama" in w
    assert rl.dominance_warning([R(f"r{i}", "A", "A") for i in range(20)]) == ""


def test_summarize_cv_and_headline_in_summary():
    from types import SimpleNamespace
    from src.ml.pipeline import summarize_cv
    from src.ml.train_classifier import TrainingResult
    fold_rows = [{"tekrar": t, "kat": k, "sinif_agirligi": True, "dogruluk": 0.9 + 0.01 * k, "makro_f1": 0.8 + 0.02 * k}
                 for t in (1, 2) for k in (1, 2, 3)]
    fold_rows.append({"tekrar": 1, "kat": 9, "sinif_agirligi": False, "dogruluk": 0.1, "makro_f1": 0.1})
    pc = [{"sinif_agirligi": True, "tur": "A", "f1": 0.9}, {"sinif_agirligi": False, "tur": "A", "f1": 0.1}]
    cv = summarize_cv(SimpleNamespace(fold_rows=fold_rows, per_class_rows=pc), True, 3, 2)
    assert cv["n_deger"] == 6 and abs(cv["dogruluk"][0] - 0.92) < 1e-9
    assert cv["dogruluk_aralik"] == (0.91, 0.93) and cv["tur_f1"] == {"A": 0.9}
    tr = TrainingResult(test_accuracy=.8, test_macro_f1=.7, per_class_report={}, confusion_matrix=np.zeros((1, 1)),
                        class_names=["A"], n_train=1, n_val=1, n_test=1, n_train_recordings=1, n_val_recordings=1,
                        n_test_recordings=1, cv_summary=cv)
    text = tr.format_summary()
    assert text.index("ÇAPRAZ DOĞRULAMA") < text.index("Test doğruluğu")
    assert "%92.0" in text and "ana sonuç" in text
    tr.cv_summary = {"hata": "x"}
    assert "çalıştırılamadı" in tr.format_summary()
