"""richness_inference: pencere tespitlerinden nokta başına tür listesi ve doğrulama (sahte sınıflandırıcıyla)."""
import numpy as np
import pytest

from src.ml import richness_inference as RI
from src.ml.dataset_builder import TrainingSample

SR = 32000
FREQS = [1000.0, 2500.0, 4500.0]


class FakeClassifier:
    """Pencerenin üç banttaki enerjisini normalize ederek (3,) 'gömme' üretir; çağrı sayısını tutar."""
    def __init__(self):
        self.calls = 0

    def predict_window_embedding(self, chunk):
        self.calls += 1
        spec = np.abs(np.fft.rfft(chunk))
        f = np.fft.rfftfreq(len(chunk), 1 / SR)
        e = np.array([spec[(f > fr - 200) & (f < fr + 200)].sum() for fr in FREQS])
        return e / (e.sum() + 1e-9) * 10.0


def _wav(path, tone_idx, seg_seconds=10, total_seconds=30, seed=0):
    """Türler sırayla birer seg_seconds'lık parça halinde öter (aynı pencerede karışmaz); gerisi sessiz."""
    import soundfile as sf
    rng = np.random.default_rng(seed)
    t = np.arange(int(total_seconds * SR)) / SR
    y = 0.001 * rng.standard_normal(len(t))
    n = int(seg_seconds * SR)
    for j, i in enumerate(tone_idx):
        sl = slice(j * n, (j + 1) * n)
        y[sl] += 0.5 * np.sin(2 * np.pi * FREQS[i] * t[sl])
    sf.write(path, y.astype(np.float32), SR)


def test_count_independent_and_detection_incidence():
    assert RI.count_independent([0, 2.5, 5, 7.5, 20], min_gap=5.0) == 3
    assert RI.count_independent([], 5.0) == 0
    rec_idx = np.array([0, 0, 0, 1, 1])
    start = np.array([0, 5, 10, 0, 2.5])
    pred = np.array([1, 1, 2, 1, 1])
    conf = np.array([.9, .9, .9, .9, .4])
    # nokta A: kayıt0 (tür1 x2, tür2 x1); nokta B: kayıt1 (tür1 x1 güvenli + 1 düşük güvenli)
    inc1 = RI.detection_incidence(rec_idx, start, pred, conf, ["A", "B"], ["A", "B"], 3, tau=0.5, k=1)
    assert inc1.tolist() == [[False, True, True], [False, True, False]]
    inc2 = RI.detection_incidence(rec_idx, start, pred, conf, ["A", "B"], ["A", "B"], 3, tau=0.5, k=2)
    assert inc2.tolist() == [[False, True, False], [False, False, False]]   # k=2: yalnızca A'daki tür1


def test_scan_recording_uses_cache(tmp_path):
    p = str(tmp_path / "a.wav")
    _wav(p, [0])
    clf = FakeClassifier()
    s1, e1 = RI.scan_recording(clf, p, hop=5.0, cache_dir=str(tmp_path / "c"))
    n = clf.calls
    assert n == len(s1) and e1.shape == (len(s1), 3)
    s2, e2 = RI.scan_recording(clf, p, hop=5.0, cache_dir=str(tmp_path / "c"))
    assert clf.calls == n and np.allclose(e1, e2)          # ikinci çağrı önbellekten


def test_resolve_audio_falls_back_to_index(tmp_path):
    sub = tmp_path / "x" / "y"; sub.mkdir(parents=True)
    f = sub / "Kayit_1.WAV"; f.write_bytes(b"x")
    idx = RI.build_audio_index(str(tmp_path))
    assert RI.resolve_audio("C:/yok/Kayit_1.WAV", "Kayit_1", idx) == str(f)
    assert RI.resolve_audio("C:/yok/baska.wav", "baska", idx) is None


def test_end_to_end_validation_recovers_expert_lists(tmp_path):
    rng = np.random.default_rng(1)
    clf = FakeClassifier()
    points = {"P1": [0, 1], "P2": [1, 2], "P3": [0, 2], "P4": [0, 1, 2]}
    records, kept, X, point_of_rec = [], [], [], {}
    for pi, (pt, sp) in enumerate(points.items()):
        for r in range(3):
            rid = f"{pt}_{r}"
            path = str(tmp_path / f"{rid}.wav")
            _wav(path, sp, seed=pi * 10 + r)
            records.append({"rec_id": rid, "point": pt, "path": path})
            point_of_rec[rid] = pt
            for c in sp:   # uzmanın etiketlediği segmentler: türün saf tonu
                v = np.zeros(3); v[c] = 10.0
                X.append(v + 0.3 * rng.standard_normal(3))
                kept.append(TrainingSample(filepath=f"{rid}_{c}.wav", species=f"T{c}", recording_id=rid))
    X = np.array(X)
    class_names = sorted({s.species for s in kept})
    table = RI.build_window_table(records, clf, hop=5.0, cache_dir=str(tmp_path / "cache"))
    assert not table["failures"] and len(table["start"]) == 12 * 6
    preds = [RI.out_of_fold_window_proba(table, X, kept, class_names, method="lr", n_splits=3, seed=42)]
    rows = RI.validate_scan_vs_expert(table, preds, kept, class_names, point_of_rec,
                                      taus=[0.5, 0.9], ks=[1, 2], default_tau=0.5, default_k=2)
    assert len(rows) == 4 and sum(r["onceden_belirlenen"] for r in rows) == 1
    default = [r for r in rows if r["onceden_belirlenen"]][0]
    assert default["nokta_sayisi"] == 4
    assert default["var_yok_duyarlilik"] >= 0.9            # tonlu pencereler doğru türe gider
    assert default["tur_uzman"] == 3 and default["tur_model"] == 3
    ozet = RI.summarize(rows)
    assert len(ozet) == 4 and "var_yok_jaccard_ort" in ozet[0]


def test_apply_and_grid_and_lists(tmp_path):
    rng = np.random.default_rng(2)
    clf = FakeClassifier()
    recs, kept, X = [], [], []
    for c in range(3):
        for r in range(3):
            rid = f"T{c}_{r}"
            v = np.zeros(3); v[c] = 10.0
            X.append(v + 0.3 * rng.standard_normal(3))
            kept.append(TrainingSample(filepath=rid, species=f"T{c}", recording_id=rid))
    X = np.array(X)
    for i, (pt, sp) in enumerate([("A", [0]), ("B", [0, 1]), ("C", [2]), ("D", [1, 2])]):
        path = str(tmp_path / f"n{i}.wav"); _wav(path, sp, seed=i)
        recs.append({"rec_id": f"n{i}", "point": pt, "path": path})
    table = RI.build_window_table(recs, clf, 5.0, str(tmp_path / "c"))
    pred, conf = RI.apply_model_to_table(table, X, kept, ["T0", "T1", "T2"], method="lr")
    grid = RI.richness_grid(table, pred, conf, ["T0", "T1", "T2"], [0.5], [2], 0.5, 2)
    assert grid[0]["nokta_sayisi"] == 4 and grid[0]["belirlenen_tur"] == 3 and grid[0]["onceden_belirlenen"]
    lst = RI.species_lists(table, pred, conf, ["T0", "T1", "T2"], 0.5, 2)
    by_point = {}
    for r in lst:
        by_point.setdefault(r["nokta"], set()).add(r["tur"])
    assert by_point["A"] == {"T0"} and by_point["B"] == {"T0", "T1"} and by_point["D"] == {"T1", "T2"}


def test_validation_only_uses_scanned_recordings(tmp_path):
    """Ham kayıtların yalnızca bir kısmı varsa, taranmayan kayıtların etiketleri 'kaçırılan tür' sayılmaz."""
    rng = np.random.default_rng(3)
    clf = FakeClassifier()
    points = {"P1": [0, 1], "P2": [1, 2], "P3": [0, 2], "P4": [0, 1, 2]}
    records, kept, X, por = [], [], [], {}
    for pi, (pt, sp) in enumerate(points.items()):
        for r in range(3):
            rid = f"{pt}_{r}"
            por[rid] = pt
            path = str(tmp_path / f"{rid}.wav")
            _wav(path, sp, seed=pi * 10 + r)
            if r < 2:                       # her noktanın 3. kaydının ham sesi YOK
                records.append({"rec_id": rid, "point": pt, "path": path})
            for c in sp:
                v = np.zeros(3); v[c] = 10.0
                X.append(v + 0.3 * rng.standard_normal(3))
                kept.append(TrainingSample(filepath=f"{rid}_{c}", species=f"T{c}", recording_id=rid))
    X = np.array(X)
    cn = sorted({s.species for s in kept})
    table = RI.build_window_table(records, clf, 5.0, str(tmp_path / "c"))
    preds = [RI.out_of_fold_window_proba(table, X, kept, cn, method="lr", n_splits=3, seed=42)]
    rows = RI.validate_scan_vs_expert(table, preds, kept, cn, por, [0.5], [2], 0.5, 2)
    assert rows[0]["var_yok_duyarlilik"] >= 0.9 and rows[0]["sahte_tur_nokta_cifti"] == 0
