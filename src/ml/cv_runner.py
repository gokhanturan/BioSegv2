"""
Arayüzden "Sınıflandırmayı çalıştır" için çapraz doğrulama çalıştırıcısı.

capraz_dogrulama.py ile AYNI işi yapar (aynı çıktı dosyaları, aynı önbellek adları),
ancak ilerleme/iptal geri çağrılarıyla bir arka plan iş parçacığından kullanılabilir.
Veri seti CSV'sine ve eğitilmiş modele DOKUNMAZ; yalnızca `cikti_klasoru` içine
cv_*.csv ve önbellek dosyalarını yazar.
Embedding önbelleği veri değişince otomatik yenilenir (pipeline.load_or_extract_embeddings).
"""
import csv
import os
from typing import Callable, Dict, Optional

from . import pipeline
from .dataset_builder import DEFAULT_MIN_SEGMENTS
from . import improvements


class Cancelled(Exception):
    pass


def _yaz(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def run_evaluation(cikti_klasoru: str, n_splits: int = 5, n_repeats: int = 3, max_epochs: int = 100,
                   min_segment: int = DEFAULT_MIN_SEGMENTS, min_kayit: int = 2, temel_cizgi: bool = True,
                   sifir_atis: bool = False, kus_filtresi: bool = False, guclendirme: bool = True,
                   mlp_topluluk: int = 3, csv_path: Optional[str] = None,
                   progress: Optional[Callable[[str, int, int], None]] = None,
                   cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, object]:
    """Döndürür: {'dosyalar': [...], 'uyarilar': [...], 'n_ornek': int, 'n_tur': int}."""
    import numpy as np

    def P(stage, c, t):
        if cancelled and cancelled():
            raise Cancelled()
        if progress:
            progress(stage, c, t)

    def out(name):
        return os.path.join(cikti_klasoru, name)

    uyarilar, dosyalar = [], []
    # Veri izi: sonuçların HANGİ veriyle üretildiği (başlangıçta alınır; yarıda kesilirse eski iz silinmiş olur).
    from . import classification_diagnosis as _cd
    fp = _cd.data_fingerprint(csv_path or os.path.join(cikti_klasoru, "bird_sound_dataset.csv"))
    _cd.clear_fingerprint(cikti_klasoru)
    X, kept = pipeline.load_or_extract_embeddings(
        min_samples_per_class=min_segment, min_recordings_per_class=min_kayit,
        restrict_to_known_bird_species=kus_filtresi, cache_path=out("embedding_onbellek.npz"),
        progress_callback=lambda st, c, t: P(st, c, t))
    n_tur = len({s.species for s in kept})
    if n_tur < 2:
        raise ValueError("Çapraz doğrulama için en az 2 uygun tür gerekir "
                         f"(şu an {n_tur}; eşik: ≥{min_segment} segment, ≥{min_kayit} kayıt).")

    res = pipeline.run_cross_validation(X, kept, n_splits=n_splits, n_repeats=n_repeats, max_epochs=max_epochs,
                                        progress_callback=lambda d, t: P("MLP çapraz doğrulama", d, t))
    _yaz(out("cv_sonuclari.csv"), res.fold_rows); dosyalar.append("cv_sonuclari.csv")
    _yaz(out("cv_tur_bazli.csv"), res.per_class_rows); dosyalar.append("cv_tur_bazli.csv")

    if temel_cizgi:
        from . import baselines
        mfcc = baselines.load_or_extract_mfcc(kept, cache_path=out("mfcc_onbellek.npz"),
                                              progress_callback=lambda c, t: P("MFCC", c, t))
        zs = None
        if sifir_atis:
            try:
                zs = baselines.compute_zero_shot_scores(
                    kept, res.class_names, cache_path=out("sifir_atis_onbellek.npz"),
                    progress_callback=lambda c, t: P("Sıfır-atış", c, t))
                kapsam = float(np.mean(np.isfinite(zs).any(axis=0)))
                if kapsam < 1.0:
                    uyarilar.append(f"Perch etiket listesinde bulunan tür oranı: {kapsam:.2f}")
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                zs = None
                uyarilar.append(f"Sıfır-atış atlandı: {str(exc).splitlines()[0]}")
        rows = baselines.run_baselines(X, kept, mfcc=mfcc, zero_shot_scores=zs, n_splits=n_splits,
                                       n_repeats=n_repeats, progress_callback=lambda d, t: P("Temel çizgiler", d, t))
        sel_rows = []
        if guclendirme:
            from sklearn.preprocessing import LabelEncoder
            y_enc = LabelEncoder().fit_transform([s.species for s in kept])
            extra = {}
            if mlp_topluluk and mlp_topluluk > 1:
                try:
                    import tensorflow  # noqa: F401
                    extra["MLP_Topluluk"] = improvements.mlp_ensemble_proba_fn(
                        X, y_enc, len(res.class_names), n_models=mlp_topluluk, max_epochs=max_epochs)
                except ImportError:
                    uyarilar.append("MLP topluluğu atlandı (TensorFlow yok).")
            imp_rows, sel_rows = improvements.run_improvements(
                X, kept, zero_shot_scores=zs, n_splits=n_splits, n_repeats=n_repeats, extra_proba_fns=extra,
                progress_callback=lambda d, t: P("Güçlendirme denemeleri", d, t))
            rows = rows + imp_rows
            if sel_rows:
                _yaz(out("cv_secici_tahmin.csv"), sel_rows); dosyalar.append("cv_secici_tahmin.csv")
        ozet = baselines.summarize(rows)
        for opt in (False, True):
            if opt in res.summary:
                a, f = res.summary[opt]["dogruluk"], res.summary[opt]["makro_f1"]
                ozet.append({"yontem": "MLP_BirdNET", "sinif_agirligi": opt, "n_kat": n_splits * n_repeats,
                             "dogruluk_ort": a[0], "dogruluk_ss": a[1], "makro_f1_ort": f[0], "makro_f1_ss": f[1]})
        fark = []
        pairs = [("LR_BirdNET", True), ("LR_BirdNET", False), ("SifirAtis_Perch", None), ("LR_MFCC", True)]
        pairs += [(n, None) for n in sorted({r["yontem"] for r in rows}
                                            & {"Prototip_BirdNET", "Topluluk_BirdNET", "LR_SifirAtis_Perch", "MLP_Topluluk"})]
        for ad, cw in pairs:
            for metrik in ("dogruluk", "makro_f1"):
                try:
                    fark.append(baselines.paired_comparison(res.fold_rows, rows, ad, cw, metric=metrik))
                except ValueError:
                    pass
        if fark:
            _yaz(out("cv_eslestirilmis_fark.csv"), fark); dosyalar.append("cv_eslestirilmis_fark.csv")
        _yaz(out("cv_temel_cizgi.csv"), rows); dosyalar.append("cv_temel_cizgi.csv")
        _yaz(out("cv_yontem_ozeti.csv"), ozet); dosyalar.append("cv_yontem_ozeti.csv")
    _cd.write_fingerprint(cikti_klasoru, fp)
    if fp:
        dosyalar.append(_cd.FINGERPRINT_FILE)
    return {"dosyalar": dosyalar, "uyarilar": uyarilar, "n_ornek": len(kept), "n_tur": n_tur}
