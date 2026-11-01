"""
src/ml/pipeline.py
================
Sınıflandırıcı eğitiminin tüm adımlarını (veri seti oluşturma -> embedding
çıkarma -> grup-farkında bölme -> MLP eğitimi -> değerlendirme -> modeli
kaydetme) tek bir fonksiyonda birleştiren orkestrasyon katmanı. PyQt5'ten
bağımsızdır (UI olmadan da test edilebilir/çağrılabilir); ilerleme
bildirimleri bir callback üzerinden yapılır.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from ..detection.detector import get_shared_perch_classifier
from ..data.config import get_project_root as get_src_root
from .dataset_builder import build_training_samples, DEFAULT_MIN_SEGMENTS
from .embedding_extraction import extract_embeddings_for_samples
from . import birdnet_embedding as bne
from .olcutler import agirlikli_f1, havuzlanmis, makro_pr_auc
from .train_classifier import (
    group_aware_split, group_kfold_splits, build_and_train_mlp, evaluate_model, save_model, TrainingResult,
)

DEFAULT_MODEL_SAVE_DIR_NAME = "trained_classifier"

ProgressCallback = Callable[[str, int, int], None]  # (aşama, mevcut, toplam)


class InsufficientDataError(RuntimeError):
    """Eğitim için yeterli/uygun veri bulunamadığında fırlatılır."""


def _load_model_with_progress(classifier, report) -> None:
    """
    Perch 2.0 modelini AYRI bir aşamada ("model") yükler (yalnızca backend="perch" araştırma yolunda).
    Test sahteleri (ensure_loaded yok / classifier None) sessizce atlanır.
    """
    ensure = getattr(classifier, "ensure_loaded", None)
    if ensure is None:
        return
    from ..detection.perch_classifier import PerchModelLoadError
    report("model", 0, 1)
    try:
        ensure()
    except PerchModelLoadError as exc:
        raise InsufficientDataError(f"Perch 2.0 modeli yüklenemedi.\n\n{exc}") from exc
    report("model", 1, 1)


EMBEDDING_BACKENDS = ("birdnet", "perch", "combined")
DEFAULT_BACKEND = "birdnet"
COMBINED_PARTS = ("birdnet", "perch")      # "combined": bu gömmelerin segment bazında yan yana eklenmesi (sıra sabit)


def _make_extractor(backend: str, report):
    """(samples, progress_cb) -> (X, kept, n_fail, last_error) döndüren bir çıkarıcı verir. Model AYRI bir
    "model" aşamasında bir kez yüklenir; yükleme hatası tek kez, açık bir mesajla bildirilir."""
    if backend not in EMBEDDING_BACKENDS:
        raise ValueError(f"Bilinmeyen gömme kaynağı: {backend!r} (geçerli: {EMBEDDING_BACKENDS})")
    if backend == "combined":
        raise ValueError("'combined' tek bir çıkarıcı değildir; bileşenleri _extract_combined ile çıkarılır.")
    if backend == "birdnet":
        def _birdnet(samples, cb):
            try:
                return bne.extract_embeddings_for_samples(
                    samples, progress_callback=cb, stage_callback=report)
            except bne.BirdNETModelLoadError as exc:
                raise InsufficientDataError(f"BirdNET modeli yüklenemedi.\n\n{exc}") from exc
        return _birdnet
    classifier = get_shared_perch_classifier()
    _load_model_with_progress(classifier, report)
    return lambda samples, cb: extract_embeddings_for_samples(classifier, samples, progress_callback=cb)


def _extraction_failure_message(n_samples: int, last_error, backend: str) -> str:
    ad = {"birdnet": "BirdNET 2.4", "perch": "Perch 2.0",
          "combined": "BirdNET 2.4 ya da Perch 2.0"}.get(backend, "Öznitelik çıkarıcı")
    return (f"{n_samples} dosyanın HİÇBİRİNDEN embedding çıkarılamadı. Bu genellikle {ad} modelinin "
            "yüklenememesi ya da ses dosyalarının okunamaması gibi SİSTEMİK bir sorunu gösterir "
            "(ayrıntı için aşağıdaki son hata mesajına bakın).\n\n"
            f"Son karşılaşılan hata: {last_error}")


def _resolve_trainer(trainer: str):
    if trainer == "mlp":
        return build_and_train_mlp          # modül düzeyinde ad: testlerde değiştirilebilir
    raise ValueError(f"Bilinmeyen eğitici: {trainer!r} (yalnızca 'mlp' desteklenir)")


def _preproc_signature(backend: str) -> str:
    """Gömme önbelleğinin ön işleme imzası. BirdNET/Perch önbellekleri imzasızdır (boş dize)."""
    return ""


def _file_sizes(samples) -> np.ndarray:
    out = []
    for x in samples:
        try:
            out.append(os.path.getsize(x.filepath))
        except OSError:
            out.append(-1)
    return np.array(out, dtype=np.int64)


def _cache_matches(data, backend: str, keys: np.ndarray, samples) -> bool:
    """Önbellek yalnızca örnek listesi, kaynak, ön işleme imzası VE (kayıtlıysa) dosya boyutları eşleşirse kullanılır.
    Boyut kaydı olmayan eski önbellekler önceki davranışla (boyut denetimi yok) kabul edilir."""
    if "keys" not in data.files or _cache_backend(data) != backend or not np.array_equal(data["keys"], keys):
        return False
    stored_sig = str(data["params"]) if "params" in data.files else ""
    if stored_sig != _preproc_signature(backend):
        return False
    if "sizes" in data.files and not np.array_equal(data["sizes"], _file_sizes(samples)):
        return False
    return True


def _save_cache(path: str, X: np.ndarray, backend: str, kept) -> None:
    np.savez(path, X=X, backend=np.array(backend), keys=_sample_keys(kept),
             params=np.array(_preproc_signature(backend)), sizes=_file_sizes(kept))


def _sample_keys(samples) -> np.ndarray:
    return np.array([f"{x.filepath}||{x.species}||{x.recording_id}" for x in samples])


def _extract_combined(samples, component_caches, report, progress):
    """Birleşik gömme (early fusion): BirdNET 2.4 ve Perch 2.0 gömmeleri AYRI ayrı (kendi önbelleğiyle) çıkarılır,
    iki tarafta da bulunan segmentler BirdNET sırasına göre hizalanır ve yan yana eklenir: [BirdNET | Perch].
    Hiçbir ölçek/istatistik veri kümesinden öğrenilmez (ham gömmeler eklenir; tek-model MLP'leriyle aynı eğitici),
    böylece tek-model sonuçlarından farkı YALNIZCA birleştirmedir.
    `component_caches`: {parça: (önbellek_yolu, yalnızca_okunur_önbellekler)}; eksik parça = önbelleksiz."""
    from .dataset_builder import TrainingSample
    comp = component_caches or {}
    n_parts = len(COMBINED_PARTS)
    parts = {}
    for i, part in enumerate(COMBINED_PARTS):
        cache, reads = comp.get(part, (None, ()))
        X, kept, n_fail, last_error = _extract_with_cache(
            samples, part, cache, report,
            lambda d, t, i=i: progress(i * t + d, n_parts * t), read_caches=tuple(reads or ()))
        parts[part] = (X, kept, n_fail, last_error)
    kept_by_part = {p: {_sample_keys([s])[0]: j for j, s in enumerate(v[1])} for p, v in parts.items()}
    first = COMBINED_PARTS[0]
    common = [(k, j) for k, j in kept_by_part[first].items() if all(k in kept_by_part[p] for p in COMBINED_PARTS[1:])]
    last_error = next((v[3] for v in parts.values() if v[3]), None)
    n_fail = len(samples) - len(common)
    dim = sum(parts[p][0].shape[1] for p in COMBINED_PARTS if parts[p][0].ndim == 2 and parts[p][0].shape[0] > 0)
    if not common:
        return np.zeros((0, dim), np.float32), [], len(samples), last_error
    blocks = [np.asarray(parts[first][0])[[j for _k, j in common]]]
    for p in COMBINED_PARTS[1:]:
        blocks.append(np.asarray(parts[p][0])[[kept_by_part[p][k] for k, _j in common]])
    X = np.hstack(blocks).astype(np.float32)
    kept = [TrainingSample(filepath=f, species=sp, recording_id=r)
            for f, sp, r in (k.split("||") for k, _j in common)]
    return X, kept, n_fail, last_error


def _extract_with_cache(samples, backend: str, cache_path: Optional[str], report, progress, read_caches=(),
                        component_caches=None):
    """(X, kept, n_fail, last_error). Örnek listesi ve gömme kaynağı eşleşen bir önbellek varsa model HİÇ
    yüklenmeden ondan okunur: önce `cache_path`, sonra yalnızca-okunur `read_caches` (örn. araştırma betiklerinin
    yazdığı eski önbellek). Aksi halde çıkarılır ve (hatasızsa) `cache_path`'e yazılır. `cache_path=None`: yazma yok.
    backend="combined": bkz. `_extract_combined` (`component_caches` ile)."""
    from .dataset_builder import TrainingSample
    if backend == "combined":
        return _extract_combined(samples, component_caches, report, progress)
    keys = _sample_keys(samples)
    for path in [p for p in [cache_path, *read_caches] if p]:
        if not os.path.isfile(path):
            continue
        try:
            data = np.load(path, allow_pickle=False)
            hit = _cache_matches(data, backend, keys, samples)
        except Exception:  # noqa: BLE001  (bozuk/okunamayan önbellek yok sayılır)
            continue
        if hit:
            kept = [TrainingSample(filepath=f, species=sp, recording_id=r)
                    for f, sp, r in (k.split("||") for k in data["keys"])]
            X = data["X"]
            if cache_path and path != cache_path and not os.path.isfile(cache_path):
                np.savez(cache_path, **{k: data[k] for k in data.files})   # olduğu gibi kopyalanır (boyut kaydı uydurulmaz)
            return X, kept, 0, None
    extract = _make_extractor(backend, report)
    X, kept, n_fail, last_error = extract(samples, progress)
    if cache_path and n_fail == 0 and len(kept) > 0:
        _save_cache(cache_path, X, backend, kept)
    return X, kept, n_fail, last_error


def run_full_training_pipeline(
    min_samples_per_class: int = DEFAULT_MIN_SEGMENTS,
    min_recordings_per_class: int = 2,
    test_size: float = 0.2,
    val_size: float = 0.15,
    max_epochs: int = 100,
    random_state: int = 42,
    restrict_to_known_bird_species: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    csv_path: Optional[str] = None,
    classification_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    use_class_weights: bool = False,
    cv_folds: int = 0,
    cv_repeats: int = 2,
    backend: str = DEFAULT_BACKEND,
    embedding_cache_path: Optional[str] = None,
    embedding_read_caches: tuple = (),
    trainer: str = "mlp",
    learning_curve: bool = False,
    point_csv_path: Optional[str] = None,
    component_caches: Optional[dict] = None,
) -> "PipelineResult":
    """
    Tüm eğitim hattını çalıştırır. Aşamalar sırasıyla:
    "samples" (veri seti taranıyor) -> "model" -> "embedding" (BirdNET 2.4
    ile embedding çıkarılıyor) -> "training" (MLP eğitiliyor) -> "done".

    `csv_path`/`classification_dir`/`save_dir` verilmezse, uygulamanın
    gerçek konumlarına göre otomatik çözümlenir; testler için elle
    belirtilebilir.

    `min_recordings_per_class`: bir türün eğitime dahil edilmesi için en
    az kaç FARKLI orijinal kayıttan gelmesi gerektiği (varsayılan 2 — en
    az 1'i train'e, 1'i test'e ayrılabilsin diye). Bu, `min_samples_per_class`
    (segment sayısı) şartından AYRI ve tamamlayıcı bir şart.

    `cv_folds` (>=2 ise): son model eğitilmeden ÖNCE aynı MLP ile kayıt-gruplu tekrarlı çapraz
    doğrulama (`cv_folds` kat x `cv_repeats` tekrar) çalıştırılır; sonuç `TrainingResult.cv_summary`
    içinde ve özet metninde ANA sonuç olarak yer alır. Tek bir bölme (tek test kümesi) küçük veride
    kararsızdır; bir kayıt sonucu belirleyebilir. Varsayılan 0 = kapalı.

    `restrict_to_known_bird_species`: True ise, projenin 518 türlük kuş
    veritabanında (`bird_species_data.json`) BULUNMAYAN türler eğitim
    setinden çıkarılır — BirdNET'in kurbağa/böcek/memeli de
    önerebilmesine karşı KESİN bir "sadece kuş" garantisi.
    """

    def report(stage: str, current: int, total: int):
        if progress_callback is not None:
            progress_callback(stage, current, total)

    report("samples", 0, 1)
    samples, excluded_species, raw_counts = build_training_samples(
        csv_path=csv_path, classification_dir=classification_dir,
        min_samples_per_class=min_samples_per_class,
        min_recordings_per_class=min_recordings_per_class,
        restrict_to_known_bird_species=restrict_to_known_bird_species,
    )

    unique_species = sorted({s.species for s in samples})
    if len(samples) == 0:
        raise InsufficientDataError(
            "Eğitim için uygun örnek bulunamadı. Önce 'Sınıflandırma Klasörleri' penceresini "
            "çalıştırdığınızdan ve yeterli sayıda etiketlenmiş segmentiniz olduğundan emin olun."
        )
    if len(unique_species) < 2:
        raise InsufficientDataError(
            f"Eğitim için en az 2 farklı tür gerekli (min. {min_samples_per_class} örnek/tür "
            f"eşiğini geçen tür sayısı: {len(unique_species)}). Eşiği düşürmeyi ya da daha "
            "fazla veri toplamayı deneyin."
        )
    report("samples", 1, 1)

    def embedding_progress(done: int, total: int):
        report("embedding", done, total)

    X, kept_samples, n_failures, last_error = _extract_with_cache(
        samples, backend, embedding_cache_path, report, embedding_progress,
        read_caches=tuple(embedding_read_caches or ()), component_caches=component_caches)

    if len(kept_samples) == 0:
        raise InsufficientDataError(_extraction_failure_message(len(samples), last_error, backend))

    if n_failures > 0:
        excluded_species = list(excluded_species) + [
            f"(not: {n_failures} dosya embedding çıkarılırken atlandı — son hata: {last_error})"
        ]

    from sklearn.preprocessing import LabelEncoder

    y_species = [s.species for s in kept_samples]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_species)
    class_names = list(label_encoder.classes_)

    split = group_aware_split(kept_samples, test_size=test_size, val_size=val_size, random_state=random_state)

    if split.species_missing_in_test:
        excluded_species = list(excluded_species) + [
            f"(not: test kümesinde HİÇ görünmeyen {len(split.species_missing_in_test)} tür için test doğruluğu ölçülemedi: "
            + ", ".join(split.species_missing_in_test[:5]) + (" ..." if len(split.species_missing_in_test) > 5 else "") + ")"
        ]

    if len(split.test_idx) == 0:
        raise InsufficientDataError(
            "Hiçbir tür için ayrı bir TEST kaydı oluşturulamadı — bu, geriye kalan türlerin "
            "her birinin sadece TEK bir farklı kayıttan geldiğini gösteriyor (bir kayıt ya "
            "eğitime ya teste gidebilir, ikisine birden gidemez — aksi veri sızıntısı olur). "
            "Güvenilir bir değerlendirme için her tür için en az 2 FARKLI kayıttan segment "
            "toplamanız gerekiyor."
        )

    cv_summary = None
    lc_result = None
    zenginlik = None
    if cv_folds and cv_folds >= 2:
        try:
            n_rec = len({s.recording_id for s in kept_samples})
            k = min(int(cv_folds), n_rec)
            reps = max(1, int(cv_repeats))
            if k >= 2:
                report("cv", 0, max(1, k * reps))
                cv_res = run_cross_validation(
                    X, kept_samples, n_splits=k, n_repeats=reps,
                    class_weight_options=(bool(use_class_weights),), max_epochs=max_epochs,
                    base_seed=random_state, trainer=trainer, progress_callback=lambda d, t: report("cv", d, t))
                cv_summary = summarize_cv(cv_res, bool(use_class_weights), k, reps)
                zenginlik = _richness_agreement(cv_res, kept_samples, bool(use_class_weights),
                                                point_csv_path or csv_path)
                if learning_curve:
                    try:
                        from .learning_curve import run_learning_curve
                        report("lc", 0, 1)
                        lc_result = run_learning_curve(
                            X, kept_samples, n_splits=k, n_repeats=reps, max_epochs=max_epochs,
                            base_seed=random_state, use_class_weights=bool(use_class_weights), trainer=trainer,
                            progress_callback=lambda d, t: report("lc", d, t),
                            full_rows=[r for r in cv_res.fold_rows if bool(r["sinif_agirligi"]) == bool(use_class_weights)])
                    except Exception as exc:  # noqa: BLE001  (ek analiz; eğitimi asla durdurmaz)
                        lc_result = {"hata": (str(exc).splitlines() or [type(exc).__name__])[0]}
        except Exception as exc:  # noqa: BLE001  (ek analiz; eğitimi asla durdurmaz)
            cv_summary = {"hata": (str(exc).splitlines() or [type(exc).__name__])[0]}

    X_train, y_train = X[split.train_idx], y[split.train_idx]
    X_val, y_val = X[split.val_idx], y[split.val_idx]
    X_test, y_test = X[split.test_idx], y[split.test_idx]

    report("training", 0, max(1, int(max_epochs)))

    def training_progress(epoch: int, total_epochs: int, logs: dict):
        report("training", epoch, total_epochs)

    train_fn = _resolve_trainer(trainer)
    model, history = train_fn(
        X_train, y_train, X_val, y_val,
        n_classes=len(class_names),
        embedding_dim=X.shape[1] if len(X) else bne.EMBEDDING_DIM,
        max_epochs=max_epochs,
        random_state=random_state,
        progress_callback=training_progress,
        use_class_weights=use_class_weights,
    )

    eval_result = evaluate_model(model, X_test, y_test, class_names=class_names)
    try:   # kayıt bazlı hata / tür varlığı analizi için segment başına tahminler
        from .recording_level import build_prediction_rows
        test_predictions = build_prediction_rows(
            kept_samples, split.test_idx, y_test, model.predict(X_test, verbose=0), class_names)
    except Exception:  # noqa: BLE001  (ek analiz; eğitimi asla durdurmaz)
        test_predictions = None

    def _n_recordings(indices: np.ndarray) -> int:
        return len({kept_samples[i].recording_id for i in indices})

    result = TrainingResult(
        test_accuracy=eval_result["accuracy"],
        test_macro_f1=eval_result["macro_f1"],
        per_class_report=eval_result["report"],
        confusion_matrix=eval_result["confusion_matrix"],
        class_names=class_names,
        n_train=len(split.train_idx),
        n_val=len(split.val_idx),
        n_test=len(split.test_idx),
        n_train_recordings=_n_recordings(split.train_idx),
        n_val_recordings=_n_recordings(split.val_idx),
        n_test_recordings=_n_recordings(split.test_idx),
        excluded_species=excluded_species,
        history=history,
        test_predictions=test_predictions,
        cv_summary=cv_summary,
        learning_curve=lc_result,
        zenginlik_uyumu=zenginlik,
    )

    save_dir = save_dir or os.path.join(get_src_root(), DEFAULT_MODEL_SAVE_DIR_NAME)
    save_model(model, class_names, save_dir)

    report("done", 1, 1)

    return PipelineResult(training_result=result, save_dir=save_dir, raw_species_counts=raw_counts)


@dataclass
class PipelineResult:
    training_result: TrainingResult
    save_dir: str
    raw_species_counts: dict


# ---------------------------------------------------------------------------
# Tekrarlı grup (kayıt) bazlı k-katlı çapraz doğrulama
# ---------------------------------------------------------------------------

@dataclass
class CrossValidationResult:
    fold_rows: List[dict]          # her (tekrar, kat, ayar) için doğruluk / makro F1 / destek
    per_class_rows: List[dict]     # birleştirilmiş (tüm katlar+tekrarlar) tür bazlı P/R/F1/destek, ayar bazında
    summary: dict                  # ayar -> {metrik: (ortalama, std)}; ayrıca eşleştirilmiş fark
    class_names: List[str]
    n_samples: int
    n_recordings: int
    confusion: Optional[dict] = None   # ayar -> birleştirilmiş (n_sınıf x n_sınıf) karışıklık matrisi (satır=gerçek)
    oof: Optional[dict] = None         # ayar -> [(tahmin[N], güven[N]) her tekrar için]; her segment tekrarda bir test katında
    split_hash: Optional[str] = None   # bölmelerin parmak izi: aynı değer = aynı (tekrar, kat) test kayıtları ve sınıf kümesi


def fold_extra_metrics(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> tuple:
    return agirlikli_f1(y_true, np.argmax(proba, axis=1)), makro_pr_auc(y_true, proba)


pooled_metrics = havuzlanmis


def _richness_agreement(cv_res, kept_samples, use_class_weights: bool, csv_path: Optional[str]):
    """Çapraz doğrulama tahminlerinden nokta bazlı tür listesi uyumu. Noktası (Habitat_ID) yoksa None."""
    try:
        from . import richness_validation as RV
        from .dataset_builder import get_csv_root
        path = csv_path or os.path.join(get_csv_root(), "bird_sound_dataset.csv")
        point_map = RV.load_point_map(path)
        if not (set(s.recording_id for s in kept_samples) & set(point_map)):
            return None
        try:
            info = RV.load_point_info(path)
        except Exception:  # noqa: BLE001  (habitat bilgisi yalnızca açıklayıcı)
            info = None
        return RV.agreement_from_oof(kept_samples, point_map, cv_res.class_names, cv_res.oof[bool(use_class_weights)],
                                     point_info=info)
    except Exception as exc:  # noqa: BLE001  (ek analiz; eğitimi asla durdurmaz)
        return {"hata": (str(exc).splitlines() or [type(exc).__name__])[0]}


def summarize_cv(res: "CrossValidationResult", class_weight: bool, n_splits: int, n_repeats: int) -> dict:
    """CrossValidationResult -> arayüz/özet için sözlük (ortalama, s.s., kat aralığı, tür başına F1)."""
    rows = [r for r in res.fold_rows if bool(r["sinif_agirligi"]) == bool(class_weight)]
    out = {"n_kat": int(n_splits), "n_tekrar": int(n_repeats), "n_deger": len(rows), "sinif_agirligi": bool(class_weight)}
    for key in ("dogruluk", "makro_f1"):
        v = np.array([r[key] for r in rows], dtype=float)
        out[key] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
        out[key + "_aralik"] = (float(v.min()), float(v.max()))
    for key in ("agirlikli_f1", "makro_pr_auc"):
        v = np.array([r[key] for r in rows if key in r], dtype=float)
        v = v[~np.isnan(v)]
        if len(v):
            out[key] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    confusion = getattr(res, "confusion", None)
    if confusion and bool(class_weight) in confusion:
        out["karisiklik"] = np.asarray(confusion[bool(class_weight)]).astype(int).tolist()
        out["siniflar"] = list(res.class_names)
        out["havuzlanmis"] = pooled_metrics(confusion[bool(class_weight)])
    out["tur_f1"] = {r["tur"]: float(r["f1"]) for r in res.per_class_rows if bool(r["sinif_agirligi"]) == bool(class_weight)}
    # Modeller arası eşleştirilmiş anlamlılık testi için kat bazlı sonuçlar + bölme parmak izi
    need = ("tekrar", "kat", "dogruluk", "makro_f1", "n_train", "n_test")
    out["kat_satirlari"] = [
        {"tekrar": int(r["tekrar"]), "kat": int(r["kat"]), "dogruluk": float(r["dogruluk"]),
         "makro_f1": float(r["makro_f1"]), "n_train": int(r["n_train"]), "n_val": int(r.get("n_val", 0)),
         "n_test": int(r["n_test"])} for r in rows if all(k in r for k in need)]
    out["bolme_izi"] = getattr(res, "split_hash", None)
    return out


def _split_fingerprint(class_names, per_fold_test_records) -> str:
    """Bölmelerin parmak izi: sınıf kümesi + her (tekrar, kat) için SIRALI test kayıtları. İki model aynı değere
    sahipse aynı katlarda, aynı test kayıtlarıyla ve aynı sınıf kümesiyle değerlendirilmiştir (eşleştirilebilir)."""
    import hashlib
    h = hashlib.sha1()
    h.update("|".join(class_names).encode("utf-8"))
    for (rep, fold), recs in sorted(per_fold_test_records.items()):
        h.update(f"#{rep}:{fold}:".encode("utf-8"))
        h.update("\x1f".join(sorted(recs)).encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_backend(data) -> str:
    """Önbellek dosyasının gömme kaynağı; etiketsiz (eski) önbellekler Perch'e aittir."""
    return str(data["backend"]) if "backend" in data.files else "perch"


def load_or_extract_embeddings(
    min_samples_per_class: int = DEFAULT_MIN_SEGMENTS,
    min_recordings_per_class: int = 2,
    restrict_to_known_bird_species: bool = False,
    csv_path: Optional[str] = None,
    classification_dir: Optional[str] = None,
    cache_path: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
    backend: str = DEFAULT_BACKEND,
):
    """
    (X, kept_samples) döndürür. `cache_path` verilirse ve o dosya mevcutsa embedding'ler oradan okunur
    (örnek listesi VE gömme kaynağı eşleşiyorsa); aksi halde `backend` ("birdnet" varsayılan, "perch"
    araştırma betikleri için) ile çıkarılıp o dosyaya kaydedilir. Önbellek hangi kaynağa aitse
    dosyada etiketlenir; farklı kaynaktan kalan bir önbellek ASLA yeniden kullanılmaz.
    """
    from .dataset_builder import TrainingSample

    def report(stage, cur, tot):
        if progress_callback is not None:
            progress_callback(stage, cur, tot)

    if backend not in EMBEDDING_BACKENDS:
        raise ValueError(f"Bilinmeyen gömme kaynağı: {backend!r} (geçerli: {EMBEDDING_BACKENDS})")
    samples, excluded, _ = build_training_samples(
        csv_path=csv_path, classification_dir=classification_dir,
        min_samples_per_class=min_samples_per_class,
        min_recordings_per_class=min_recordings_per_class,
        restrict_to_known_bird_species=restrict_to_known_bird_species,
    )
    if len(samples) == 0:
        raise InsufficientDataError("Eğitim için uygun örnek bulunamadı.")

    keys = np.array([f"{x.filepath}||{x.species}||{x.recording_id}" for x in samples])
    if cache_path and os.path.isfile(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        if _cache_matches(data, backend, keys, samples):
            kept = [TrainingSample(filepath=f, species=sp, recording_id=r)
                    for f, sp, r in (k.split("||") for k in data["keys"])]
            return data["X"], kept

    extract = _make_extractor(backend, report)
    X, kept, n_fail, last_error = extract(samples, lambda d, t: report("embedding", d, t))
    if len(kept) == 0:
        raise InsufficientDataError(_extraction_failure_message(len(samples), last_error, backend))
    if cache_path and n_fail == 0:
        _save_cache(cache_path, X, backend, kept)
    return X, kept


def run_cross_validation(
    X: np.ndarray,
    kept_samples: list,
    n_splits: int = 5,
    n_repeats: int = 3,
    class_weight_options=(False, True),
    max_epochs: int = 100,
    base_seed: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    trainer: str = "mlp",
) -> CrossValidationResult:
    """
    Tekrarlı, kayıt bazlı (grup-farkında), tür-dengeli k-katlı çapraz doğrulama.
    Her (tekrar, kat) için AYNI bölmede `class_weight_options` içindeki her ayar
    eğitilir (eşleştirilmiş karşılaştırma). Test kümesi hiçbir ayar seçimi için
    kullanılmaz; doğrulama kümesi yalnızca erken durdurma içindir.
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import precision_recall_fscore_support

    train_fn = _resolve_trainer(trainer)
    le = LabelEncoder()
    y = le.fit_transform([s.species for s in kept_samples])
    class_names = list(le.classes_)
    n_classes = len(class_names)

    fold_rows: List[dict] = []
    pooled = {opt: ([], []) for opt in class_weight_options}   # ayar -> (y_true, y_pred)
    n_all = len(y)
    oof = {bool(opt): [(np.full(n_all, -1, dtype=int), np.zeros(n_all)) for _ in range(n_repeats)]
           for opt in class_weight_options}
    total = n_repeats * n_splits * len(class_weight_options)
    done = 0
    test_records: dict = {}

    for rep in range(n_repeats):
        seed = base_seed + rep
        splits = group_kfold_splits(kept_samples, n_splits=n_splits, random_state=seed)
        for fold_i, sp in enumerate(splits):
            test_records[(rep + 1, fold_i + 1)] = {kept_samples[i].recording_id for i in sp.test_idx}
            for opt in class_weight_options:
                X_val = X[sp.val_idx] if len(sp.val_idx) else None
                y_val = y[sp.val_idx] if len(sp.val_idx) else None
                model, _ = train_fn(
                    X[sp.train_idx], y[sp.train_idx], X_val, y_val,
                    n_classes=n_classes, embedding_dim=X.shape[1], max_epochs=max_epochs,
                    random_state=seed * 100 + fold_i, use_class_weights=opt,
                )
                ev = evaluate_model(model, X[sp.test_idx], y[sp.test_idx], class_names)
                proba = model.predict(X[sp.test_idx], verbose=0)
                y_hat = np.argmax(proba, axis=1)
                oof[bool(opt)][rep][0][sp.test_idx] = y_hat
                oof[bool(opt)][rep][1][sp.test_idx] = np.max(proba, axis=1)
                agirlikli_f1, makro_pr_auc = fold_extra_metrics(y[sp.test_idx], proba, n_classes)
                pooled[opt][0].extend(y[sp.test_idx].tolist())
                pooled[opt][1].extend(y_hat.tolist())
                fold_rows.append({
                    "tekrar": rep + 1, "kat": fold_i + 1, "sinif_agirligi": bool(opt),
                    "dogruluk": float(ev["accuracy"]), "makro_f1": float(ev["macro_f1"]),
                    "agirlikli_f1": agirlikli_f1, "makro_pr_auc": makro_pr_auc,
                    "n_train": len(sp.train_idx), "n_val": len(sp.val_idx), "n_test": len(sp.test_idx),
                    "n_test_kayit": len({kept_samples[i].recording_id for i in sp.test_idx}),
                })
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except Exception:  # noqa: BLE001
                    pass
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total)

    summary: dict = {}
    for opt in class_weight_options:
        rows = [r for r in fold_rows if r["sinif_agirligi"] == bool(opt)]
        summary[bool(opt)] = {
            m: (float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows], ddof=1)) if len(rows) > 1 else 0.0)
            for m in ("dogruluk", "makro_f1", "agirlikli_f1", "makro_pr_auc")
        }
    if len(class_weight_options) == 2 and set(map(bool, class_weight_options)) == {False, True}:
        for m in ("dogruluk", "makro_f1"):
            diffs = []
            for rep_kat in {(r["tekrar"], r["kat"]) for r in fold_rows}:
                a = next(r[m] for r in fold_rows if (r["tekrar"], r["kat"]) == rep_kat and not r["sinif_agirligi"])
                b = next(r[m] for r in fold_rows if (r["tekrar"], r["kat"]) == rep_kat and r["sinif_agirligi"])
                diffs.append(b - a)
            summary.setdefault("fark_agirlikli_eksi_agirliksiz", {})[m] = (
                float(np.mean(diffs)), float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0)

    per_class_rows: List[dict] = []
    from sklearn.metrics import confusion_matrix as _cm
    confusion = {bool(opt): _cm(yt, yp, labels=list(range(n_classes))) for opt, (yt, yp) in pooled.items()}
    for opt, (yt, yp) in pooled.items():
        p, r, f, sup = precision_recall_fscore_support(
            yt, yp, labels=list(range(n_classes)), zero_division=0)
        for i, name in enumerate(class_names):
            per_class_rows.append({"sinif_agirligi": bool(opt), "tur": name, "precision": float(p[i]),
                                   "recall": float(r[i]), "f1": float(f[i]), "destek_segment": int(sup[i])})

    return CrossValidationResult(
        fold_rows=fold_rows, per_class_rows=per_class_rows, summary=summary,
        class_names=class_names, n_samples=len(kept_samples),
        n_recordings=len({s.recording_id for s in kept_samples}), confusion=confusion, oof=oof,
        split_hash=_split_fingerprint(class_names, test_records),
    )
