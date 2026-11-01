"""
src/ml/train_classifier.py
================
Perch 2.0 embedding'leri üzerinde küçük bir MLP (çok katmanlı algılayıcı)
sınıflandırıcı eğitir — önceki SoftwareX çalışmanızdaki ("frozen Perch +
MLP") yaklaşımın aynısı, artık BioSeg içinde.

METODOLOJİK NOT — grup-farkında bölme (group-aware split): Standart
rastgele train/test bölme, AYNI orijinal kayıttan gelen segmentlerin
farklı setlere düşmesine izin verir — bu, modelin türü değil, o SPESİFİK
kaydın arka plan gürültüsünü/karakteristiğini öğrenip yapay olarak yüksek
doğruluk göstermesine yol açabilir (veri sızıntısı). Bu modül, her
segmentin `recording_id`'sini (orijinal kayıt kimliği) kullanarak
kayıt bazında KÜRESEL atama ile böler — aynı kayıttan gelen TÜM segmentler her
zaman AYNI sette (train, val ya da test) kalır (bkz. `group_aware_split`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .dataset_builder import TrainingSample

MIN_RECORDINGS_FOR_RELIABLE_SPLIT = 3  # tür başına en az bu kadar farklı KAYIT (segment değil) önerilir


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    species_missing_in_test: List[str] = field(default_factory=list)


def group_aware_split(
    samples: List[TrainingSample],
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
) -> SplitResult:
    """
    Örnekleri KAYIT (`recording_id`) bazında train/val/test'e böler: her kayıt
    KÜRESEL olarak TEK bir kümeye atanır ve o kaydın tüm segmentleri o kümede kalır.

    DÜZELTME (v2.0.1): Önceki sürüm kayıtları her tür için AYRI ayrı dağıtıyordu.
    Bir kayıt birden fazla türün segmentini içerdiğinde (gerçek veride kayıtların
    çoğu böyle), kayıt tür A için "train", tür B için "test" olarak işaretlenebiliyor
    ve segmentleri her iki kümeye de giriyordu — yani aynı örnekler hem eğitimde hem
    testte yer alıp veri sızıntısına ve şişirilmiş doğruluğa yol açıyordu. Artık:
      * her kayıt yalnızca bir kümeye atanır (küresel atama),
      * tür kotaları (test/val kayıt sayısı) hâlâ tür bazında hesaplanır; atama,
        nadir türlerden başlayarak açgözlü yapılır ve bir kaydı test/val'a ayırmak
        başka bir türün TÜM train kayıtlarını silecekse o kayıt ayrılmaz,
      * sonuçta kümelerin kayıt kümelerinin ayrık olduğu doğrulanır; ihlal
        olursa ValueError fırlatılır (sızıntı sessizce geçmez).

    Tür kotası: 1 kayıt -> hepsi train (test edilemez); 2 kayıt -> 1 train + 1 test;
    3+ kayıt -> round(n*test_size) test (en az 1), round(n*val_size) val
    (en az 1 train kalacak şekilde). Yeterli kayıt yoksa kota tam
    karşılanamayabilir; test kümesinde hiç görünmeyen türler
    `SplitResult.species_missing_in_test` içinde bildirilir.
    """
    rng = np.random.default_rng(random_state)

    species_to_recs: dict = {}
    rec_to_species: dict = {}
    for s in samples:
        species_to_recs.setdefault(s.species, set()).add(s.recording_id)
        rec_to_species.setdefault(s.recording_id, set()).add(s.species)

    assignment: dict = {}  # recording_id -> "val" | "test"; atanmamış = train

    def _can_hold_out(rec) -> bool:
        # Bu kaydı train'den çıkarmak, içerdiği hiçbir türün train'deki son kaydını silmemeli.
        for sp in rec_to_species[rec]:
            remaining = [r for r in species_to_recs[sp] if r != rec and r not in assignment]
            if not remaining:
                return False
        return True

    def _fill(species, recs_sorted, split_name, target):
        have = sum(1 for r in recs_sorted if assignment.get(r) == split_name)
        candidates = [r for r in recs_sorted if r not in assignment]
        rng.shuffle(candidates)
        for r in candidates:
            if have >= target:
                break
            if _can_hold_out(r):
                assignment[r] = split_name
                have += 1

    # Nadir türler önce (en az kaydı olan en kısıtlı olandır), eşitlikte ada göre.
    order = sorted(species_to_recs, key=lambda sp: (len(species_to_recs[sp]), str(sp)))
    for species in order:
        recs_sorted = sorted(species_to_recs[species])
        n = len(recs_sorted)
        if n == 1:
            continue
        if n == 2:
            _fill(species, recs_sorted, "test", 1)
            continue
        n_test = min(max(1, round(n * test_size)), n - 1)
        remaining_after_test = n - n_test
        n_val = min(max(0, round(n * val_size)), remaining_after_test - 1) if remaining_after_test >= 2 else 0
        _fill(species, recs_sorted, "test", n_test)
        _fill(species, recs_sorted, "val", n_val)

    train_idx = np.array([i for i, s in enumerate(samples) if s.recording_id not in assignment], dtype=int)
    val_idx = np.array([i for i, s in enumerate(samples) if assignment.get(s.recording_id) == "val"], dtype=int)
    test_idx = np.array([i for i, s in enumerate(samples) if assignment.get(s.recording_id) == "test"], dtype=int)

    # Güvence: kümelerin kayıt kümeleri ayrık olmalı (sızıntı yok).
    rec_sets = [{samples[i].recording_id for i in idx} for idx in (train_idx, val_idx, test_idx)]
    if (rec_sets[0] & rec_sets[1]) or (rec_sets[0] & rec_sets[2]) or (rec_sets[1] & rec_sets[2]):
        raise ValueError("Veri sızıntısı: aynı kayıt birden fazla kümede (train/val/test) göründü.")

    test_species = {samples[i].species for i in test_idx}
    missing = sorted(str(sp) for sp in species_to_recs if sp not in test_species)

    return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                       species_missing_in_test=missing)


def stratified_group_folds(y, groups, n_splits: int, seed: int) -> np.ndarray:
    """KAYIT bazlı (grup-farkında), tür-dengeli kat ataması: her örnek için kat numarası (0..n_splits-1).

    sklearn StratifiedGroupKFold ile aynı açgözlü ölçütü kullanır (her türün kat paylarının
    dağılımını eşitlemeye çalışır) ama sıralamaya tohuma bağlı rastgele bozulma ekler. Neden:
    sklearn'de karıştırma yalnızca EŞİT anahtarlı gruplar arasında etkilidir; kayıtlar çok türlü
    ve farklı bileşimli olduğunda farklı tohumlar BİREBİR AYNI bölmeyi üretir ve "tekrarlı"
    çapraz doğrulama aslında tekrarlanmaz. Burada her tohum farklı (ama yine dengeli) bir bölme verir.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    classes, y_idx = np.unique(y, return_inverse=True)
    uniq_groups, g_idx = np.unique(groups, return_inverse=True)
    n_g, n_c = len(uniq_groups), len(classes)
    if n_splits > n_g:
        raise ValueError(f"n_splits ({n_splits}) grup sayısından ({n_g}) büyük olamaz.")
    counts = np.zeros((n_g, n_c))
    np.add.at(counts, (g_idx, y_idx), 1)
    total = counts.sum(axis=0)
    rng = np.random.default_rng(seed)
    key = -(counts.std(axis=1) * (0.6 + 0.8 * rng.random(n_g)))      # yoğun bileşimli kayıtlar önce, tohuma göre bozulmuş
    order = np.argsort(key, kind="stable")
    fold_counts = np.zeros((n_splits, n_c))
    fold_size = np.zeros(n_splits)
    assign = np.empty(n_g, dtype=int)
    for g in order:
        best, best_cost, best_size = -1, np.inf, np.inf
        for f in rng.permutation(n_splits):
            fold_counts[f] += counts[g]
            cost = float(np.mean(np.std(fold_counts / total, axis=0)))
            fold_counts[f] -= counts[g]
            if cost < best_cost - 1e-12 or (abs(cost - best_cost) <= 1e-12 and fold_size[f] < best_size):
                best, best_cost, best_size = int(f), cost, fold_size[f]
        assign[g] = best
        fold_counts[best] += counts[g]
        fold_size[best] += counts[g].sum()
    for f in range(n_splits):                       # güvence: boş kat kalmasın
        if not (assign == f).any():
            donor = int(np.bincount(assign, minlength=n_splits).argmax())
            cand = np.flatnonzero(assign == donor)
            assign[cand[np.argmin(counts[cand].sum(axis=1))]] = f
    return assign[g_idx]


def group_kfold_splits(
    samples: List[TrainingSample],
    n_splits: int = 5,
    random_state: int = 42,
    val_fraction_of_train: float = 0.2,
) -> List[SplitResult]:
    """
    KAYIT bazlı (grup-farkında), tür-dengeli k-katlı bölmeler üretir (bkz. stratified_group_folds;
    farklı `random_state` değerleri GERÇEKTEN farklı bölmeler verir). Her kat için: test = o katın
    kayıtları; kalan kayıtlardan yine kayıt bazlı bir doğrulama kümesi ayrılır (erken durdurma için).
    Her kayıt her katta TEK bir kümededir; ihlal olursa ValueError.
    """
    y = np.array([s.species for s in samples])
    groups = np.array([s.recording_id for s in samples])
    n_groups = len(set(groups))
    if n_splits < 2 or n_splits > n_groups:
        raise ValueError(f"n_splits ({n_splits}) 2 ile kayıt sayısı ({n_groups}) arasında olmalı.")

    fold_of = stratified_group_folds(y, groups, n_splits, random_state)
    splits: List[SplitResult] = []
    for fold_i in range(n_splits):
        test_idx = np.flatnonzero(fold_of == fold_i)
        train_all = np.flatnonzero(fold_of != fold_i)
        n_inner = max(2, int(round(1.0 / val_fraction_of_train)))
        train_idx = train_all
        val_idx = np.array([], dtype=int)
        if len(set(groups[train_all])) >= n_inner:
            inner = stratified_group_folds(y[train_all], groups[train_all], n_inner, random_state + 1000 + fold_i)
            train_idx, val_idx = train_all[inner != 0], train_all[inner == 0]

        rec_sets = [set(groups[i]) for i in (train_idx, val_idx, test_idx)]
        if (rec_sets[0] & rec_sets[1]) or (rec_sets[0] & rec_sets[2]) or (rec_sets[1] & rec_sets[2]):
            raise ValueError("Veri sızıntısı: aynı kayıt birden fazla kümede göründü (k-katlı bölme).")
        test_species = {samples[i].species for i in test_idx}
        missing = sorted({str(sp) for sp in y} - {str(sp) for sp in test_species})
        splits.append(SplitResult(
            train_idx=np.asarray(train_idx, dtype=int), val_idx=np.asarray(val_idx, dtype=int),
            test_idx=np.asarray(test_idx, dtype=int), species_missing_in_test=missing,
        ))
    return splits


def build_and_train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    n_classes: int,
    embedding_dim: int = 1536,
    max_epochs: int = 100,
    random_state: int = 42,
    progress_callback: Optional[Callable[[int, int, dict], None]] = None,
    use_class_weights: bool = False,
):
    """
    Embedding -> tür sınıflandırması yapan küçük bir MLP eğitir
    (Dense + BatchNorm + Dropout katmanları, erken durdurma ile aşırı
    uyumu önler). Mimari, önceki çalışmanızdaki derin MLP'nin sadeleştirilmiş
    bir versiyonu — küçük veri setleri için daha az parametre, daha az
    aşırı uyum riski.

    ÖNEMLİ — tekrarlanabilirlik: `random_state`, sadece veri bölmesini
    değil, TensorFlow/Keras'ın kendi ağırlık başlangıç değerlerini ve
    eğitim sırasındaki rastgeleliğini (dropout, batch karıştırma) de
    sabitler. Bunsuz, AYNI veriyle bile her çalıştırmada FARKLI bir model
    ortaya çıkabilir — özellikle az örnekli sınıflarda bu fark çok büyük
    olabilir (gerçek bir kullanımda gözlemlendi: aynı veri bölmesiyle bir
    çalıştırmada bir türün F1'i 0,87 iken, seed'siz başka bir çalıştırmada
    aynı tür 0,0 F1 verdi). CPU'da bu, sonuçları tam olarak
    tekrarlanabilir kılar (GPU'da bazı işlemler doğası gereği hâlâ küçük
    farklılıklar gösterebilir).
    """
    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(random_state)

    hidden1 = min(512, max(64, embedding_dim // 3))
    hidden2 = min(256, max(32, hidden1 // 2))

    model = keras.Sequential([
        keras.layers.Input(shape=(embedding_dim,)),
        keras.layers.Dense(hidden1, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(hidden2, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    ]

    if progress_callback is not None:
        class _ProgressCallback(keras.callbacks.Callback):
            def on_epoch_end(self_inner, epoch, logs=None):
                progress_callback(epoch + 1, max_epochs, logs or {})

        callbacks.append(_ProgressCallback())

    fit_kwargs = dict(
        epochs=max_epochs,
        batch_size=min(16, max(4, len(X_train) // 8)),
        callbacks=callbacks,
        verbose=0,
    )

    if use_class_weights:
        # "Dengeli" sınıf ağırlıkları: w_c = N / (K * n_c). Nadir türlerin kaybı
        # daha ağır basar (makro F1'i hedefleyen bir ayar; genel doğruluğu düşürebilir).
        classes, counts = np.unique(y_train, return_counts=True)
        fit_kwargs["class_weight"] = {
            int(c): float(len(y_train) / (len(classes) * n)) for c, n in zip(classes, counts)
        }

    if X_val is not None and len(X_val) > 0:
        fit_kwargs["validation_data"] = (X_val, y_val)
    else:
        # Grup-farkında bir doğrulama seti için yeterli farklı kayıt
        # yoktu (bkz. group_aware_split). Erken durdurmanın izleyecek bir
        # sinyali olması için, eğitim verisinin kendi içinden Keras'ın
        # KENDİ (grup-farkında OLMAYAN) rastgele bir dilimini kullanıyoruz
        # — bu SADECE erken durdurma için bir uzlaşıdır; nihai test
        # değerlendirmesi (bkz. evaluate_model) buna dahil DEĞİLDİR ve
        # hâlâ tamamen grup-izole kalır, bu yüzden asıl raporlanan
        # sonuçlarda veri sızıntısı riski yoktur.
        fit_kwargs["validation_split"] = 0.15
        for cb in callbacks:
            if isinstance(cb, keras.callbacks.EarlyStopping):
                cb.monitor = "val_loss"  # validation_split ile de val_loss üretilir

    history = model.fit(X_train, y_train, **fit_kwargs)
    return model, history.history


@dataclass
class TrainingResult:
    test_accuracy: float
    test_macro_f1: float
    per_class_report: Dict[str, dict]
    confusion_matrix: np.ndarray
    class_names: List[str]
    n_train: int
    n_val: int
    n_test: int
    n_train_recordings: int
    n_val_recordings: int
    n_test_recordings: int
    excluded_species: List[str] = field(default_factory=list)
    history: Optional[dict] = None
    test_predictions: Optional[List[dict]] = None   # segment başına gerçek/tahmin/güven (recording_level için)
    cv_summary: Optional[dict] = None               # pipeline.summarize_cv çıktısı (kayıt-gruplu çapraz doğrulama)
    learning_curve: Optional[dict] = None           # learning_curve.run_learning_curve çıktısı ({'ozet','satirlar',...}) ya da {'hata'}
    zenginlik_uyumu: Optional[dict] = None          # richness_validation.agreement_from_oof çıktısı ya da {'hata'}

    def format_summary(self) -> str:
        lines = []
        cv = self.cv_summary
        if cv and "hata" in cv:
            lines.append(f"⚠ Çapraz doğrulama çalıştırılamadı: {cv['hata']}")
        elif cv:
            (a, asd), (f, fsd) = cv["dogruluk"], cv["makro_f1"]
            lines += [
                f"ÇAPRAZ DOĞRULAMA ({cv['n_kat']} kat × {cv['n_tekrar']} tekrar, kayıt bazlı) — ana sonuç:",
                f"  Doğruluk: %{a * 100:.1f} ± {asd * 100:.1f} (kat aralığı %{cv['dogruluk_aralik'][0] * 100:.1f}–"
                f"{cv['dogruluk_aralik'][1] * 100:.1f})",
                f"  Makro F1: {f:.3f} ± {fsd:.3f} (kat aralığı {cv['makro_f1_aralik'][0]:.3f}–{cv['makro_f1_aralik'][1]:.3f})",
            ]
            if "agirlikli_f1" in cv:
                lines.append(f"  Ağırlıklı F1: {cv['agirlikli_f1'][0]:.3f} ± {cv['agirlikli_f1'][1]:.3f}")
            if "makro_pr_auc" in cv:
                lines.append(f"  Makro PR-AUC: {cv['makro_pr_auc'][0]:.3f} ± {cv['makro_pr_auc'][1]:.3f}")
            if "havuzlanmis" in cv:
                hv = cv["havuzlanmis"]
                lines.append(f"  Havuzlanmış (tüm testlerin tahminleri birleştirilmiş): doğruluk %{hv['dogruluk'] * 100:.1f}, "
                             f"makro F1 {hv['makro_f1']:.3f}, ağırlıklı F1 {hv['agirlikli_f1']:.3f}")
            lines.append("Örnek bölme (tek test kümesi; kararsız olabilir, ana sonuç yukarıdaki çapraz doğrulamadır):")
        lines += [
            f"Test doğruluğu (accuracy): %{self.test_accuracy * 100:.1f}",
            f"Test makro F1: {self.test_macro_f1:.3f}",
            f"Eğitim: {self.n_train} segment ({self.n_train_recordings} farklı kayıt), "
            f"Doğrulama: {self.n_val} segment ({self.n_val_recordings} kayıt), "
            f"Test: {self.n_test} segment ({self.n_test_recordings} kayıt)",
            f"Sınıflandırılan tür sayısı: {len(self.class_names)}",
        ]
        if self.learning_curve:
            from .learning_curve import format_text
            lines.append(format_text(self.learning_curve))
        if self.zenginlik_uyumu:
            from .richness_validation import format_text as _rt
            lines.append(_rt(self.zenginlik_uyumu))
        if self.test_predictions:
            from .recording_level import dominance_warning, format_presence_summary
            lines.append(format_presence_summary(self.test_predictions))
            warn = dominance_warning(self.test_predictions)
            if warn:
                lines.append(warn)
        if self.n_val == 0:
            lines.append(
                "⚠ Ayrı bir kayıt-bazlı doğrulama seti için yeterli farklı kayıt yoktu; "
                "erken durdurma, eğitim verisinin kendi içinden rastgele küçük bir dilim "
                "kullanılarak izlendi. Bu, sadece erken durdurma sinyalini etkiler — "
                "yukarıdaki TEST sonuçları hâlâ tamamen kayıt-izole ve güvenilirdir. Daha "
                "fazla farklı kayıt topladıkça bu uyarı kalkacaktır."
            )
        if self.excluded_species:
            lines.append(
                f"⚠ {len(self.excluded_species)} tür, yetersiz örnek sayısı nedeniyle "
                f"eğitime dahil edilmedi: {', '.join(self.excluded_species[:5])}"
                + (" ..." if len(self.excluded_species) > 5 else "")
            )
        return "\n".join(lines)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from .olcutler import makro_f1

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    report = classification_report(
        y_test, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": makro_f1(y_test, y_pred),
        "report": report,
        "confusion_matrix": cm,
    }


def save_model(model, class_names: List[str], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, "model.keras"))
    with open(os.path.join(save_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(list(class_names), f, ensure_ascii=False, indent=2)


def load_model(save_dir: str):
    """Kaydedilmiş modeli ve tür isim listesini geri yükler."""
    from tensorflow import keras
    model = keras.models.load_model(os.path.join(save_dir, "model.keras"))
    with open(os.path.join(save_dir, "classes.json"), encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names