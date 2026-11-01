"""
src/ml/birdnet_zero_shot.py
===========================
BirdNET 2.4'ün hazır (yerel veriyle eğitilmemiş) tür tahminlerini temel çizgi olarak kullanır.

  kisitli : tahmin, yalnızca veri setindeki sınıflara karşılık gelen BirdNET türleri arasından en yüksek olasılıklı tür
  acik    : tahmin, BirdNET'in tüm türleri arasından en yüksek olasılıklı tür (veri setinde olmayan tür = yanlış)

Eşleştirme BirdNET etiketinin bilimsel ad kısmı ("Parus major_Great Tit" -> "parus major") üzerinden yapılır.
Karşılığı bulunamayan sınıf hiçbir zaman doğru tahmin edilemez ve kapsam raporunda görünür.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embedding_comparison import BIRDNET_SR, BIRDNET_WINDOW_S


def label_scientific(label: str) -> str:
    return str(label).split("_")[0].strip().lower()


def match_classes(class_names: Sequence[str], model_species: Sequence[str]) -> Dict[str, Optional[str]]:
    """Sınıf adı -> BirdNET etiketi (bulunamazsa None)."""
    index: Dict[str, str] = {}
    for lab in model_species:
        index.setdefault(label_scientific(lab), lab)
    return {c: index.get(str(c).strip().lower()) for c in class_names}


def _prepare(y: np.ndarray) -> np.ndarray:
    n = int(BIRDNET_SR * BIRDNET_WINDOW_S)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)


def _collect(res, i: int):
    """i. girdinin (ad, olasılık) listesi: tüm pencereler, maskelenmemiş girdiler."""
    names = np.asarray(res.species_list)
    ids = np.asarray(res.species_ids)[i]
    probs = np.asarray(res.species_probs)[i]
    masked = np.asarray(res.species_masked)[i]
    out = []
    for w in range(ids.shape[0]):
        for k in range(ids.shape[1]):
            if not masked[w, k]:
                out.append((str(names[int(ids[w, k])]), float(probs[w, k])))
    return out


def zero_shot_predictions(kept_samples, class_names: Sequence[str], model=None, library: str = "litert",
                          chunk_size: int = 32,
                          progress_callback: Optional[Callable[[int, int], None]] = None
                          ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Optional[str]]]:
    """Döndürür: (kisitli_tahmin (N,), acik_tahmin (N,), eşleştirme). Tahminler sınıf indeksi; -1 = karşılığı yok.
    Ses 48 kHz'e yeniden örneklenir, 3 sn'den kısa segmentler sıfırla tamamlanır."""
    import librosa

    if model is None:
        from .birdnet_embedding import get_birdnet_model
        model = get_birdnet_model(library)
    match = match_classes(class_names, list(model.species_list))
    class_of_label = {lab: j for j, c in enumerate(class_names) for lab in [match[c]] if lab}
    restricted_list = [lab for lab in match.values() if lab]
    n = len(kept_samples)
    kis = np.full(n, -1, dtype=int)
    aci = np.full(n, -1, dtype=int)
    for start in range(0, n, max(1, int(chunk_size))):
        part = kept_samples[start:start + chunk_size]
        batch = []
        for s in part:
            y, _ = librosa.load(s.filepath, sr=BIRDNET_SR, mono=True)
            if len(y) == 0:
                raise ValueError(f"Ses dosyası boş: {s.filepath}")
            batch.append((_prepare(y), BIRDNET_SR))
        if restricted_list:
            res_k = model.predict_arrays(batch, top_k=len(restricted_list), default_confidence_threshold=0.0,
                                         custom_species_list=restricted_list)
        res_a = model.predict_arrays(batch, top_k=1, default_confidence_threshold=0.0)
        for i in range(len(part)):
            if restricted_list:
                cand = [(p, class_of_label[nm]) for nm, p in _collect(res_k, i) if nm in class_of_label]
                if cand:
                    kis[start + i] = max(cand, key=lambda t: (t[0], -t[1]))[1]      # eşitlikte düşük indeks (argmax gibi)
            top = _collect(res_a, i)
            if top:
                nm = max(top, key=lambda t: t[1])[0]
                aci[start + i] = class_of_label.get(nm, -1)
        if progress_callback:
            progress_callback(min(start + chunk_size, n), n)
    return kis, aci, match


def fold_rows(pred: np.ndarray, kept_samples: list, name: str, n_splits: int = 5, n_repeats: int = 3,
              base_seed: int = 42) -> List[dict]:
    """Eğitimsiz tahminleri, run_embedding_comparison ile AYNI kayıt-gruplu bölmelerde değerlendirir."""
    from sklearn.preprocessing import LabelEncoder
    from .baselines import _metrics
    from .train_classifier import group_kfold_splits

    y = LabelEncoder().fit_transform([s.species for s in kept_samples])
    n_classes = int(y.max()) + 1
    rows = []
    for rep in range(n_repeats):
        for fold_i, sp in enumerate(group_kfold_splits(kept_samples, n_splits=n_splits, random_state=base_seed + rep)):
            rows.append({"tekrar": rep + 1, "kat": fold_i + 1, "yontem": name, "sinif_agirligi": False,
                         **_metrics(y[sp.test_idx], pred[sp.test_idx], n_classes),
                         "n_train": len(sp.train_idx), "n_test": len(sp.test_idx)})
    return rows
