"""Sınıflandırma ölçütleri: tüm modüller makro F1 ve ek ölçütler için bu tanımları kullanır."""
import numpy as np


def makro_f1(y_true, y_pred) -> float:
    """Makro F1: yalnızca y_true içinde örneği bulunan türler üzerinden ortalama."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true)
    if labels.size == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


def agirlikli_f1(y_true, y_pred) -> float:
    """Ağırlıklı F1: tür başına F1, y_true içindeki segment sayısıyla ağırlıklandırılır."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, average="weighted", labels=np.unique(y_true), zero_division=0))


def makro_pr_auc(y_true, proba) -> float:
    """Bir-karşı-hepsi ortalama kesinlik (PR-AUC); yalnızca y_true içinde örneği olan türler ortalanır."""
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    aps = [float(average_precision_score(y_true == c, proba[:, c])) for c in np.unique(y_true)]
    return float(np.mean(aps)) if aps else float("nan")


def havuzlanmis(confusion) -> dict:
    """Birleştirilmiş karışıklık matrisinden doğruluk, makro F1 ve ağırlıklı F1."""
    cm = np.asarray(confusion, dtype=float)
    tp = np.diag(cm)
    destek = cm.sum(axis=1)
    payda = destek + cm.sum(axis=0)
    f1 = np.divide(2 * tp, payda, out=np.zeros_like(tp), where=payda > 0)
    var = destek > 0
    toplam = cm.sum()
    return {
        "dogruluk": float(tp.sum() / toplam) if toplam else 0.0,
        "makro_f1": float(f1[var].mean()) if var.any() else 0.0,
        "agirlikli_f1": float((f1 * destek).sum() / destek.sum()) if destek.sum() else 0.0,
    }
