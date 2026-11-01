"""
src/ml/segment_audit.py
=======================
Segment WAV dosyalarının teknik özelliklerini (örnekleme hızı, kanal, süre, ses seviyesi, spektral üst sınır)
dosya başına ve tür başına özetler. Hiçbir ayarı değiştirmez; yalnızca ölçer.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

SPEKTRAL_YUZDELER = (0.90, 0.99)


def _dbfs(x: float) -> Optional[float]:
    return 20.0 * math.log10(x) if x > 0 else None


def audit_file(path: str) -> dict:
    """Tek bir WAV için ölçümler. Okunamazsa {'dosya': path, 'hata': ...} döner."""
    import soundfile as sf

    row: dict = {"dosya": path}
    try:
        info = sf.info(path)
        y, sr = sf.read(path, always_2d=True, dtype="float64")
    except Exception as exc:  # noqa: BLE001
        row["hata"] = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        return row
    mono = y.mean(axis=1)
    n = len(mono)
    row.update({
        "ornekleme_hizi_hz": int(sr), "kanal": int(info.channels), "bit_derinligi": info.subtype,
        "sure_sn": round(n / sr, 4) if sr else None, "ornek_sayisi": int(n),
    })
    if n == 0:
        row["hata"] = "boş dosya"
        return row
    peak = float(np.abs(mono).max())
    rms = float(np.sqrt(np.mean(mono ** 2)))
    row["tepe_dbfs"] = _round(_dbfs(peak))
    row["rms_dbfs"] = _round(_dbfs(rms))
    row["kirpilma_orani"] = float(np.mean(np.abs(y) >= 0.9999))
    if rms > 0:
        spec = np.abs(np.fft.rfft(mono * np.hanning(n))) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        cum = np.cumsum(spec) / spec.sum()
        for p in SPEKTRAL_YUZDELER:
            row[f"enerji_%{int(p * 100)}_hz"] = round(float(freqs[min(np.searchsorted(cum, p), len(freqs) - 1)]), 1)
    row["nyquist_hz"] = sr / 2
    return row


def _round(v):
    return None if v is None else round(v, 2)


def _q(values: List[float], q: float):
    return None if not values else float(np.percentile(values, q))


def summarize(rows: List[dict], species_of: Dict[str, str]) -> List[dict]:
    """Tür başına (ve 'TÜMÜ') özet satırları: sayılar, hız/süre dağılımı, seviye ve spektral çeyrekler."""
    groups: Dict[str, List[dict]] = {"TÜMÜ": [r for r in rows if "hata" not in r]}
    for r in rows:
        if "hata" in r:
            continue
        groups.setdefault(species_of.get(r["dosya"], "?"), []).append(r)
    errors = sum(1 for r in rows if "hata" in r)

    out = []
    for name, rs in groups.items():
        def col(k):
            return [r[k] for r in rs if r.get(k) is not None]
        rates = sorted({r["ornekleme_hizi_hz"] for r in rs})
        out.append({
            "tur": name, "n_dosya": len(rs),
            "okunamayan_dosya": errors if name == "TÜMÜ" else 0,
            "ornekleme_hizlari_hz": "/".join(str(x) for x in rates),
            "kanal_ayri_degerler": "/".join(str(x) for x in sorted({r["kanal"] for r in rs})),
            "sure_min_sn": _round(min(col("sure_sn"))) if col("sure_sn") else None,
            "sure_medyan_sn": _round(_q(col("sure_sn"), 50)),
            "sure_maks_sn": _round(max(col("sure_sn"))) if col("sure_sn") else None,
            "rms_dbfs_min": _round(min(col("rms_dbfs"))) if col("rms_dbfs") else None,
            "rms_dbfs_q25": _round(_q(col("rms_dbfs"), 25)),
            "rms_dbfs_medyan": _round(_q(col("rms_dbfs"), 50)),
            "rms_dbfs_q75": _round(_q(col("rms_dbfs"), 75)),
            "rms_dbfs_maks": _round(max(col("rms_dbfs"))) if col("rms_dbfs") else None,
            "tepe_dbfs_medyan": _round(_q(col("tepe_dbfs"), 50)),
            "kirpilan_dosya": sum(1 for r in rs if r.get("kirpilma_orani", 0) > 0),
            "enerji_%99_hz_medyan": _round(_q(col("enerji_%99_hz"), 50)),
            "enerji_%99_hz_q95": _round(_q(col("enerji_%99_hz"), 95)),
        })
    return out
