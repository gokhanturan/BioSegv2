"""
src/ml/ai_richness.py
=====================
"Yapay zekâ ile tür zenginliği" sekmesinin verisi: kayıtlı model sonuçlarındaki `zenginlik_uyumu`
(richness_validation.agreement_from_oof çıktısı) modeller yan yana, eşik eşik tablolanır.

Ne ölçülür (açıkça): girdi uzmanın ayırdığı SEGMENTLERDİR; her segmentin sınıflandırması, o segmentin kaydını
eğitimde görmemiş bir modelden gelir (kayıt bazlı çapraz doğrulama). Ölçülen, sınıflandırma hatalarının
nokta düzeyindeki tür zenginliğine yansımasıdır; tespit/segmentasyon hatası ve habitat karşılaştırması
kapsam dışıdır. Evren, modelin tanıdığı türlerdir (tüm belirlenen türlerin alt kümesi).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

COLUMNS = ("Model", "Güven eşiği", "Tespit oranı", "Tür (uzman)", "Tür (model)", "Nokta başı |fark|",
           "Spearman", "Kesinlik", "Duyarlılık", "Jaccard", "Yanlış pozitif", "Kaçırılan")
COLUMN_HELP = (
    "Model.",
    "Pencere/segment, modelin en yüksek olasılığı bu değerin altındaysa 'tür yok' sayılır (0 = hepsi sayılır).",
    "Eşiği geçen segmentlerin oranı.",
    "Tüm noktalarda uzman etiketlerinden belirlenen farklı tür sayısı (yalnızca modelin tanıdığı türler).",
    "Aynı noktalarda modelin belirlediği farklı tür sayısı.",
    "Nokta başına |uzman − model| tür sayısı farkının ortalaması (0 = her noktada aynı sayı).",
    "Nokta başına tür sayısının uzman ve model arasındaki sıra korelasyonu (noktaları aynı sıraya dizip dizmediği).",
    "Modelin 'var' dediği (nokta, tür) çiftlerinin uzmanca doğru olan oranı.",
    "Uzmanın 'var' dediği (nokta, tür) çiftlerinin model tarafından bulunan oranı.",
    "(nokta, tür) çiftlerinde uzman ve modelin kesişim/birleşim oranı: tür bazında uyum.",
    "Modelin eklediği, uzmanda olmayan (nokta, tür) çifti sayısı (ortalama).",
    "Uzmanın bildirdiği ama modelin bulamadığı (nokta, tür) çifti sayısı (ortalama).",
)

SCOPE_NOTE = (
    "<b>Kapsam:</b> Girdi, uzmanın ayırdığı segmentlerdir; model her segmentin türünü, o kaydı eğitimde görmemiş bir "
    "çapraz doğrulama modeliyle tahmin eder. Ölçülen, <b>sınıflandırma hatalarının nokta düzeyindeki tür zenginliğine "
    "yansımasıdır</b>. Ham kayıt taraması (tespit/segmentasyon hatası) ve habitat karşılaştırması bu tabloda yoktur. "
    "Evren, modelin tanıdığı türlerdir; diğer türler hesaba girmez. Segmentler ve etiketler BirdNET tespit ve önerisine "
    "dayandığından BirdNET lehine yanlılık olabilir. ± değerleri tekrarlar arası standart sapmadır."
)


def _f(v, nd=2) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if x != x:      # NaN
        return "—"
    return f"{x:.{nd}f}".replace(".", ",")


def _pm(o: dict, key: str, nd=2) -> str:
    """'ortalama ± standart sapma'. Sapma, en az 2 tekrar varsa her zaman yazılır (0,00 dahil); tek tekrarda sapma
    tanımsızdır ve yazılmaz."""
    m, s = o.get(key + "_ort"), o.get(key + "_ss")
    if m is None:
        return "—"
    out = _f(m, nd)
    n_rep = o.get("n_tekrar")
    if out != "—" and s is not None and (n_rep is None or n_rep >= 2) and _f(s, nd) != "—":
        out += " ± " + _f(s, nd)
    return out


def build_rows(stored_list: Sequence[Tuple[object, object]]) -> dict:
    """[(ModelSpec, StoredResult)] -> {'satirlar': [...], 'uyarilar': [...], 'modeller': [...]}.

    Her satır bir (model, güven eşiği). Zenginlik uyumu kayıtlı olmayan modeller uyarıya yazılır."""
    rows: List[dict] = []
    warns: List[str] = []
    models: List[dict] = []
    for spec, st in stored_list:
        tr = st.result.training_result
        zu = getattr(tr, "zenginlik_uyumu", None)
        if not zu:
            warns.append(f"{spec.ad}: tür zenginliği uyumu kayıtlı değil; modelin eğitim penceresinde eğitimi yeniden "
                         "çalıştırın.")
            continue
        if "hata" in zu:
            warns.append(f"{spec.ad}: tür zenginliği uyumu hesaplanamadı ({zu['hata']}).")
            continue
        models.append({"id": spec.id, "ad": spec.ad, "n_sinif": len(getattr(tr, "class_names", []) or []),
                       "n_nokta": zu.get("n_nokta"), "n_tekrar": zu.get("n_tekrar"),
                       "nokta": zu.get("nokta") or []})
        if zu.get("n_noktasiz_kayit"):
            warns.append(f"{spec.ad}: {zu['n_noktasiz_kayit']} kaydın Habitat_ID değeri yok; karşılaştırmaya girmedi.")
        for o in zu.get("esikler", []):
            rows.append({
                "model_id": spec.id, "model": spec.ad, "esik": float(o["guven_esigi"]),
                "hucreler": (spec.ad, _f(o["guven_esigi"], 1), _f(o.get("tespit_orani_ort")),
                             _f(o.get("tur_uzman_ort"), 1), _pm(o, "tur_model", 1),
                             _pm(o, "nokta_basina_tur_mutlak_fark_ort", 2),
                             _pm(o, "nokta_basina_tur_spearman"), _pm(o, "var_yok_kesinlik"),
                             _pm(o, "var_yok_duyarlilik"), _pm(o, "var_yok_jaccard"),
                             _pm(o, "sahte_tur_nokta_cifti", 1), _pm(o, "kacirilan_tur_nokta_cifti", 1)),
                "jaccard": o.get("var_yok_jaccard_ort"),
            })
    return {"satirlar": rows, "uyarilar": warns, "modeller": models}


def to_csv(result: dict) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(COLUMNS)
    for r in result["satirlar"]:
        w.writerow(r["hucreler"])
    return buf.getvalue()


def coverage_text(models: Sequence[dict], n_species_total: int) -> str:
    """Modelin tanıdığı tür sayısının, veri setinde belirlenen tüm türlere oranı."""
    if not models or not n_species_total:
        return ""
    parts = []
    for m in models:
        if m["n_sinif"]:
            parts.append(f"{m['ad']}: {m['n_sinif']}/{n_species_total} tür (%{100 * m['n_sinif'] / n_species_total:.0f})")
    return "Modelin tanıdığı tür sayısı — " + "; ".join(parts) + "." if parts else ""
