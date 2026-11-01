"""
Başarı sınırı analizi: "Sınıflandırmanın sınırı model mi, veri mi?" sorusunu
veri setinden ve (varsa) çapraz doğrulama çıktılarından, veri setine özgü
sayı/tür adı gömmeden yanıtlar.

Kanıt türleri (hepsi çıktıdaki metinde açıkça belirtilir):
  1. Veri yeterliliği: tür başına segment ve FARKLI kayıt sayısı; eğitim
     süzgecinin (>= min_segment, >= min_kayit) dışında kalan türler.
  2. Yöntem platosu: BirdNET tabanlı yöntemlerin doğruluğu birbirine yakınsa
     model karmaşıklığı artışı kazanç getirmiyor demektir.
  3. Destek–başarı ilişkisi: tür başına gerçek segment sayısı ile F1 arasındaki
     Spearman ilişkisi. Güçlü pozitif ilişki, zayıf türlerin "az örnek" yüzünden
     zayıf olduğunu düşündürür.

Uyarı: Bu kanıtlar nedensellik KANITLAMAZ (zor türler aynı zamanda nadir
türler olabilir). Kesin sınama öğrenme eğrisidir (eğitim verisini artırarak
başarının nasıl değiştiğine bakmak); metin bunu belirtir.
"""
import csv
import json
import os
import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .dataset_builder import DEFAULT_MIN_SEGMENTS

# Eşikler (a priori, buluşsal): çıktıda "buluşsal" olarak etiketlenir.
PLATEAU_ACC_SPREAD = 0.03     # BirdNET tabanlı yöntemlerin doğruluk aralığı bu değerin altındaysa plato
STRONG_RHO = 0.5              # destek–F1 Spearman'ı bu değerin üstündeyse "güçlü"
WEAK_F1 = 0.7                 # bu değerin altındaki F1 "zayıf" sayılır

EMBEDDING_METHODS = ("MLP_BirdNET", "LR_BirdNET", "SifirAtis_Perch")    # eski (CSV tabanlı) çalıştırmalar
MODEL_IDS = ("birdnet", "perch", "birlesik_birdnet_perch")            # projenin üç modeli (model_registry kimlikleri)
METHOD_NAMES = {
    "birdnet": "BirdNET 2.4",
    "perch": "Perch 2.0",
    "birlesik_birdnet_perch": "BirdNET + Perch (birleşik)",
    "MLP_BirdNET": "BirdNET + MLP",
    "LR_BirdNET": "BirdNET + Lojistik regresyon",
    "SifirAtis_Perch": "Perch hazır çıktı (sıfır-atış)",
    "LR_MFCC": "MFCC + Lojistik regresyon",
    "LR_SifirAtis_Perch": "Perch + sıfır-atış skorları (LR)",
    "Prototip_BirdNET": "BirdNET + prototip (kosinüs)",
    "Topluluk_BirdNET": "BirdNET topluluğu (LR+prototip)",
    "MLP_Topluluk": "BirdNET + MLP topluluğu",
}
ORDER = ("birlesik_birdnet_perch", "perch", "birdnet", "SifirAtis_Perch", "LR_MFCC", "Prototip_BirdNET", "LR_BirdNET", "LR_SifirAtis_Perch",
         "Topluluk_BirdNET", "MLP_BirdNET", "MLP_Topluluk")

SUMMARY_FILE = "cv_yontem_ozeti.csv"
PER_CLASS_FILE = "cv_tur_bazli.csv"
PAIRED_FILE = "cv_eslestirilmis_fark.csv"
SELECTIVE_FILE = "cv_secici_tahmin.csv"
SINGLE_SPLIT_FILE = "son_egitim_sonucu.json"
FINGERPRINT_FILE = "cv_veri_izi.json"
_FP_COLS = ("Original_Recording_Name", "Start_Time", "End_Time", "Scientific_Name")


def _fmt_time(ts: Optional[float]) -> str:
    return _dt.datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M") if ts else "?"


def file_times(folder: str) -> Dict[str, float]:
    """Sonuç dosyalarının son değişme zamanları (yoksa anahtar yok)."""
    out = {}
    for n in (SUMMARY_FILE, PER_CLASS_FILE, PAIRED_FILE, SELECTIVE_FILE, SINGLE_SPLIT_FILE):
        p = os.path.join(folder, n)
        if os.path.exists(p):
            out[n] = os.path.getmtime(p)
    return out


def data_fingerprint(csv_path: str) -> Optional[dict]:
    """
    Veri setinin "izi": eğitim örneklerini belirleyen sütunların (kayıt, başlangıç, bitiş, tür) içerik özeti.
    Dosya zamanına bağlı değildir: dosya kopyalanır/geri yüklenirse değişmez, bir etiket eklenir/silinir/
    değişirse değişir. Dosya yoksa/boşsa None.
    """
    import hashlib
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return None
    if df.empty:
        return None
    cols = [c for c in _FP_COLS if c in df.columns]
    if not cols:
        return None
    key = sorted("\x1f".join(str(v).strip() for v in row) for row in df[cols].itertuples(index=False, name=None))
    h = hashlib.sha256("\n".join(key).encode("utf-8")).hexdigest()[:16]
    sp = df["Scientific_Name"].str.strip() if "Scientific_Name" in df.columns else pd.Series(dtype=str)
    rc = df["Original_Recording_Name"].str.strip() if "Original_Recording_Name" in df.columns else pd.Series(dtype=str)
    return {"n_segment": int(len(df)), "n_tur": int(sp[sp != ""].nunique()),
            "n_kayit": int(rc[rc != ""].nunique()), "hash": h}


def write_fingerprint(folder: str, fp: Optional[dict]) -> None:
    if not fp:
        return
    data = dict(fp, zaman=_dt.datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(folder, FINGERPRINT_FILE), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def clear_fingerprint(folder: str) -> None:
    try:
        os.remove(os.path.join(folder, FINGERPRINT_FILE))
    except OSError:
        pass


def load_fingerprint(folder: str) -> Optional[dict]:
    p = os.path.join(folder, FINGERPRINT_FILE)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_single_split(folder: str, tr) -> str:
    """'Tür Sınıflandırma Eğitimi' (tek bölme) sonucunu JSON olarak yazar; analiz penceresi ayrı bölümde gösterir."""
    rep = tr.per_class_report or {}
    per = {c: {"f1": float(rep[c]["f1-score"]), "destek": int(rep[c].get("support", 0))}
           for c in tr.class_names if c in rep}
    data = {"zaman": _dt.datetime.now().isoformat(timespec="seconds"),
            "dogruluk": float(tr.test_accuracy), "makro_f1": float(tr.test_macro_f1),
            "n_train": int(tr.n_train), "n_val": int(tr.n_val), "n_test": int(tr.n_test),
            "n_test_kayit": int(tr.n_test_recordings), "n_tur": len(tr.class_names), "tur": per}
    path = os.path.join(folder, SINGLE_SPLIT_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return path


def load_single_split(folder: str) -> Optional[dict]:
    p = os.path.join(folder, SINGLE_SPLIT_FILE)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _emb_keys(methods: dict) -> List[str]:
    """Plato karşılaştırmasına giren yöntemler: üç model varsa onlar, yoksa eski CSV çalıştırmasının BirdNET/Perch yöntemleri."""
    new = [k for k in MODEL_IDS if k in methods]
    return new if len(new) >= 2 else [k for k in EMBEDDING_METHODS if k in methods]


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1")


def read_semicolon_csv(path: str) -> List[dict]:
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


@dataclass
class Diagnosis:
    species_table: pd.DataFrame            # tur, segment, kayit, uygun, f1 (varsa)
    n_species: int = 0
    n_eligible: int = 0
    n_segments: int = 0
    n_segments_excluded: int = 0
    min_segment: int = DEFAULT_MIN_SEGMENTS
    min_kayit: int = 2
    has_cv: bool = False
    methods: Dict[str, dict] = field(default_factory=dict)   # yontem -> {dogruluk, dogruluk_ss, makro_f1, makro_f1_ss}
    embedding_acc_spread: Optional[float] = None
    plateau: Optional[bool] = None
    model_mode: bool = False                       # True: sonuçlar kayıtlı model sonuçlarından (BirdNET/Perch/birleşik)
    model_infos: List[dict] = field(default_factory=list)
    model_notes: List[str] = field(default_factory=list)
    min_p_models: Optional[float] = None            # modeller arası en küçük düzeltilmiş p (model modu)
    model_f1: Dict[str, Dict[str, float]] = field(default_factory=dict)   # model -> tür -> F1
    rho: Optional[float] = None
    rho_p: Optional[float] = None
    n_rho: int = 0
    seg_min: Optional[int] = None   # F1'i olan türlerin en az / en çok segmenti (ρ'nun geçerlilik aralığı)
    seg_max: Optional[int] = None
    n_weak: int = 0
    weakest: List[dict] = field(default_factory=list)
    paired: List[dict] = field(default_factory=list)
    selective: List[dict] = field(default_factory=list)
    best_new: str = ""
    fp_state: str = ""        # "guncel" | "eski" | "yok" | "" (veri/iz bilinmiyor)
    fp_note: str = ""
    stale_note: str = ""
    single: Optional[dict] = None
    times: Dict[str, float] = field(default_factory=dict)
    data_time: Optional[float] = None
    age_note: str = ""
    verdict: str = ""                      # "veri", "model", "belirsiz", "cv_yok"
    summary_rows: List[dict] = field(default_factory=list)
    per_class_rows: List[dict] = field(default_factory=list)

    def format(self) -> str:
        L = ["<b>🔬 Başarı Sınırı Analizi</b>", ""]
        pct = 100 * self.n_segments_excluded / self.n_segments if self.n_segments else 0
        L.append(f"<b>Veri:</b> {self.n_segments} segment, {self.n_species} tür. Eğitim süzgecini "
                 f"(≥{self.min_segment} segment ve ≥{self.min_kayit} farklı kayıt) {self.n_eligible} tür geçiyor; "
                 f"{self.n_species - self.n_eligible} tür dışarıda ({self.n_segments_excluded} segment, %{pct:.0f}).")
        if self.n_species - self.n_eligible > 0:
            L.append("Dışarıda kalan türler sınıflandırılamaz; ancak tür zenginliği hesabında yer alır. "
                     "Bu türler için etiketli segment artırılmalıdır.")
        if not self.has_cv and self.model_mode:
            L += ["", "Hiçbir modelin kayıtlı <b>çapraz doğrulama</b> sonucu yok. BirdNET 2.4, Perch 2.0 ve "
                      "BirdNET + Perch için ilgili eğitim penceresinde eğitimi (çapraz doğrulama açık) çalıştırın; "
                      "sonuçlar buraya otomatik gelir."]
            L += [f"• {n}" for n in self.model_notes]
            L += self._single_lines()
            return "<br>".join(L)
        if not self.has_cv:
            L += ["", "Çapraz doğrulama sonuçları bulunamadı "
                      f"({SUMMARY_FILE}, {PER_CLASS_FILE}). Yöntem karşılaştırması ve tür bazlı başarı için proje "
                      "klasöründe şunu çalıştırın:",
                  "<code>python capraz_dogrulama.py --temel-cizgi --sifir-atis</code>",
                  "veya yukarıdaki \"Sınıflandırmayı Çalıştır\" düğmesini kullanın."]
            L += self._single_lines()
            return "<br>".join(L)

        src = self.times.get(SUMMARY_FILE) or self.times.get(PER_CLASS_FILE)
        if self.model_mode:
            L += ["", "<b>Model karşılaştırması</b> (kayıt bazlı gruplu, tekrarlı çapraz doğrulama; ortalama ± s.s.). "
                      "<i>Kaynak: her modelin kayıtlı eğitim sonucu</i>"]
        else:
            L += ["", f"<b>Yöntem karşılaştırması</b> (kayıt bazlı gruplu, tekrarlı çapraz doğrulama; ortalama ± s.s.). "
                      f"<i>Kaynak: {SUMMARY_FILE}, {_fmt_time(src)}</i>"]
        if self.fp_note:
            L.append(("✓ " if self.fp_state == "guncel" else "⚠️ ") + self.fp_note)
        if self.age_note:
            L.append(f"⚠️ {self.age_note}")
        for k in reversed(ORDER):
            if k in self.methods:
                m = self.methods[k]
                inf = next((i for i in self.model_infos if i["id"] == k), None)
                ek = (f" <i>({inf['n_kat']}×{inf['n_tekrar']} çapraz doğrulama; {_fmt_time(inf['zaman'])})</i>"
                      if inf else "")
                L.append(f"• {METHOD_NAMES[k]}: doğruluk {m['dogruluk']:.3f} ± {m['dogruluk_ss']:.3f}; "
                         f"makro F1 {m['makro_f1']:.3f} ± {m['makro_f1_ss']:.3f}{ek}")
        L += [f"⚠️ {n}" for n in self.model_notes]
        if self.embedding_acc_spread is not None and self.model_mode and self.min_p_models is not None:
            sonuc = ("modeller arasında istatistiksel olarak anlamlı fark yok (örnek küçük olduğundan bu, farkın "
                     "olmadığını kanıtlamaz)" if self.plateau else "en az bir model farkı anlamlı")
            L.append(f"Modellerin doğruluk aralığı {self.embedding_acc_spread:.3f}; eşleştirilmiş testlerde en küçük "
                     f"düzeltilmiş p = {self.min_p_models:.3f}: {sonuc}.")
        elif self.embedding_acc_spread is not None:
            ad = "Modellerin" if self.model_mode else "BirdNET tabanlı yöntemlerin"
            L.append(f"{ad} doğruluk aralığı: {self.embedding_acc_spread:.3f} "
                     f"(plato eşiği ≤{PLATEAU_ACC_SPREAD}, buluşsal).")
        f1s = [self.methods[k]["makro_f1"] for k in _emb_keys(self.methods)]
        if self.plateau and len(f1s) >= 2 and max(f1s) - min(f1s) > 0.05:
            L.append(f"Doğruluk platoda, ancak makro F1 yöntemler arasında {max(f1s) - min(f1s):.3f} farklı: "
                     "doğruluk çok örnekli türlerce belirlenir, makro F1 ise az örnekli türlerin başarısını yansıtır.")
        for p in self.paired:
            L.append(f"• {p['karsilastirma']} ({p['metrik']}): fark {float(p['fark_ort']):+.3f}, "
                     f"düzeltilmiş p {'< 0.001' if float(p['p_duzeltilmis']) < 0.001 else '= %.3f' % float(p['p_duzeltilmis'])}")
        if self.model_mode and self.paired:
            L.append("<i>p değerleri Nadeau–Bengio düzeltmeli eşleştirilmiş testten gelir (Model Karşılaştırma "
                     "penceresiyle aynı); fark = ilk model − ikinci model. Bu veri kümesinin yeniden örneklenen "
                     "bölmelerinde modellerin farkını sınar, yeni nokta/alan için genelleme iddiası değildir.</i>")
        elif len(self.paired) > 2:
            L.append("<i>Çok sayıda yöntem karşılaştırıldığı için p değerleri çoklu karşılaştırma düzeltmesi içermez; "
                     "keşifsel kabul edin. Fark = MLP − yöntem (pozitif: MLP üstün).</i>")

        L += self._selective_lines()
        if self.model_mode:
            L += ["", "<b>Tür bazlı başarı</b> <i>(kaynak: modellerin çapraz doğrulaması; tür başına model ortalaması, "
                      "zayıflık en düşük modele göre)</i>:"]
        else:
            L += ["", f"<b>Tür bazlı başarı</b> <i>(kaynak: {PER_CLASS_FILE}, {_fmt_time(self.times.get(PER_CLASS_FILE))})</i>:"]
        if self.rho is not None:
            ptxt = f", p = {self.rho_p:.3f}" if self.rho_p is not None else ""
            rng = f"; segment aralığı {self.seg_min}–{self.seg_max}" if self.seg_min is not None else ""
            L.append(f"Tür başına segment sayısı ile F1 arasındaki Spearman ρ = {self.rho:.2f} "
                     f"(n = {self.n_rho} tür{ptxt}{rng}). F1 &lt; {WEAK_F1} olan tür sayısı: {self.n_weak}.")
        if self.weakest:
            L.append("En zayıf türler (F1; gerçek segment / kayıt sayısı):")
            for w in self.weakest:
                L.append(f"• {w.get('etiket', w['tur'])}: F1 {w['f1']:.2f} (segment {w['segment']}, kayıt {w['kayit']})")
        if self.stale_note:
            L += ["", f"⚠️ {self.stale_note}"]

        L += self._single_lines()

        L += ["", "<b>Sonuç:</b> " + self.verdict_text(), "",
              "<i>Not: Bu kanıtlar ilişkiseldir, nedensellik kanıtlamaz (zor türler aynı zamanda nadir olabilir). "
              "Kesin sınama öğrenme eğrisidir: eğitim verisini artırınca başarı hâlâ yükseliyorsa sınır veridir.</i>"]
        return "<br>".join(L)

    def _selective_lines(self) -> List[str]:
        if not self.selective:
            return []
        ms = {r["yontem"] for r in self.selective}
        pick = next((m for m in ("MLP_Topluluk", "Topluluk_BirdNET", "LR_BirdNET") if m in ms), None)
        if pick is None:
            return []
        rows = [r for r in self.selective if r["yontem"] == pick]
        L = ["", f"<b>Seçici tahmin</b> ({METHOD_NAMES.get(pick, pick)}; olasılık eşiğinin altında tahmin yapılmaz; "
                 f"ECE = {float(rows[0]['ece']):.3f}) <i>(kaynak: {SELECTIVE_FILE})</i>:"]
        for r in rows:
            if float(r["esik"]) in (0.0, 0.5, 0.7, 0.9):
                L.append(f"• eşik {float(r['esik']):.1f}: kapsam %{100 * float(r['kapsam']):.0f}, "
                         f"kabul edilenlerde doğruluk {float(r['dogruluk_kabul']):.3f}, "
                         f"makro F1 {float(r['makro_f1_kabul']):.3f}")
        return L

    def _single_lines(self) -> List[str]:
        L: List[str] = []
        if self.single:
            sg = self.single
            L += ["", f"<b>Son eğitim (tek bölme)</b> <i>(kaynak: {SINGLE_SPLIT_FILE}, "
                      f"{sg.get('zaman', '?').replace('T', ' ')})</i>:",
                  f"Test doğruluğu {sg['dogruluk']:.3f}; makro F1 {sg['makro_f1']:.3f}; eğitim {sg['n_train']}, "
                  f"doğrulama {sg['n_val']}, test {sg['n_test']} segment ({sg['n_test_kayit']} kayıt); {sg['n_tur']} tür.",
                  "Tek bir bölme olduğu için özellikle az örnekli türlerin F1'i hangi kayıtların teste düştüğüne çok bağlıdır; "
                  "genel sonuç olarak çapraz doğrulamayı esas alın."]
            few = sorted((v["destek"], k) for k, v in sg.get("tur", {}).items() if v["destek"] <= 5)
            if few:
                L.append(f"Testte ≤5 segmenti olan tür sayısı: {len(few)} (bu türlerin F1'i çok gürültülüdür).")

        return L

    def verdict_text(self) -> str:
        if not self.plateau:
            plato = ""
        elif self.model_mode:
            mp = f"{self.min_p_models:.3f}" if self.min_p_models is not None else "?"
            plato = (f"Modeller arasında istatistiksel olarak anlamlı fark bulunamadı (en küçük düzeltilmiş p = {mp}; "
                     "örnek küçük olduğundan bu, farkın olmadığını kanıtlamaz): gömme/model seçimi başarıyı "
                     "kanıtlanabilir biçimde değiştirmiyor. ")
        else:
            plato = ("Aynı BirdNET gömmesi üzerinde sınıflandırıcıyı karmaşıklaştırmak (LR → MLP) kazanç "
                     "getirmiyor. ")
        if self.verdict == "veri":
            return (f"Sınırlayıcı etken büyük olasılıkla <b>veridir</b>: {plato}"
                    "zayıf türler az segmentli türlerdir. "
                    "Öncelik: zayıf türler için etiketli segment/kayıt toplamak.")
        if self.verdict == "kapsam":
            out = self.n_species - self.n_eligible
            pct = 100 * out / self.n_species if self.n_species else 0
            rho = ""
            if self.rho is not None:
                p = f", p = {self.rho_p:.3f}" if self.rho_p is not None else ""
                rho = (f" Segment sayısı–F1 ilişkisi (ρ = {self.rho:.2f}{p}, n = {self.n_rho}) yalnızca süzgeci geçen, "
                       f"dar bir aralıktaki ({self.seg_min}–{self.seg_max} segment) türlerde ölçüldü; "
                       "dar aralık ve küçük n nedeniyle sınırlı bir kanıttır ve nedensellik göstermez.")
            return (f"Süzgeci geçen {self.n_eligible} türün tamamında F1 ≥ {WEAK_F1}: bu türlerde başarıyı "
                    f"sınırlayan bir zayıflık görülmüyor. {plato}"
                    f"Asıl sınır <b>veri kapsamıdır</b>: {out} tür (%{pct:.0f}) eğitim süzgecini geçemediği için "
                    f"sınıflandırılamıyor; bu türler tür zenginliği hesabında yer alır ama kapalı küme sınıflandırıcısında yer almaz."
                    + rho)
        if self.verdict == "model":
            if self.model_mode:
                mp = f" (en küçük düzeltilmiş p = {self.min_p_models:.3f})" if self.min_p_models is not None else ""
                return (f"Modeller arasında anlamlı fark var{mp}: gömme/model seçimi başarıyı etkiliyor; "
                        "en iyi modelin seçimi gerekçelendirilebilir.")
            return ("Yöntemler arasında belirgin fark var; model/özellik seçimi hâlâ başarıyı etkiliyor. "
                    "Daha güçlü yöntemler denemeye değer.")
        return ("Kanıtlar tek başına yeterli değil (plato ve destek–başarı ilişkisi birlikte sağlanmıyor). "
                "Öğrenme eğrisiyle sınanması önerilir.")


def species_support(df: pd.DataFrame, min_segment: int = DEFAULT_MIN_SEGMENTS, min_kayit: int = 2) -> pd.DataFrame:
    """Tür başına segment ve farklı kayıt sayısı; eğitim süzgecini geçip geçmediği."""
    if "Scientific_Name" not in df.columns:
        return pd.DataFrame(columns=["tur", "segment", "kayit", "uygun", "etiket"])
    d = df.copy()
    d["tur"] = d["Scientific_Name"].astype(str).str.strip()
    d = d[(d["tur"] != "") & (d["tur"].str.lower() != "nan")]
    rec = "Original_Recording_Name" if "Original_Recording_Name" in d.columns else None
    g = d.groupby("tur")
    out = pd.DataFrame({"segment": g.size(),
                        "kayit": g[rec].nunique() if rec else 1}).reset_index()
    out["uygun"] = (out["segment"] >= min_segment) & (out["kayit"] >= min_kayit)
    # Gösterim etiketi: "Türkçe ad (Latince ad)"; Türkçe ad yoksa yalnızca Latince. Eşleştirme anahtarı "tur" (Latince) kalır.
    out["etiket"] = out["tur"]
    if "Turkish_Name" in d.columns:
        tr = d["Turkish_Name"].astype(str).str.strip()
        tr = tr.where((tr != "") & (tr.str.lower() != "nan"))
        ad = d.assign(_tr=tr).dropna(subset=["_tr"]).groupby("tur")["_tr"].agg(lambda x: x.mode().iloc[0])
        out["etiket"] = [f"{ad[t]} ({t})" if t in ad.index else t for t in out["tur"]]
    return out.sort_values("segment", ascending=False).reset_index(drop=True)


def _method_table(summary_rows: List[dict]) -> Dict[str, dict]:
    best: Dict[str, dict] = {}
    for r in summary_rows:
        k = r["yontem"]
        if k not in best or _truthy(r["sinif_agirligi"]):   # ağırlıklı satır varsa onu tercih et
            best[k] = r
    return {k: {"dogruluk": float(r["dogruluk_ort"]), "dogruluk_ss": float(r["dogruluk_ss"]),
                "makro_f1": float(r["makro_f1_ort"]), "makro_f1_ss": float(r["makro_f1_ss"])}
            for k, r in best.items()}


def _per_class_f1(rows: List[dict]) -> Dict[str, float]:
    use = [r for r in rows if _truthy(r.get("sinif_agirligi"))] or list(rows)
    return {r["tur"]: float(r["f1"]) for r in use}


def diagnose(df: pd.DataFrame, summary_rows: Optional[List[dict]] = None,
             per_class_rows: Optional[List[dict]] = None, paired_rows: Optional[List[dict]] = None,
             min_segment: int = DEFAULT_MIN_SEGMENTS, min_kayit: int = 2, single_split: Optional[dict] = None,
             selective_rows: Optional[List[dict]] = None,
             times: Optional[Dict[str, float]] = None, data_time: Optional[float] = None,
             fp_saved: Optional[dict] = None, fp_now: Optional[dict] = None,
             model_infos: Optional[List[dict]] = None, model_fps: Optional[Dict[str, Optional[dict]]] = None,
             model_notes: Optional[List[str]] = None, model_mode: bool = False) -> Diagnosis:
    sp = species_support(df, min_segment, min_kayit)
    d = Diagnosis(species_table=sp, n_species=len(sp), n_eligible=int(sp["uygun"].sum()) if len(sp) else 0,
                  n_segments=int(sp["segment"].sum()) if len(sp) else 0,
                  n_segments_excluded=int(sp.loc[~sp["uygun"], "segment"].sum()) if len(sp) else 0,
                  min_segment=min_segment, min_kayit=min_kayit)
    d.single, d.times, d.data_time = single_split, times or {}, data_time
    d.model_mode = bool(model_mode or model_infos)
    d.model_infos, d.model_notes = list(model_infos or []), list(model_notes or [])
    d.selective = list(selective_rows or [])
    summary_rows, per_class_rows = summary_rows or [], per_class_rows or []
    d.summary_rows, d.per_class_rows = summary_rows, per_class_rows
    if not summary_rows and not per_class_rows:
        d.verdict = "cv_yok"
        return d
    d.has_cv = True
    cv_t = d.times.get(SUMMARY_FILE) or d.times.get(PER_CLASS_FILE)
    if fp_now and model_fps:
        ad = {i["id"]: i["ad"] for i in d.model_infos}
        eski = [ad.get(k, k) for k, v in model_fps.items() if v and v.get("hash") and v["hash"] != fp_now["hash"]]
        yok = [ad.get(k, k) for k, v in model_fps.items() if not (v and v.get("hash"))]
        if eski:
            d.fp_state = "eski"
            d.fp_note = (f"ESKİ veriyle eğitilmiş model(ler): {', '.join(eski)}. Şimdiki veri: {fp_now['n_segment']} "
                         f"segment, {fp_now['n_tur']} tür, {fp_now['n_kayit']} kayıt. İlgili modeli yeniden eğitin; "
                         "modeller farklı veriyle üretildiyse doğrudan karşılaştırılamaz.")
        elif yok:
            d.fp_state = "yok"
            d.fp_note = (f"Veri izi kayıtlı değil ({', '.join(yok)}): bu sonuçların hangi veriyle üretildiği bilinmiyor; "
                         "kesin olması için yeniden eğitin.")
        else:
            d.fp_state = "guncel"
            d.fp_note = (f"Tüm modeller güncel veriyle eğitildi ({fp_now['n_segment']} segment, {fp_now['n_tur']} tür, "
                         f"{fp_now['n_kayit']} kayıt; veri izi {fp_now['hash']}).")
    elif fp_now and fp_saved and fp_saved.get("hash"):
        if fp_saved["hash"] == fp_now["hash"]:
            d.fp_state = "guncel"
            d.fp_note = (f"Sonuçlar güncel veriyle üretildi ({fp_now['n_segment']} segment, {fp_now['n_tur']} tür, "
                         f"{fp_now['n_kayit']} kayıt; veri izi {fp_now['hash']}).")
        else:
            d.fp_state = "eski"
            d.fp_note = (f"Sonuçlar ESKİ veriyle üretildi: {fp_saved.get('n_segment', '?')} segment, "
                         f"{fp_saved.get('n_tur', '?')} tür, {fp_saved.get('n_kayit', '?')} kayıt. Şimdiki veri: "
                         f"{fp_now['n_segment']} segment, {fp_now['n_tur']} tür, {fp_now['n_kayit']} kayıt. "
                         "'Sınıflandırmayı Çalıştır' ile yenileyin.")
    elif fp_now:
        d.fp_state = "yok"
        d.fp_note = ("Bu sonuçların hangi veriyle üretildiği kayıtlı değil (veri izi yok; eski çalıştırma). "
                     "Güncelliği dosya zamanından tahmin ediliyor; kesin olması için yeniden çalıştırın.")
    if d.fp_state in ("", "yok") and cv_t and data_time and data_time > cv_t + 1:
        d.age_note = (f"Veri seti ({_fmt_time(data_time)}) bu sonuçlardan ({_fmt_time(cv_t)}) daha yeni; "
                      "sonuçlar güncel veriyi yansıtmıyor olabilir. 'Sınıflandırmayı Çalıştır' ile yenileyin.")
    d.methods = _method_table(summary_rows)
    emb = [d.methods[k]["dogruluk"] for k in _emb_keys(d.methods)]
    if len(emb) >= 2:
        d.embedding_acc_spread = max(emb) - min(emb)
        d.plateau = d.embedding_acc_spread <= PLATEAU_ACC_SPREAD
    seen = set()
    for p in (paired_rows or []):
        # LR_BirdNET hem sınıf ağırlıklı hem ağırlıksız karşılaştırılır; aynı adla ikinci satırı (ağırlıksız) atla.
        key = (p.get("karsilastirma"), p.get("metrik"))
        if str(p.get("metrik")) == "makro_f1" and key not in seen:
            seen.add(key)
            d.paired.append(p)

    if d.model_mode and d.paired and len(emb) >= 2:
        # Üç modelde "plato" kararı: doğruluk aralığı (buluşsal) değil, eşleştirilmiş testin sonucu belirler.
        ps = [float(q["p_duzeltilmis"]) for q in d.paired]
        d.min_p_models = min(ps)
        d.plateau = not any(v < 0.05 for v in ps)

    by_model = _per_class_by_model(per_class_rows)
    tbl = None
    if by_model:
        d.model_f1 = by_model
        tbl = pd.DataFrame(by_model)                    # satır: tür, sütun: model
        f1 = tbl.mean(axis=1, skipna=True).to_dict()
    else:
        f1 = _per_class_f1(per_class_rows)
    if f1:
        t = sp.set_index("tur")
        sp2 = sp.copy()
        sp2["f1"] = sp2["tur"].map(f1)
        if tbl is not None:
            sp2["f1_min"] = sp2["tur"].map(tbl.min(axis=1))
            sp2["f1_max"] = sp2["tur"].map(tbl.max(axis=1))
            sp2["n_model"] = sp2["tur"].map(tbl.notna().sum(axis=1))
        d.species_table = sp2
        both = sp2.dropna(subset=["f1"])
        zayif = both["f1_min"] if "f1_min" in both.columns else both["f1"]
        d.n_rho = len(both)
        d.n_weak = int((zayif < WEAK_F1).sum())
        if len(both) >= 5 and both["segment"].nunique() > 1 and both["f1"].nunique() > 1:
            from scipy import stats
            _r = stats.spearmanr(both["segment"], both["f1"])
            d.rho, d.rho_p = float(_r[0]), float(_r[1])
        if len(both):
            d.seg_min, d.seg_max = int(both["segment"].min()), int(both["segment"].max())
        for _, r in both.assign(_z=zayif).sort_values("_z").head(5).iterrows():
            if r["_z"] < WEAK_F1:
                d.weakest.append({"tur": r["tur"], "etiket": r.get("etiket", r["tur"]), "f1": float(r["_z"]),
                                  "segment": int(r["segment"]), "kayit": int(r["kayit"])})
        missing = set(f1) - set(t.index)
        extra = int(sp["uygun"].sum()) - len(set(f1) & set(t.index))
        if missing or extra > 0:
            d.stale_note = (f"Çapraz doğrulama dosyası {len(f1)} tür içeriyor; veri setinde şu an {d.n_eligible} uygun "
                            f"tür var ({len(missing)} tür veri setinde yok, {max(extra, 0)} uygun tür doğrulamada yok). "
                            "Sonuçlar eski olabilir; doğrulamayı yeniden çalıştırın.")

    strong = d.rho is not None and d.rho >= STRONG_RHO and (d.rho_p is None or d.rho_p < 0.05)
    if d.plateau is False:
        d.verdict = "model"
    elif d.n_rho > 0 and d.n_weak == 0:
        # Süzgeci geçen hiçbir tür zayıf değil: sınır başarıda değil, süzgeç dışında kalan türlerde (kapsam).
        d.verdict = "kapsam"
    elif d.plateau and strong:
        d.verdict = "veri"
    else:
        d.verdict = "belirsiz"
    return d


def _per_class_by_model(rows: List[dict]) -> Dict[str, Dict[str, float]]:
    """Satırlar 'yontem' = model kimliği taşıyorsa {model: {tür: F1}}; yoksa boş (eski tek-model CSV)."""
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        m = r.get("yontem")
        if m in MODEL_IDS:
            out.setdefault(m, {})[r["tur"]] = float(r["f1"])
    return out


def from_stored(stored_list) -> dict:
    """
    [(ModelSpec, StoredResult)] (model_registry.stored_results) -> diagnose() için anahtar sözcüklü argümanlar.
    Her modelin kayıtlı çapraz doğrulama özetinden (cv_summary) yöntem ve tür bazlı F1 satırları üretilir;
    modeller arası fark, Model Karşılaştırma penceresiyle AYNI eşleştirilmiş testten (Nadeau–Bengio) gelir.
    Çapraz doğrulaması olmayan (yalnızca tek bölme) model yöntem tablosuna alınmaz, nedeni notlanır.
    """
    from . import model_significance
    summary, per_class, infos, fps, notes = [], [], [], {}, []
    for spec, st in stored_list:
        cv = st.result.training_result.cv_summary
        if not (isinstance(cv, dict) and "dogruluk" in cv):
            notes.append(f"{spec.ad}: çapraz doğrulama sonucu yok (yalnızca tek bölme); bu analize alınmadı.")
            continue
        cw = str(bool(cv.get("sinif_agirligi", True)))
        summary.append({"yontem": spec.id, "sinif_agirligi": cw,
                        "n_kat": int(cv.get("n_kat", 0)) * int(cv.get("n_tekrar", 1)),
                        "dogruluk_ort": cv["dogruluk"][0], "dogruluk_ss": cv["dogruluk"][1],
                        "makro_f1_ort": cv["makro_f1"][0], "makro_f1_ss": cv["makro_f1"][1]})
        tur_f1 = cv.get("tur_f1") or {}
        if not tur_f1:
            notes.append(f"{spec.ad}: tür bazlı F1 kayıtlı değil (eski kayıt); yeniden eğitin.")
        for tur, f in tur_f1.items():
            per_class.append({"yontem": spec.id, "sinif_agirligi": cw, "tur": tur, "f1": f})
        infos.append({"id": spec.id, "ad": spec.ad, "zaman": st.saved_at.timestamp(),
                      "n_kat": cv.get("n_kat"), "n_tekrar": cv.get("n_tekrar")})
        fps[spec.id] = st.fingerprint
    paired = []
    try:
        sig = model_significance.compare_stored(stored_list)
    except Exception as exc:  # noqa: BLE001
        sig = {"satirlar": [], "uyarilar": [f"Modeller arası test hesaplanamadı: {exc}"]}
    for r in sig["satirlar"]:
        if r.get("durum") == "ok" and r.get("p_duzeltilmis") is not None:
            paired.append({"karsilastirma": f"{r['a']} − {r['b']}", "metrik": r["metrik"],
                           "fark_ort": r["fark"], "p_duzeltilmis": r["p_duzeltilmis"]})
    notes += [n for n in sig["uyarilar"] if n not in notes]
    return {"summary_rows": summary, "per_class_rows": per_class, "paired_rows": paired,
            "model_infos": infos, "model_fps": fps, "model_notes": notes, "model_mode": True}


def load_cv_results(folder: str):
    return (read_semicolon_csv(os.path.join(folder, SUMMARY_FILE)),
            read_semicolon_csv(os.path.join(folder, PER_CLASS_FILE)),
            read_semicolon_csv(os.path.join(folder, PAIRED_FILE)))


def load_selective(folder: str) -> List[dict]:
    return read_semicolon_csv(os.path.join(folder, SELECTIVE_FILE))
