"""
tez_sekilleri.py
================
Tez için yüksek çözünürlüklü (varsayılan 300 dpi PNG + vektörel PDF), Türkçe başlıklı ve
yazılımdaki grafiklerle AYNI stilde şekilleri üretir. Girdi dosyası bulunmayan şekiller atlanır.

Kullanım (proje klasöründe):
    python tez_sekilleri.py                                   # varsayılan dosya adlarıyla
    python tez_sekilleri.py --csv bird_sound_dataset.csv --dpi 600 --cikti tez_sekilleri

Girdiler (hepsi isteğe bağlı):
    --csv         etiketli veri seti (; ayraçlı)        -> birikim eğrisi, bölge, nokta başına tür
    --kok         proje klasörü (varsayılan: .)         -> BirdNET 2.4 / Perch 2.0 / BirdNET+Perch kayıtlı çapraz
                                                           doğrulama sonuçlarından yöntem karşılaştırması, tür bazlı F1,
                                                           veri yeterliliği ve destek–F1 şekilleri
    --cv-ozet / --cv-tur  (eski) cv_yontem_ozeti.csv / cv_tur_bazli.csv -> yalnızca AÇIKÇA verilirse kullanılır;
                          eski dosyalar projenin üç modelinin dışındaki yöntemleri (örn. MFCC temel çizgisi) içerebilir
    --zen-nokta / --zen-ozet  zenginlik_nokta.csv / zenginlik_ozet.csv -> uzman-model zenginlik karşılaştırması
Çıktı: --cikti klasörüne sekil_XX_ad.png / .pdf
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")


def _oku(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def _kaydet(fig, klasor, ad, dpi):
    os.makedirs(klasor, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(klasor, f"{ad}.{ext}"), dpi=dpi, bbox_inches="tight", facecolor="white")
    print("Yazıldı:", os.path.join(klasor, ad + ".png"), "(+ .pdf)")


def uret(csv_yolu, cv_ozet, cv_tur, klasor, dpi, zen_nokta=None, zen_ozet=None, kok=None):
    """Üretilen dosya adlarının (uzantısız) listesini döndürür."""
    import pandas as pd
    from src.data import dataset_summary
    from src.visualization import richness_plots as rp, thesis_plots as tp

    uretilen = []

    def yaz(fig, ad):
        _kaydet(fig, klasor, ad, dpi)
        uretilen.append(ad)

    if csv_yolu and os.path.isfile(csv_yolu):
        df = dataset_summary.load_dataset(csv_yolu)
        if df is not None:
            sm = dataset_summary.compute_summary(df)
            if sm.pooled_curve is not None and len(sm.pooled_curve) > 0:
                yaz(rp.accumulation(sm.pooled_curve), "sekil_01_birikim_egrisi")
            if sm.location_rarefaction is not None and len(sm.location_rarefaction) > 1:
                yaz(rp.location_rarefaction(sm.location_rarefaction), "sekil_03_bolge_seyreltme")
            if len(sm.habitat_richness) > 0:
                yaz(rp.point_richness_chart(sm.habitat_richness), "sekil_04_nokta_basina_tur")
    if kok and csv_yolu and os.path.isfile(csv_yolu):
        from src.ml import model_registry, classification_diagnosis as cd
        from src.visualization import classification_plots as cp
        stored = model_registry.stored_results(kok)
        df = dataset_summary.load_dataset(csv_yolu)
        if stored and df is not None:
            d = cd.diagnose(df, fp_now=cd.data_fingerprint(csv_yolu), **cd.from_stored(stored))
            if len(d.species_table):
                yaz(cp.data_sufficiency(d.species_table, d.min_segment, d.min_kayit), "sekil_05_veri_yeterliligi")
            if d.has_cv:
                yaz(tp.cv_methods(d.summary_rows, "makro_f1"), "sekil_07_cv_yontemler_makro_f1")
                yaz(tp.cv_methods(d.summary_rows, "dogruluk"), "sekil_08_cv_yontemler_dogruluk")
                if "f1" in d.species_table.columns and d.species_table["f1"].notna().sum() >= 3:
                    yaz(cp.support_vs_f1(d.species_table, d.rho, p=d.rho_p), "sekil_06_destek_f1")
                    t = d.species_table.dropna(subset=["f1"])
                    rows = [{"tur": r["etiket"], "f1": r["f1"], "destek_segment": int(r["segment"]),
                             "sinif_agirligi": "True"} for _, r in t.iterrows()]
                    yaz(tp.cv_per_class(rows), "sekil_09_cv_tur_f1")
    if cv_ozet and os.path.isfile(cv_ozet):
        yaz(tp.cv_methods(_oku(cv_ozet), "makro_f1"), "sekil_07_cv_yontemler_makro_f1")
        yaz(tp.cv_methods(_oku(cv_ozet), "dogruluk"), "sekil_08_cv_yontemler_dogruluk")
    if cv_tur and os.path.isfile(cv_tur):
        yaz(tp.cv_per_class(_oku(cv_tur)), "sekil_09_cv_tur_f1")
    if zen_nokta and zen_ozet and os.path.isfile(zen_nokta) and os.path.isfile(zen_ozet):
        yaz(tp.richness_agreement(_oku(zen_nokta), _oku(zen_ozet)), "sekil_10_uzman_model_zenginlik")
    return uretilen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="bird_sound_dataset.csv")
    ap.add_argument("--kok", default=".", help="proje klasörü (modellerin kayıtlı sonuçları burada aranır)")
    ap.add_argument("--cv-ozet", dest="cv_ozet", default=None)
    ap.add_argument("--cv-tur", dest="cv_tur", default=None)
    ap.add_argument("--zen-nokta", dest="zen_nokta", default="zenginlik_nokta.csv")
    ap.add_argument("--zen-ozet", dest="zen_ozet", default="zenginlik_ozet.csv")
    ap.add_argument("--cikti", default="tez_sekilleri")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    n = uret(a.csv, a.cv_ozet, a.cv_tur, a.cikti, a.dpi, a.zen_nokta, a.zen_ozet, kok=a.kok)
    if not n:
        print("Hiçbir girdi dosyası bulunamadı; --csv/--cv-ozet/--cv-tur yollarını kontrol edin.")


if __name__ == "__main__":
    main()
