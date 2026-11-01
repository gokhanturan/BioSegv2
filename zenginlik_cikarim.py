"""
zenginlik_cikarim.py
====================
Uçtan uca zincir: ses kayıtları -> Perch gömmeleri -> sınıflandırıcı -> nokta (Habitat_ID)
başına tür listesi -> tür zenginliği belirleme (belirlenen tür sayısı). Ayrıntı ve sınırlar:
src/ml/richness_inference.py

İki kip:

1) DOĞRULAMA (varsayılan): uzman etiketli noktalardaki TÜM kaydı tara; her kaydı eğitimde
   görmemiş bir modelle (kayıt bazlı çapraz doğrulama) tür listesi çıkar ve uzman listesiyle
   karşılaştır.
       python zenginlik_cikarim.py --ses-klasoru "D:\\kayitlar"

2) UYGULAMA: tüm etiketli veriyle eğitilen modeli yeni (etiketsiz) kayıtlara uygula.
   Eşleme dosyası (; ayraçlı): Dosya;Habitat_ID   (Dosya: dosya adı veya yol)
       python zenginlik_cikarim.py --kip uygula --esleme esleme.csv --ses-klasoru "D:\\yeni_kayitlar"

Ön-tanımlı kural (veriye bakılmadan sabitlenmiştir): güven eşiği 0.5, en az 2 bağımsız tespit.
Diğer kombinasyonlar duyarlılık analizi olarak raporlanır; en iyisi seçilmez.

Önemli: kayıt taraması uzun sürer (kayıt başına ~17 dk ses). Pencere gömmeleri
"tarama_onbellek" klasöründe saklanır; yarıda kesilirse kaldığı yerden devam eder.
Veri setine ve modele DOKUNMAZ (salt-okunur).
"""
import argparse
import csv
import os
import sys

import pandas as pd

from src.ml import richness_inference as RI
from src.ml.pipeline import load_or_extract_embeddings


def _yaz(name, rows):
    if not rows:
        return
    with open(name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def _ilerleme(etiket):
    return lambda c, t: print(f"\r{etiket}: {c}/{t}", end="", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kip", choices=["dogrula", "uygula"], default="dogrula")
    ap.add_argument("--ses-klasoru", default=None, help="kayıtların (tam uzunlukta) bulunduğu klasör (alt klasörler dahil)")
    ap.add_argument("--esleme", default=None, help="uygula kipi: Dosya;Habitat_ID CSV'si")
    ap.add_argument("--csv", default=None, help="veri seti CSV (varsayılan: proje CSV'si)")
    ap.add_argument("--yontem", choices=["mlp", "lr"], default="mlp")
    ap.add_argument("--atlama", type=float, default=3.0, help="pencere atlama süresi (sn); 3 = BirdNET penceresi, örtüşmesiz")
    ap.add_argument("--kat", type=int, default=5)
    ap.add_argument("--tekrar", type=int, default=3)
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--esikler", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3], help="en az bağımsız tespit sayıları")
    ap.add_argument("--onceden-esik", type=float, default=0.5)
    ap.add_argument("--onceden-k", type=int, default=2)
    ap.add_argument("--min-segment", type=int, default=30)
    ap.add_argument("--min-kayit", type=int, default=2)
    ap.add_argument("--onbellek", default="tarama_onbellek")
    ap.add_argument("--cikti-oneki", default="zenginlik_cikarim")
    a = ap.parse_args(argv)

    # --- Eğitim verisi (etiketli segmentler; gömmeler önbellekten) ---
    X, kept = load_or_extract_embeddings(
        min_samples_per_class=a.min_segment, min_recordings_per_class=a.min_kayit,
        csv_path=a.csv, cache_path="embedding_onbellek.npz",
        progress_callback=lambda st, c, t: print(f"\r{st}: {c}/{t}", end="", flush=True))
    class_names = sorted({s.species for s in kept})
    print(f"\nEğitim: {len(kept)} segment, {len({s.recording_id for s in kept})} kayıt, {len(class_names)} tür")

    from src.ml.birdnet_embedding import BirdNETEmbedder
    classifier = BirdNETEmbedder()
    index = RI.build_audio_index(a.ses_klasoru) if a.ses_klasoru else None

    if a.kip == "dogrula":
        csv_path = a.csv
        if csv_path is None:
            from src.ml.dataset_builder import get_csv_root
            csv_path = os.path.join(get_csv_root(), "bird_sound_dataset.csv")
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
        d = df.dropna(subset=["Original_Recording_Name", "Habitat_ID"]).drop_duplicates("Original_Recording_Name")
        point_of_rec = {str(r).strip(): str(h).strip() for r, h in zip(d["Original_Recording_Name"], d["Habitat_ID"])}
        records, eksik = [], []
        for name, path in zip(d["Original_Recording_Name"], d.get("Original_File_Path", [None] * len(d))):
            rid = str(name).strip()
            real = RI.resolve_audio(path if isinstance(path, str) else None, rid, index)
            (records if real else eksik).append({"rec_id": rid, "point": point_of_rec[rid], "path": real} if real else rid)
        print(f"Kayıt: {len(records)} bulundu, {len(eksik)} bulunamadı.")
        if eksik:
            print("⚠ Bulunamayan kayıtlar taramaya girmez (--ses-klasoru verin): " + ", ".join(eksik[:8]) + (" …" if len(eksik) > 8 else ""))
        if not records:
            print("Taranacak kayıt yok.")
            return 1
        table = RI.build_window_table(records, classifier, a.atlama, a.onbellek, _ilerleme("Tarama"))
        print(f"\nToplam pencere: {len(table['start'])}; başarısız kayıt: {len(table['failures'])}")
        preds = []
        for rep in range(a.tekrar):
            preds.append(RI.out_of_fold_window_proba(
                table, X, kept, class_names, method=a.yontem, n_splits=a.kat, seed=42 + rep,
                max_epochs=a.epoch, progress=_ilerleme(f"Model (tekrar {rep + 1}/{a.tekrar})")))
        print()
        rows = RI.validate_scan_vs_expert(table, preds, kept, class_names, point_of_rec, a.esikler, a.k,
                                          a.onceden_esik, a.onceden_k, min_gap=max(5.0, a.atlama))
        ozet = RI.summarize(rows)
        print(f"\nYöntem: {a.yontem} | nokta sayısı: {rows[0]['nokta_sayisi']} | evren: {len(class_names)} tür")
        for o in ozet:
            isaret = "  <- ön-tanımlı" if o["onceden_belirlenen"] else ""
            print(f"eşik {o['guven_esigi']:.2f}, k={o['en_az_tespit']}: tür uzman {o['tur_uzman_ort']:.0f} / "
                  f"tarama {o['tur_model_ort']:.1f} | nokta başına tür Spearman {o['nokta_basina_tur_spearman_ort']:.2f} | "
                  f"kesinlik {o['var_yok_kesinlik_ort']:.2f}, duyarlılık {o['var_yok_duyarlilik_ort']:.2f}, "
                  f"Jaccard {o['var_yok_jaccard_ort']:.2f} | sahte tür {o['sahte_tur_nokta_cifti_ort']:.0f}{isaret}")
        _yaz(f"{a.cikti_oneki}_dogrulama.csv", rows)
        _yaz(f"{a.cikti_oneki}_dogrulama_ozet.csv", ozet)
        pred0, conf0 = preds[0]
        _yaz(f"{a.cikti_oneki}_tur_listesi.csv", RI.species_lists(table, pred0, conf0, class_names, a.onceden_esik, a.onceden_k))
        print(f"Yazıldı: {a.cikti_oneki}_dogrulama.csv, _dogrulama_ozet.csv, _tur_listesi.csv")
        return 0

    # --- Uygulama kipi ---
    if not a.esleme or not os.path.isfile(a.esleme):
        print("Uygulama kipi için --esleme (Dosya;Habitat_ID) gerekir.")
        return 1
    m = pd.read_csv(a.esleme, sep=";", encoding="utf-8-sig", dtype=str)
    records, eksik = [], []
    for dosya, hid in zip(m["Dosya"], m["Habitat_ID"]):
        real = RI.resolve_audio(dosya, os.path.splitext(os.path.basename(str(dosya)))[0], index)
        if real:
            records.append({"rec_id": os.path.splitext(os.path.basename(real))[0], "point": str(hid).strip(), "path": real})
        else:
            eksik.append(str(dosya))
    print(f"Kayıt: {len(records)} bulundu, {len(eksik)} bulunamadı.")
    if not records:
        return 1
    table = RI.build_window_table(records, classifier, a.atlama, a.onbellek, _ilerleme("Tarama"))
    print(f"\nToplam pencere: {len(table['start'])}; nokta sayısı: {len({r['point'] for r in table['records']})}")
    pred, conf = RI.apply_model_to_table(table, X, kept, class_names, method=a.yontem, max_epochs=a.epoch)
    grid = RI.richness_grid(table, pred, conf, class_names, a.esikler, a.k, a.onceden_esik, a.onceden_k)
    for g in grid:
        isaret = "  <- ön-tanımlı" if g["onceden_belirlenen"] else ""
        print(f"eşik {g['guven_esigi']:.2f}, k={g['en_az_tespit']}: m={g['nokta_sayisi']}, belirlenen tür {g['belirlenen_tur']}{isaret}")
    _yaz(f"{a.cikti_oneki}_zenginlik.csv", grid)
    _yaz(f"{a.cikti_oneki}_tur_listesi.csv", RI.species_lists(table, pred, conf, class_names, a.onceden_esik, a.onceden_k))
    print(f"Yazıldı: {a.cikti_oneki}_zenginlik.csv, _tur_listesi.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
