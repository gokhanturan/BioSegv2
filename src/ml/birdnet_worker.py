"""
src/ml/birdnet_worker.py
========================
BirdNET gömme çıkarmayı AYRI, temiz bir Python sürecinde çalıştırır:

    python -m src.ml.birdnet_worker GIRDI.json CIKTI.npz KUTUPHANE PARCA_BOYUTU

`birdnet` paketi kendi alt süreçlerini (Windows'ta `spawn`) başlatır ve her alt süreç ana modülü yeniden
içe aktarır. Uygulama penceresinden çağrıldığında ana modül `src.main` olur (PyQt5, matplotlib, tüm arayüz);
bu da alt süreçlerin çökmesine ("Analysis was cancelled") yol açabilir. Bu modül yalnızca hafif bağımlılıklar
içerdiği için alt süreçler sorunsuz başlar.

Protokol (stdout, satır satır): "@@MODEL a b", "@@PROG a b", "@@ERR metin". Sonuç CIKTI.npz içine yazılır:
X (n, d), ok (n,) bool, n_fail, last_error.
"""
import json
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    in_json, out_npz, library, chunk = argv[0], argv[1], argv[2], int(argv[3])

    import numpy as np
    from src.ml import birdnet_embedding as bne
    from src.ml.dataset_builder import TrainingSample

    with open(in_json, "r", encoding="utf-8") as f:
        paths = json.load(f)
    samples = [TrainingSample(filepath=p, species="", recording_id="") for p in paths]

    bne.configure_model_dir()
    print("@@MODEL 0 1", flush=True)
    try:
        model = bne.get_birdnet_model(library)
    except Exception as exc:  # noqa: BLE001
        print("@@ERR " + " ".join(str(exc).split()), flush=True)
        return 2
    print("@@MODEL 1 1", flush=True)

    X, kept, n_fail, last = bne.extract_embeddings_for_samples(
        samples, model=model, chunk_size=chunk,
        progress_callback=lambda c, t: print(f"@@PROG {c} {t}", flush=True))
    kept_paths = {s.filepath for s in kept}
    ok = np.array([p in kept_paths for p in paths], dtype=bool)
    np.savez(out_npz, X=X, ok=ok, n_fail=np.array(n_fail), last_error=np.array(last or ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
