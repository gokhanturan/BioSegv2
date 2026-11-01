"""Testler için BirdNET gömme sahteleri (model indirilmez, ses okunmaz)."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import numpy as np


def adapt_per_file(fake_extract):
    """(classifier, filepath) -> gömme biçimindeki eski sahteyi, bne.extract_embeddings_for_samples
    imzasına uyarlar: başarısız dosyalar atlanır ve sayılır."""
    def _f(samples, model=None, progress_callback=None, stage_callback=None, **_kw):
        if stage_callback:
            stage_callback("model", 0, 1)
            stage_callback("model", 1, 1)
        X, kept, n_fail, last = [], [], 0, None
        for i, s in enumerate(samples):
            try:
                X.append(fake_extract(None, s.filepath))
                kept.append(s)
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                last = str(exc)
            if progress_callback:
                progress_callback(i + 1, len(samples))
        return (np.array(X) if X else np.zeros((0, 4))), kept, n_fail, last
    return _f


@contextmanager
def fake_birdnet(fake_extract):
    with ExitStack() as st:
        st.enter_context(patch("src.ml.pipeline.bne.extract_embeddings_for_samples",
                               side_effect=adapt_per_file(fake_extract)))
        yield
