"""Spektrogram görünüm ayarları: kontrast, frekans aralığı, mel Hz etiketleri; ana pencerede şerit, seçim yansıması, PNG."""
import os

import numpy as np
import pytest

from src.visualization import spectrogram_view as sv
from src.visualization.spectrogram import compute_linear_spectrogram, compute_mel_spectrogram


def test_contrast_auto_keeps_old_behaviour():
    S = np.linspace(-80, 0, 1000).reshape(10, 100)
    assert sv.contrast_limits(S, 0) == (None, None)
    assert sv.contrast_limits(np.array([]), 40) == (None, None)


def test_contrast_ignores_a_single_loud_spike():
    S = np.full((50, 200), -60.0)
    S[0, 0] = 0.0                      # tek şiddetli ses (ör. mikrofon çarpması)
    vmin, vmax = sv.contrast_limits(S, 40)
    assert vmax == pytest.approx(-60.0) and vmin == pytest.approx(-100.0)   # tepe noktası kontrastı bozmaz


def test_freq_range_validation_and_limits():
    assert sv.clean_freq_range(0, 0, 16000) == (0.0, 16000)            # 0 = tümü
    assert sv.clean_freq_range(500, 99999, 16000) == (500.0, 16000)    # Nyquist'e kırpılır
    assert sv.clean_freq_range(8000, 4000, 16000) == (0.0, 16000)      # ters aralık -> tümü
    assert sv.y_limits("spec", 500, 8000, 16000) == (500.0, 8000.0)
    lo, hi = sv.y_limits("melspec", 0, 0, 16000, n_mels=128)
    assert lo == 0.0 and hi == pytest.approx(128.0)
    lo2, hi2 = sv.y_limits("melspec", 500, 8000, 16000, n_mels=128)
    assert 0 < lo2 < hi2 < 128                                          # mel ölçeği doğrusal değil, aralık içinde


def test_mel_ticks_are_labelled_in_hz_and_inside_range():
    pos, lab = sv.mel_ticks(500, 8000, 16000)
    assert lab[0] == "500 Hz" and lab[-1] == "8 kHz" and len(pos) == len(lab)
    assert pos == sorted(pos)


def test_mel_row_matches_computed_filterbank_axis():
    sr = 32000
    y = np.random.default_rng(0).standard_normal(sr) * 0.01
    S, ext = compute_mel_spectrogram(y, sr)
    assert ext[3] == 128 and sv.hz_to_mel_row(sr / 2, sr / 2, ext[3]) == pytest.approx(128.0)


# ------------------------------------------------------------------ ana pencere
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APP = []


@pytest.fixture(scope="module")
def win():
    _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    from src.ui.main_window import BirdSoundApp
    w = BirdSoundApp()
    sr = 32000
    y = (np.random.default_rng(1).standard_normal(sr * 3) * 0.05).astype(np.float32)
    w.y, w.sr, w.duration = y, sr, 3.0
    w.spec_cache = compute_linear_spectrogram(y, sr)
    w.melspec_cache = compute_mel_spectrogram(y, sr)
    return w


def _set_mode(w, mode):
    w.view_mode = mode
    w.plot_waveform()


def test_defaults_are_unchanged(win):
    assert win.cmb_cmap.currentData() == "magma" and win.spn_contrast.value() == 0
    assert win.spn_fmin.value() == 0 and win.spn_fmax.value() == 0 and not win.chk_wave_strip.isChecked()
    _set_mode(win, "spec")
    assert win.ax_wave is None
    assert win.ax.images[0].get_cmap().name == "magma"
    assert win.ax.get_ylim() == pytest.approx((0.0, win.sr / 2))


def test_cmap_contrast_and_frequency_range_apply_to_the_image(win):
    _set_mode(win, "spec")
    win.cmb_cmap.setCurrentIndex(1)
    win.spn_contrast.setValue(40)
    win.spn_fmin.setValue(500)
    win.spn_fmax.setValue(8000)
    im = win.ax.images[0]
    assert im.get_cmap().name == "gray_r"
    vmin, vmax = im.get_clim()
    assert vmax - vmin == pytest.approx(40.0)
    assert win.ax.get_ylim() == pytest.approx((500.0, 8000.0))
    # ayarlar spektrogram verisini değiştirmez
    assert win.spec_cache[0].shape == compute_linear_spectrogram(win.y, win.sr)[0].shape
    _set_mode(win, "melspec")
    labels = [t.get_text() for t in win.ax.get_yticklabels()]
    assert "1 kHz" in labels and win.ax.get_ylabel() == "Frekans (mel ölçeğinde)"
    win.btn_view_reset.click()
    assert win.cmb_cmap.currentData() == "magma" and win.spn_contrast.value() == 0 and win.spn_fmax.value() == 0


def test_wave_strip_on_top_shares_time_axis_and_mirrors_selection(win):
    _set_mode(win, "spec")
    win.chk_wave_strip.setChecked(True)
    assert win.ax_wave is not None and win.ax_wave is not win.ax
    assert win.ax_wave.get_position().y0 > win.ax.get_position().y0       # şerit üstte
    win.selection_start, win.selection_end = 0.5, 1.5
    win.plot_waveform()
    assert win._wave_sel is not None and win._wave_sel.get_visible()
    assert win._wave_sel.get_x() == pytest.approx(0.5) and win._wave_sel.get_width() == pytest.approx(1.0)
    win.update_position(1200)
    assert list(win._wave_pos.get_xdata()) == [1.2, 1.2]
    win.ax.set_xlim(0.5, 2.0)
    assert win.ax_wave.get_xlim() == pytest.approx((0.5, 2.0))            # ortak zaman ekseni
    win.chk_wave_strip.setChecked(False)
    assert win.ax_wave is None and win._wave_sel is None
    assert len(win.figure.axes) == 1


def test_wave_strip_only_in_spectrogram_modes(win):
    win.chk_wave_strip.setChecked(True)
    _set_mode(win, "waveform")
    assert win.ax_wave is None and len(win.figure.axes) == 1
    _set_mode(win, "melspec")
    assert win.ax_wave is not None
    win.chk_wave_strip.setChecked(False)


def test_png_save_is_300dpi_and_hides_selection_and_position_line(win, tmp_path, monkeypatch):
    from src.ui import main_window as mw
    _set_mode(win, "spec")
    win.selection_start, win.selection_end = 0.5, 1.5
    win.plot_waveform()
    out = tmp_path / "g.png"
    monkeypatch.setattr(mw.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    monkeypatch.setattr(mw.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    saved = {}
    orig = win.figure.savefig

    def spy(path, **kw):
        saved["dpi"] = kw.get("dpi")
        saved["sel_visible"] = win.selection_rect.get_visible()
        saved["pos_visible"] = win.position_line.get_visible()
        return orig(path, **kw)

    monkeypatch.setattr(win.figure, "savefig", spy)
    win.save_view_png()
    assert out.exists() and out.stat().st_size > 2000
    assert saved == {"dpi": 300, "sel_visible": False, "pos_visible": False}
    assert win.selection_rect.get_visible() and win.position_line.get_visible()   # sonra geri gelir


def test_display_controls_are_enabled_only_for_spectrogram_views(win):
    win.cmb_view.setCurrentIndex(0)
    assert not win.cmb_cmap.isEnabled() and not win.chk_wave_strip.isEnabled()
    win.cmb_view.setCurrentIndex(1)
    assert win.cmb_cmap.isEnabled() and win.spn_contrast.isEnabled() and win.spn_fmax.isEnabled()
    win.cmb_view.setCurrentIndex(0)
