from src.detection import turkish_lookup as tl


def test_chart_label_turkish_and_fallback():
    tr = tl.get_turkish_name("Parus major")
    assert tr
    assert tl.chart_label("Parus major") == tr
    assert tl.chart_label("Parus major", with_latin=True) == f"{tr} (Parus major)"
    assert tl.chart_label("Xyz abc") == "Xyz abc"
