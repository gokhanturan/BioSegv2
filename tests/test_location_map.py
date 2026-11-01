"""src.data.location_map için birim testleri."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data.location_map import (
    extract_location_points, build_map_figure, _to_float,
    _richness_to_hex_color, is_folium_available, build_interactive_map_html,
)


class TestLocationMap(unittest.TestCase):

    def test_to_float_valid(self):
        self.assertEqual(_to_float("37.7167"), 37.7167)
        self.assertEqual(_to_float("37,7167"), 37.7167)  # Türkçe ondalık virgül

    def test_to_float_invalid_returns_none(self):
        self.assertIsNone(_to_float(""))
        self.assertIsNone(_to_float(None))
        self.assertIsNone(_to_float("not_a_number"))

    def test_extract_points_skips_missing_coordinates(self):
        df = pd.DataFrame([
            {"Lat": "37.7167", "Lon": "30.1667", "Habitat_ID": "12", "Turkish_Name": "Karatavuk"},
            {"Lat": "", "Lon": "", "Habitat_ID": "99", "Turkish_Name": "Kayıtsız"},
        ])
        points = extract_location_points(df)
        self.assertEqual(len(points), 1)

    def test_extract_points_groups_by_habitat_and_counts_richness(self):
        df = pd.DataFrame([
            {"Lat": "37.7167", "Lon": "30.1667", "Habitat_ID": "12", "Habitat_Type": "Orman",
             "Location": "Burdur", "Turkish_Name": "Karatavuk"},
            {"Lat": "37.7167", "Lon": "30.1667", "Habitat_ID": "12", "Habitat_Type": "Orman",
             "Location": "Burdur", "Turkish_Name": "Karatavuk"},  # tekrar
            {"Lat": "37.7167", "Lon": "30.1667", "Habitat_ID": "12", "Habitat_Type": "Orman",
             "Location": "Burdur", "Turkish_Name": "Orman Toygarı"},
        ])
        points = extract_location_points(df)
        self.assertEqual(len(points), 1)
        p = points[0]
        self.assertEqual(p.record_count, 3)
        self.assertEqual(p.species_richness, 2)
        self.assertIn("12", p.label)
        self.assertIn("Orman", p.label)
        self.assertIn("Burdur", p.label)

    def test_extract_points_no_lat_lon_columns_returns_empty(self):
        df = pd.DataFrame([{"Turkish_Name": "Karatavuk"}])
        points = extract_location_points(df)
        self.assertEqual(points, [])

    def test_build_map_figure_empty_points_does_not_crash(self):
        fig = build_map_figure([], use_basemap=False)
        self.assertIsNotNone(fig)

    def test_build_map_figure_with_points_does_not_crash(self):
        df = pd.DataFrame([
            {"Lat": "37.7167", "Lon": "30.1667", "Habitat_ID": "12", "Turkish_Name": "Karatavuk"},
            {"Lat": "37.7648", "Lon": "30.5566", "Habitat_ID": "69", "Turkish_Name": "Anadolu Sıvacısı"},
        ])
        points = extract_location_points(df)
        fig = build_map_figure(points, use_basemap=False)
        self.assertIsNotNone(fig)

    def test_build_map_figure_gracefully_degrades_without_network(self):
        """use_basemap=True olsa bile (contextily kurulu olmasa da, ağ
        olmasa da) fonksiyon çökmemeli — sessizce düz grafiğe düşmeli."""
        df = pd.DataFrame([
            {"Lat": "37.7167", "Lon": "30.1667", "Habitat_ID": "12", "Turkish_Name": "Karatavuk"},
        ])
        points = extract_location_points(df)
        try:
            fig = build_map_figure(points, use_basemap=True)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"build_map_figure use_basemap=True ile çöktü: {exc}")
        self.assertIsNotNone(fig)

    def test_richness_to_hex_color_range(self):
        low = _richness_to_hex_color(0.0)
        high = _richness_to_hex_color(1.0)
        self.assertTrue(low.startswith("#"))
        self.assertTrue(high.startswith("#"))
        self.assertNotEqual(low, high)

    def test_richness_to_hex_color_clamps_out_of_range(self):
        # Sınır dışı değerler hata vermeden 0-1 aralığına kırpılmalı
        self.assertEqual(_richness_to_hex_color(-5), _richness_to_hex_color(0.0))
        self.assertEqual(_richness_to_hex_color(5), _richness_to_hex_color(1.0))

    def test_build_interactive_map_html_contains_expected_content(self):
        if not is_folium_available():
            self.skipTest("folium kurulu değil")
        df = pd.DataFrame([
            {"Lat": "37.7167", "Lon": "30.1667", "Habitat_ID": "12", "Habitat_Type": "Orman",
             "Location": "Burdur", "Turkish_Name": "Karatavuk"},
        ])
        points = extract_location_points(df)
        html = build_interactive_map_html(points)
        self.assertIn("Burdur", html)
        self.assertIn("Orman", html)
        self.assertIn("leaflet", html.lower())

    def test_build_interactive_map_html_empty_points_does_not_crash(self):
        if not is_folium_available():
            self.skipTest("folium kurulu değil")
        html = build_interactive_map_html([])
        self.assertIn("leaflet", html.lower())

    def test_build_interactive_map_html_single_point_does_not_crash(self):
        if not is_folium_available():
            self.skipTest("folium kurulu değil")
        df = pd.DataFrame([
            {"Lat": "37.7167", "Lon": "30.1667", "Habitat_ID": "12", "Turkish_Name": "Karatavuk"},
        ])
        points = extract_location_points(df)
        html = build_interactive_map_html(points)
        self.assertIn("leaflet", html.lower())

    def test_basemap_choices_map_terrain_satellite(self):
        """Harita / Arazi / Uydu katmanları ve katman seçici var; varsayılan 'Harita'; Google karosu yok."""
        if not is_folium_available():
            self.skipTest("folium kurulu değil")
        from src.data.location_map import BASEMAP_LAYERS
        self.assertEqual([b[0] for b in BASEMAP_LAYERS], ["Harita", "Arazi", "Uydu"])
        for pts in ([], extract_location_points(pd.DataFrame([
                {"Lat": "37.7", "Lon": "30.1", "Habitat_ID": "1", "Turkish_Name": "A"},
                {"Lat": "37.8", "Lon": "30.2", "Habitat_ID": "2", "Turkish_Name": "B"}]))):
            html = build_interactive_map_html(pts)
            self.assertIn("L.control.layers", html)
            self.assertIn("opentopomap", html)
            self.assertIn("World_Imagery", html)
            self.assertIn("tile.openstreetmap.org", html)
            self.assertNotIn("google", html.lower())


if __name__ == "__main__":
    unittest.main()