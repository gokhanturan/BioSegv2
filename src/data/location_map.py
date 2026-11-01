"""
src/data/location_map.py
================
`bird_sound_dataset.csv`'deki Lat/Lon sütunlarından kayıt noktalarını
gösteren bir harita/koordinat grafiği oluşturur.

TASARIM KARARI: Gerçek harita karoları (OpenStreetMap vb.) için yaygın olarak
kullanılan `contextily` kütüphanesi `rasterio` üzerinden GDAL'a bağımlıdır; bu,
Windows'ta DLL sorunlarına yatkın, ağır bir bağımlılık zinciri getirir. Bu
yüzden modül VARSAYILAN OLARAK yalnızca matplotlib (zaten zorunlu bir
bağımlılık) ile sade bir koordinat dağılım grafiği çizer: ek paket ve internet
bağlantısı gerekmez, her zaman çalışır.

`contextily` isteğe bağlı olarak kuruluysa (`pip install contextily`) VE internet
erişimi varsa, gerçek OpenStreetMap karoları arka plana otomatik eklenir. Bu
tamamen isteğe bağlı bir iyileştirmedir; kurulu/erişilebilir değilse sessizce
sade grafiğe geri dönülür.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class LocationPoint:
    lat: float
    lon: float
    label: str  # örn. "12 (Orman) — Burdur"
    record_count: int
    species_richness: int


def _to_float(value) -> Optional[float]:
    try:
        f = float(str(value).strip().replace(",", "."))
        if math.isnan(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def extract_location_points(df: pd.DataFrame, species_column: str = "Turkish_Name") -> list:
    """
    CSV'den geçerli (sayısal) Lat/Lon değerine sahip satırları çıkarıp,
    aynı koordinata (Habitat_ID bazında) sahip kayıtları gruplayarak her
    nokta için kayıt sayısı ve tür zenginliğini hesaplar.

    Geçersiz/boş Lat-Lon değerine sahip satırlar sessizce atlanır (hata
    fırlatılmaz) — kullanıcının henüz konum girmediği kayıtlar haritada
    görünmez, ama uygulama çökmez.
    """
    required = {"Lat", "Lon"}
    if not required.issubset(df.columns):
        return []

    sub = df.copy()
    sub["_lat"] = sub["Lat"].apply(_to_float)
    sub["_lon"] = sub["Lon"].apply(_to_float)
    sub = sub[sub["_lat"].notna() & sub["_lon"].notna()]
    if sub.empty:
        return []

    group_cols = [c for c in ["Habitat_ID", "Habitat_Type", "Location"] if c in sub.columns]
    if not group_cols:
        group_cols = ["_lat", "_lon"]

    points = []
    for keys, group in sub.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))

        lat = group["_lat"].mean()
        lon = group["_lon"].mean()

        parts = []
        if "Habitat_ID" in key_map and str(key_map["Habitat_ID"]).strip():
            habitat_part = str(key_map["Habitat_ID"]).strip()
            if "Habitat_Type" in key_map and str(key_map["Habitat_Type"]).strip():
                habitat_part += f" ({key_map['Habitat_Type']})"
            parts.append(habitat_part)
        if "Location" in key_map and str(key_map["Location"]).strip():
            parts.append(str(key_map["Location"]).strip())
        label = " — ".join(parts) if parts else f"{lat:.3f}, {lon:.3f}"

        if species_column in group.columns:
            species_richness = group[species_column].astype(str).str.strip()
            species_richness = species_richness[species_richness != ""].nunique()
        else:
            species_richness = 0

        points.append(LocationPoint(
            lat=lat, lon=lon, label=label,
            record_count=len(group), species_richness=int(species_richness),
        ))

    return points


def _lonlat_to_web_mercator(lon: float, lat: float):
    """Basit Web Mercator (EPSG:3857) dönüşümü — pyproj/geopandas gerekmez."""
    k = 6378137.0
    x = lon * (k * math.pi / 180.0)
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) * k
    return x, y


def is_contextily_available() -> bool:
    try:
        import contextily  # noqa: F401
        return True
    except ImportError:
        return False


def build_map_figure(points: list, use_basemap: bool = True):
    """Ortak tez grafik stiliyle harita figürü üretir (bkz. _build_map_figure_impl)."""
    from ..visualization.plot_style import style_context
    with style_context():
        return _build_map_figure_impl(points, use_basemap)


def _build_map_figure_impl(points: list, use_basemap: bool = True):
    """
    Verilen LocationPoint listesinden bir matplotlib Figure döndürür.
    Nokta boyutu kayıt sayısına, renk yoğunluğu tür zenginliğine göre
    ayarlanır. `contextily` kurulu ve erişilebilirse gerçek harita
    karoları arka plana eklenir; aksi halde (veya bir hata oluşursa)
    sessizce sade bir koordinat grafiğine düşülür.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7), dpi=100)
    ax = fig.add_subplot(111)

    if not points:
        ax.text(0.5, 0.5, "(Henüz geçerli Enlem/Boylam (Lat/Lon) değeri olan kayıt yok)",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        return fig

    basemap_added = False
    if use_basemap and is_contextily_available():
        try:
            import contextily as ctx

            xs, ys = zip(*[_lonlat_to_web_mercator(p.lon, p.lat) for p in points])
            sizes = [40 + 25 * p.record_count for p in points]
            colors = [p.species_richness for p in points]
            sca = ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis",
                              edgecolors="black", linewidths=0.8, zorder=5, alpha=0.85)
            for x, y, p in zip(xs, ys, points):
                ax.annotate(p.label, (x, y), fontsize=9, xytext=(6, 6),
                            textcoords="offset points", zorder=6)
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto")
            ax.set_axis_off()
            fig.colorbar(sca, ax=ax, label="Tür zenginliği (farklı tür sayısı)", shrink=0.7)
            basemap_added = True
        except Exception:  # noqa: BLE001
            # Ağ erişimi yok, karo indirilemedi, vb. — sessizce düz
            # koordinat grafiğine düş, kullanıcıyı hata ile rahatsız etme.
            ax.clear()
            basemap_added = False

    if not basemap_added:
        lats = [p.lat for p in points]
        lons = [p.lon for p in points]
        sizes = [40 + 25 * p.record_count for p in points]
        colors = [p.species_richness for p in points]
        sca = ax.scatter(lons, lats, s=sizes, c=colors, cmap="viridis",
                          edgecolors="black", linewidths=0.8, zorder=5, alpha=0.85)
        for lon, lat, p in zip(lons, lats, points):
            ax.annotate(p.label, (lon, lat), fontsize=9, xytext=(6, 6),
                        textcoords="offset points", zorder=6)
        ax.set_xlabel("Boylam (Lon)")
        ax.set_ylabel("Enlem (Lat)")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sca, ax=ax, label="Tür zenginliği (farklı tür sayısı)", shrink=0.7)
        if not is_contextily_available():
            ax.set_title(
                "Kayıt Noktaları (gerçek harita karoları için: pip install contextily)",
                fontsize=10,
            )
        else:
            ax.set_title("Kayıt Noktaları (harita karoları yüklenemedi — internet bağlantınızı kontrol edin)",
                          fontsize=9, color="#a04000")

    if basemap_added:
        ax.set_title("Kayıt Noktaları Haritası")

    fig.tight_layout()
    return fig


LOW_EFFORT_SEGMENTS = 5     # bunun altındaki noktaların tür sayısı tek başına yorumlanmamalı (çaba çok az)


def _richness_to_hex_color(ratio: float) -> str:
    """0-1 arası bir orana (düşük->yüksek tür zenginliği) basit bir
    mavi->kırmızı renk geçişi atar."""
    ratio = max(0.0, min(1.0, ratio))
    r = int(255 * ratio)
    b = int(255 * (1 - ratio))
    return f"#{r:02x}4d{b:02x}"


def is_folium_available() -> bool:
    try:
        import folium  # noqa: F401
        return True
    except ImportError:
        return False


# Altlık harita katmanları (Google'daki "Harita / Arazi / Uydu" seçeneğinin karşılığı).
# NOT: Google'ın kendi karoları, API anahtarı ve hizmet şartları olmadan doğrudan kullanılamaz;
# bu yüzden ücretsiz, atıfla kullanılabilen eşdeğerleri seçildi.
BASEMAP_LAYERS = (
    # (ad, karo adresi, atıf, en yüksek yerel yakınlaştırma)
    ("Harita", "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
     "© OpenStreetMap katkıcıları", 19),
    ("Arazi", "https://tile.opentopomap.org/{z}/{x}/{y}.png",
     "Harita: © OpenTopoMap (CC-BY-SA); veri: © OpenStreetMap katkıcıları, SRTM", 17),
    ("Uydu", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
     "Görüntü: Esri, Maxar, Earthstar Geographics ve GIS User Community", 19),
)
PLACE_NAMES_OVERLAY = (
    "Yer adları (uydu üzerinde)",
    "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    "Etiketler: Esri", 19)


def _add_basemaps(m, folium) -> None:
    """Haritaya Harita/Arazi/Uydu altlıklarını ve sağ üstte katman seçiciyi ekler (ilk katman: Harita)."""
    for i, (name, url, attr, native) in enumerate(BASEMAP_LAYERS):
        folium.TileLayer(tiles=url, attr=attr, name=name, overlay=False, control=True,
                         show=(i == 0), max_native_zoom=native, max_zoom=19).add_to(m)
    name, url, attr, native = PLACE_NAMES_OVERLAY
    folium.TileLayer(tiles=url, attr=attr, name=name, overlay=True, control=True,
                     show=False, max_native_zoom=native, max_zoom=19).add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)


def build_interactive_map_html(points: list) -> str:
    """
    `folium` (Leaflet.js tabanlı) kullanarak GERÇEK, tüm dünyayı kapsayan,
    fare ile serbestçe gezinilebilen (kaydırma/yakınlaştırma karoları
    OTOMATİK ve dinamik olarak yüklenir) bir HTML harita üretir.

    Pencere ilk açıldığında SADECE verilen noktaları (örn. Burdur/Isparta/
    Antalya) kapsayacak şekilde otomatik çerçevelenir (`fit_bounds`); ama
    kullanıcı uzaklaştırdıkça dünyanın geri kalanını da (OpenStreetMap
    karoları dinamik olarak indirilerek) görebilir — statik bir resmin
    aksine, bu KESİNTİSİZ, gerçek bir dünya haritasıdır.

    NOT: Bu fonksiyon internet erişimi gerektirir (Leaflet kütüphanesi ve
    harita karoları CDN üzerinden yükleniyor) — bu, tarayıcı tabanlı her
    interaktif haritanın doğası gereği böyledir.
    """
    import folium

    if not points:
        m = folium.Map(location=[20, 10], zoom_start=2, tiles=None, max_zoom=19)
        _add_basemaps(m, folium)
        return m.get_root().render()

    if len(points) == 1:
        m = folium.Map(location=[points[0].lat, points[0].lon], zoom_start=11, tiles=None, max_zoom=19)
    else:
        avg_lat = sum(p.lat for p in points) / len(points)
        avg_lon = sum(p.lon for p in points) / len(points)
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=9, tiles=None, max_zoom=19)

    _add_basemaps(m, folium)

    max_richness = max((p.species_richness for p in points), default=1) or 1

    # Büyük daireler küçüklerin üstünü örtmesin diye büyükten küçüğe çizilir.
    for p in sorted(points, key=lambda q: q.record_count, reverse=True):
        # Alan segment sayısıyla orantılı (karekök) ve üst sınırlı: çok segmentli
        # noktalar haritayı kaplamaz, az segmentli noktalar da görünür kalır.
        radius = min(6 + 4 * (p.record_count ** 0.5), 28)
        color = _richness_to_hex_color(p.species_richness / max_richness)
        popup_html = (
            f"<b>{p.label}</b><br>"
            f"Segment sayısı: {p.record_count}<br>"
            f"Tür zenginliği: {p.species_richness} farklı tür"
            + (f"<br><span style='color:#b45309'>⚠ {LOW_EFFORT_SEGMENTS}'ten az segment: tür sayısı güvenilir değil</span>"
               if p.record_count < LOW_EFFORT_SEGMENTS else "")
        )
        folium.CircleMarker(
            location=[p.lat, p.lon],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=p.label,
            color="#b45309" if p.record_count < LOW_EFFORT_SEGMENTS else "#333333",
            weight=2 if p.record_count < LOW_EFFORT_SEGMENTS else 1,
            dash_array="4" if p.record_count < LOW_EFFORT_SEGMENTS else None,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
        ).add_to(m)

    legend = (
        "<div style='position:fixed; bottom:24px; left:12px; z-index:9999; background:rgba(255,255,255,.93); "
        "padding:8px 12px; border:1px solid #999; border-radius:6px; font-size:12px; line-height:1.5; max-width:250px'>"
        "<b>Nasıl okunur</b><br>"
        "Daire boyutu: etiketli segment sayısı (çaba)<br>"
        f"Renk: belirlenen tür sayısı (<span style='color:#0000ff'>■</span> az → <span style='color:#ff0000'>■</span> çok)<br>"
        f"<span style='color:#b45309'>Kesikli turuncu kenar</span>: {LOW_EFFORT_SEGMENTS}'ten az segment<br>"
        "<i>Çaba farklı olduğundan tür sayıları noktalar arasında doğrudan karşılaştırılamaz.</i></div>")
    m.get_root().html.add_child(folium.Element(legend))

    if len(points) > 1:
        lats = [p.lat for p in points]
        lons = [p.lon for p in points]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(30, 30))

    return m.get_root().render()