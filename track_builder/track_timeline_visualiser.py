#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional
import html

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

import folium
from folium.plugins import Timeline, TimelineSlider
from folium.utilities import JsCode
from branca.element import MacroElement
from jinja2 import Template

# TODO: change this import to your actual module path
from .geo.utils import GeoRefConfig, lidar_xy_to_latlon


class MapTitle(MacroElement):
    def __init__(self, title: str, subtitle: str | None = None):
        super().__init__()
        self._template = Template(
            """
            {% macro script(this, kwargs) %}
            var titleDiv = L.control({position: 'topleft'});
            titleDiv.onAdd = function (map) {
                var div = L.DomUtil.create('div', 'map-title');
                div.innerHTML = `
                    <div style="
                        background: rgba(255,255,255,0.9);
                        padding: 8px 12px;
                        border-radius: 6px;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                        font-family: sans-serif;
                    ">
                        <div style="font-size:16px;font-weight:600;">
                            {{ title }}
                        </div>
                        {% if subtitle %}
                        <div style="font-size:12px;color:#555;">
                            {{ subtitle }}
                        </div>
                        {% endif %}
                    </div>
                `;
                return div;
            };
            titleDiv.addTo({{ this._parent.get_name() }});
            {% endmacro %}
            """
        )
        self.title = html.escape(title)
        self.subtitle = html.escape(subtitle) if subtitle else None


# ---------------- Core logic (unchanged) ----------------
def build_tracks(
    df: pd.DataFrame, id_col: str, t_col: str, x_col: str, y_col: str
) -> pd.DataFrame:
    df = df[df[id_col] >= 0].sort_values([id_col, t_col]).copy()
    if df.empty:
        raise ValueError("No assigned tracks (all ids < 0 or empty file).")

    df["t_dt"] = pd.to_datetime(df[t_col], unit="s", errors="coerce")
    df["pt"] = [
        Point(xy) for xy in zip(df[x_col].astype(float), df[y_col].astype(float))
    ]

    def mk_geom(pts):
        coords = [(p.x, p.y) for p in pts]
        return LineString(coords) if len(coords) > 1 else coords[0]  # Point if single

    agg = {
        "start": (t_col, "min"),
        "end": (t_col, "max"),
        "start_dt": ("t_dt", "min"),
        "end_dt": ("t_dt", "max"),
        "n": ("pt", "count"),
        "geom": ("pt", mk_geom),
    }
    if "predicted_class" in df.columns:
        agg["predicted_class"] = (
            "predicted_class",
            lambda s: s.mode().iloc[0] if len(s.mode()) else None,
        )

    tracks = df.groupby(id_col).agg(**agg).reset_index().rename(columns={id_col: "id"})
    palette = ["red", "blue", "green", "orange", "purple"]
    tracks["color"] = [palette[i % len(palette)] for i in range(len(tracks))]
    return tracks


def to_wgs84(tracks: pd.DataFrame, cfg: GeoRefConfig) -> pd.DataFrame:
    rows = []
    for _, r in tracks.iterrows():
        g = r["geom"]
        if g.geom_type == "LineString":
            xs, ys = zip(*g.coords)
            lat, lon = lidar_xy_to_latlon(np.array(xs), np.array(ys), cfg)
            new_g = LineString(
                list(zip(lon.tolist(), lat.tolist()))
            )  # GeoJSON: (lon, lat)
        else:  # Point
            lat, lon = lidar_xy_to_latlon(np.array([g.x]), np.array([g.y]), cfg)
            new_g = Point(float(lon[0]), float(lat[0]))

        rr = r.copy()
        rr["geom"] = new_g
        rows.append(rr)
    return pd.DataFrame(rows)


def map_center(tracks_wgs84: pd.DataFrame) -> tuple[float, float]:
    xs, ys = [], []
    for g in tracks_wgs84["geom"]:
        if g.geom_type == "Point":
            xs.append(g.x)
            ys.append(g.y)
        else:
            for x, y in g.coords:
                xs.append(x)
                ys.append(y)
    return float(np.mean(ys)), float(np.mean(xs))  # lat, lon


def save_timeline_html(
    tracks_wgs84: pd.DataFrame,
    out_html: str | Path,
    zoom: int,
    title: str,
    subtitle: str | None = None,
    sensor_lat: float | None = None,
    sensor_lon: float | None = None,
    sensor_radius: int = 12,
) -> None:
    feats = []
    for _, r in tracks_wgs84.iterrows():
        g = r["geom"]
        coords = (
            [[x, y] for x, y in g.coords] if g.geom_type == "LineString" else [g.x, g.y]
        )
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": g.geom_type, "coordinates": coords},
                "properties": {
                    "id": int(r["id"]),
                    "predicted_class": r.get("predicted_class", None),
                    "point_count": int(r["n"]),
                    "start": float(r["start"]) * 1000.0,
                    "end": float(r["end"]) * 1000.0,
                    "start_dt": (
                        r["start_dt"].isoformat() if pd.notna(r["start_dt"]) else None
                    ),
                    "end_dt": (
                        r["end_dt"].isoformat() if pd.notna(r["end_dt"]) else None
                    ),
                    "color": r["color"],
                },
            }
        )
    geojson = {"type": "FeatureCollection", "features": feats}

    style = JsCode(
        "function (f) { return { color: f.properties.color, weight: 3, opacity: 0.8 }; }"
    )

    lat0, lon0 = map_center(tracks_wgs84)
    m = folium.Map(location=[lat0, lon0], zoom_start=int(zoom))

    MapTitle(title=title, subtitle=subtitle).add_to(m)

    tl = Timeline(data=geojson, style=style).add_to(m)
    folium.GeoJsonTooltip(
        fields=["id", "predicted_class", "point_count", "start_dt", "end_dt"],
        sticky=True,
    ).add_to(tl)

    TimelineSlider(
        auto_play=False,
        show_ticks=True,
        enable_keyboard_controls=True,
        playback_duration=30000,
        time_interval="PT1S",
        date_options="YYYY-MM-DD HH:mm:ss",
    ).add_timelines(tl).add_to(m)

    if sensor_lat is not None and sensor_lon is not None:
        folium.CircleMarker(
            location=(float(sensor_lat), float(sensor_lon)),
            radius=int(sensor_radius),
            color="red",
            fill=True,
            fill_opacity=0.9,
            tooltip="LiDAR sensor",
        ).add_to(m)

    out_html = Path(out_html)
    m.save(str(out_html))
    print("Saved →", out_html)


# ---------------- New: run-folder glue ----------------
def _load_georef_from_run_config(run_dir: Path) -> Optional[GeoRefConfig]:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return None
    with cfg_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    georef = obj.get("georef_params")
    if not isinstance(georef, dict):
        return None
    return GeoRefConfig(**georef)


def _default_output_csv(run_dir: Path) -> Path:
    # Most common pipeline output
    p = run_dir / "output_tracks.csv"
    if p.exists():
        return p
    # Fallback: any csv in folder (excluding qgis points, if possible)
    cands = sorted(run_dir.glob("*.csv"))
    for c in cands:
        if "qgis" not in c.name.lower():
            return c
    raise FileNotFoundError(f"No CSV found in run dir: {run_dir}")


def track_timeline_visualizer(
    *,
    run_dir: str | Path,
    csv: str | Path | None = None,
    out: str | Path | None = None,
    id_col: str = "stitched_id",
    t_col: str = "timestamp",
    x_col: str = "rts_x",
    y_col: str = "rts_y",
    zoom: int = 18,
    sensor_lat: float | None = None,
    sensor_lon: float | None = None,
    rotation_deg: float | None = None,
    swap_xy: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
) -> Path:
    run_dir = Path(run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(
            f"run_dir does not exist or is not a directory: {run_dir}"
        )

    csv_path = Path(csv) if csv else _default_output_csv(run_dir)
    out_html = Path(out) if out else (run_dir / "tracklets_timeline.html")

    df = pd.read_csv(csv_path)

    cfg = _load_georef_from_run_config(run_dir)

    if cfg is None:
        if sensor_lat is None or sensor_lon is None or rotation_deg is None:
            raise ValueError("run_config.json missing and georef not fully specified.")
        cfg = GeoRefConfig(
            sensor_lat=float(sensor_lat),
            sensor_lon=float(sensor_lon),
            rotation_deg=float(rotation_deg),
            swap_xy=swap_xy,
            flip_x=flip_x,
            flip_y=flip_y,
        )
    else:
        if sensor_lat is not None:
            cfg.sensor_lat = float(sensor_lat)
        if sensor_lon is not None:
            cfg.sensor_lon = float(sensor_lon)
        if rotation_deg is not None:
            cfg.rotation_deg = float(rotation_deg)
        if swap_xy:
            cfg.swap_xy = True
        if flip_x:
            cfg.flip_x = True
        if flip_y:
            cfg.flip_y = True

    tracks = build_tracks(df, id_col, t_col, x_col, y_col)
    tracks_wgs84 = to_wgs84(tracks, cfg)

    save_timeline_html(
        tracks_wgs84,
        out_html,
        zoom,
        title=run_dir.name,
        subtitle=csv_path.name,
    )

    return out_html


def main() -> None:
    ap = argparse.ArgumentParser(description="Run-folder → Folium TimelineSlider HTML")

    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None)

    ap.add_argument("--id-col", default="stitched_id")
    ap.add_argument("--t-col", default="timestamp")
    ap.add_argument("--x-col", default="rts_x")
    ap.add_argument("--y-col", default="rts_y")
    ap.add_argument("--zoom", type=int, default=18)

    ap.add_argument("--sensor-lat", type=float, default=None)
    ap.add_argument("--sensor-lon", type=float, default=None)
    ap.add_argument("--rotation-deg", type=float, default=None)
    ap.add_argument("--swap-xy", action="store_true")
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--flip-y", action="store_true")

    a = ap.parse_args()

    track_timeline_visualizer(
        run_dir=a.run_dir,
        csv=a.csv,
        out=a.out,
        id_col=a.id_col,
        t_col=a.t_col,
        x_col=a.x_col,
        y_col=a.y_col,
        zoom=a.zoom,
        sensor_lat=a.sensor_lat,
        sensor_lon=a.sensor_lon,
        rotation_deg=a.rotation_deg,
        swap_xy=a.swap_xy,
        flip_x=a.flip_x,
        flip_y=a.flip_y,
    )


if __name__ == "__main__":
    main()
