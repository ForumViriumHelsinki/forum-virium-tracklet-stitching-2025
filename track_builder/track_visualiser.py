#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

# Reuse the same georef utilities as your Timeline visualiser
from .geo.utils import GeoRefConfig, lidar_xy_to_latlon


# ---------------- Run-folder helpers (same pattern as your other script) ----------------
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
    p = run_dir / "output_tracks.csv"
    if p.exists():
        return p
    cands = sorted(run_dir.glob("*.csv"))
    for c in cands:
        if "qgis" not in c.name.lower():
            return c
    raise FileNotFoundError(f"No CSV found in run dir: {run_dir}")


def _infer_timestamp_unit(ts: pd.Series) -> str:
    """Heuristic: seconds ~1e9, ms ~1e12, ns ~1e18."""
    s = pd.to_numeric(ts, errors="coerce").dropna()
    if s.empty:
        return "s"
    med = float(s.median())
    if med > 1e16:
        return "ns"
    if med > 1e11:
        return "ms"
    return "s"


# ---------------- Core: CSV -> GeoDataFrame (WGS84 points) ----------------
def _points_wgs84_from_csv(
    df: pd.DataFrame,
    cfg: GeoRefConfig,
    x_col: str,
    y_col: str,
) -> gpd.GeoDataFrame:
    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()

    lat, lon = lidar_xy_to_latlon(x, y, cfg)
    geom = gpd.GeoSeries(
        [Point(float(lo), float(la)) for la, lo in zip(lat, lon)], crs="EPSG:4326"
    )

    gdf = gpd.GeoDataFrame(df.copy(), geometry=geom, crs="EPSG:4326")
    return gdf


# ---------------- Your original "tracks + broken + explore" logic ----------------
def _most_common(s: pd.Series):
    mode = s.mode()
    return mode.iloc[0] if len(mode) else None


def _make_line(points: pd.Series):
    pts = [(p.x, p.y) for p in points if p is not None]
    return LineString(pts) if len(pts) > 1 else None


def _build_tracks_explore_gdf(
    gdf_tracks: gpd.GeoDataFrame,
    *,
    id_col: str,
    t_col: str,
) -> gpd.GeoDataFrame:
    gdf_tracks = gdf_tracks.copy()
    gdf_tracks["center"] = gdf_tracks.geometry

    # timestamps
    unit = _infer_timestamp_unit(gdf_tracks[t_col])
    gdf_tracks["timestamp_dt"] = pd.to_datetime(
        gdf_tracks[t_col], unit=unit, errors="coerce"
    )

    # sort + aggregate
    gdf_tracks = gdf_tracks.sort_values([id_col, t_col])

    agg = {
        "start_timestamp": (t_col, "min"),
        "end_timestamp": (t_col, "max"),
        "start_timestamp_dt": ("timestamp_dt", "min"),
        "end_timestamp_dt": ("timestamp_dt", "max"),
        "point_count": ("center", "count"),
        "geometry": ("center", _make_line),
    }
    if "predicted_class" in gdf_tracks.columns:
        agg["predicted_class"] = ("predicted_class", _most_common)

    tracks = (
        gdf_tracks.groupby(id_col)
        .agg(**agg)
        .reset_index()
        .rename(columns={id_col: "id"})
    )

    tracks_ok = tracks[tracks.geometry.notna()].copy()
    broken = tracks[tracks.geometry.isna()].copy()

    if not broken.empty:
        broken = broken.merge(
            gdf_tracks[[id_col, "center"]].rename(columns={id_col: "id"}),
            on="id",
            how="left",
        )
        broken["geometry"] = broken["center"]
        broken = broken.drop(columns=["center"])

    tracks_ok["track_type"] = "TRACK"
    broken["track_type"] = "BROKEN_POINT"
    tracks_all = pd.concat([tracks_ok, broken], ignore_index=True)

    tracks_all = gpd.GeoDataFrame(tracks_all, geometry="geometry", crs=gdf_tracks.crs)

    # colors
    palette = ["red", "blue", "green", "orange", "purple"]
    tracks_all["color"] = [palette[i % len(palette)] for i in range(len(tracks_all))]

    return tracks_all


def track_visualizer(
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
    import numpy as np
    import folium

    run_dir = Path(run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(
            f"run_dir does not exist or is not a directory: {run_dir}"
        )

    csv_path = Path(csv) if csv else _default_output_csv(run_dir)
    out_html = Path(out) if out else (run_dir / "tracks.html")

    df = pd.read_csv(csv_path)

    # numeric + finite coords
    df[id_col] = pd.to_numeric(df.get(id_col), errors="coerce")
    df[t_col] = pd.to_numeric(df.get(t_col), errors="coerce")
    df[x_col] = pd.to_numeric(df.get(x_col), errors="coerce")
    df[y_col] = pd.to_numeric(df.get(y_col), errors="coerce")
    df = df[df[x_col].notna() & df[y_col].notna()].copy()

    if df.empty:
        raise ValueError("No valid rows to visualise after cleaning coords.")

    # Load georef
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

    # Convert to WGS84
    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    lat, lon = lidar_xy_to_latlon(x, y, cfg)

    msk = np.isfinite(lat) & np.isfinite(lon)
    df = df.loc[msk].copy()
    lat = lat[msk]
    lon = lon[msk]

    if df.empty:
        raise ValueError("All rows became invalid after lidar_xy_to_latlon conversion.")

    df["_lat"] = lat
    df["_lon"] = lon

    # sort for track drawing
    df = df.sort_values([id_col, t_col])

    # center map
    lat0 = float(np.mean(df["_lat"].to_numpy()))
    lon0 = float(np.mean(df["_lon"].to_numpy()))
    if not (np.isfinite(lat0) and np.isfinite(lon0)):
        lat0, lon0 = float(cfg.sensor_lat), float(cfg.sensor_lon)

    m = folium.Map(location=[lat0, lon0], zoom_start=int(zoom), control_scale=True)

    folium.CircleMarker(
        location=(float(cfg.sensor_lat), float(cfg.sensor_lon)),
        radius=12,
        color="red",
        fill=True,
        fill_opacity=0.9,
        tooltip="LiDAR sensor",
    ).add_to(m)

    palette = ["red", "blue", "green", "orange", "purple"]

    # ---- 1) draw real tracks (id >= 0) as thin polylines ----
    df_tracks = df[df[id_col].notna() & (df[id_col] >= 0)].copy()

    n_lines = 0
    n_single = 0
    for i, (tid, g) in enumerate(df_tracks.groupby(id_col, sort=False)):
        coords = list(
            zip(g["_lat"].astype(float), g["_lon"].astype(float))
        )  # (lat, lon)
        color = palette[i % len(palette)]
        if len(coords) >= 2:
            folium.PolyLine(
                coords,
                weight=2,
                opacity=0.8,
                color=color,
                tooltip=f"{id_col}={int(tid)}  n={len(coords)}",
            ).add_to(m)
            n_lines += 1
        elif len(coords) == 1:
            folium.CircleMarker(
                location=coords[0],
                radius=3,
                opacity=0.9,
                fill=True,
                fill_opacity=0.9,
                color=color,
                tooltip=f"{id_col}={int(tid)}  n=1",
            ).add_to(m)
            n_single += 1

    # ---- 2) draw unmatched (id == -1) as points ONLY ----
    df_unmatched = df[df[id_col] == -1].copy()
    n_unmatched = len(df_unmatched)
    for latv, lonv in zip(
        df_unmatched["_lat"].to_numpy(), df_unmatched["_lon"].to_numpy()
    ):
        folium.CircleMarker(
            location=(float(latv), float(lonv)),
            radius=2,
            opacity=0.6,
            fill=True,
            fill_opacity=0.4,
            color="gray",
        ).add_to(m)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    print(
        "Tracks (lines):",
        n_lines,
        "Single-point tracks:",
        n_single,
        "Unmatched points:",
        n_unmatched,
    )
    print("Saved →", out_html)
    return out_html


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run-folder → GeoPandas explore() HTML (tracks + broken points)"
    )

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

    track_visualizer(
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
