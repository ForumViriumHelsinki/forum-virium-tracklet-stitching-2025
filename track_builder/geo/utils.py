from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeoRefConfig:
    """
    Scene-specific georeferencing parameters.

    sensor_lat/lon: WGS84 location of LiDAR origin.
    rotation_deg: rotation of LiDAR frame relative to ENU:
        0°  => x=east, y=north
        +CCW
    swap_xy: swap x/y before rotation (rare but useful)
    flip_x/flip_y: sign flips before rotation (for coordinate handedness fixes)
    """

    sensor_lat: float
    sensor_lon: float
    rotation_deg: float = 0.0
    swap_xy: bool = False
    flip_x: bool = False
    flip_y: bool = False

    earth_radius_m: float = (
        6378137.0  # WGS84 sphere approx (good enough for small areas)
    )


def lidar_xy_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    cfg: GeoRefConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert local LiDAR XY [m] to WGS84 lat/lon using local tangent plane approximation.

    Returns: (lat, lon) arrays
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if cfg.swap_xy:
        x, y = y, x
    if cfg.flip_x:
        x = -x
    if cfg.flip_y:
        y = -y

    theta = np.deg2rad(float(cfg.rotation_deg))
    c, s = float(np.cos(theta)), float(np.sin(theta))

    # Rotate in local ENU plane
    xr = x * c - y * s  # east
    yr = x * s + y * c  # north

    R = float(cfg.earth_radius_m)
    dlat = (yr / R) * (180.0 / np.pi)
    dlon = (xr / (R * np.cos(np.deg2rad(float(cfg.sensor_lat))))) * (180.0 / np.pi)

    lat = float(cfg.sensor_lat) + dlat
    lon = float(cfg.sensor_lon) + dlon
    return lat, lon


def add_lonlat_columns(
    df: pd.DataFrame,
    *,
    cfg: GeoRefConfig,
    x_col: str = "rts_x",
    y_col: str = "rts_y",
    fallback_x: str = "centroid_x",
    fallback_y: str = "centroid_y",
    out_lon_col: str = "lon",
    out_lat_col: str = "lat",
) -> pd.DataFrame:
    """
    Adds lon/lat columns in EPSG:4326.

    Uses (x_col,y_col) if present+non-null, else falls back to (fallback_x,fallback_y).
    """
    out = df.copy()

    use_primary = (
        (x_col in out.columns) and (y_col in out.columns) and out[x_col].notna().any()
    )
    xx = out[x_col].to_numpy(float) if use_primary else out[fallback_x].to_numpy(float)
    yy = out[y_col].to_numpy(float) if use_primary else out[fallback_y].to_numpy(float)

    lat, lon = lidar_xy_to_latlon(xx, yy, cfg)
    out[out_lon_col] = lon
    out[out_lat_col] = lat
    return out


def export_qgis_points_csv(
    df: pd.DataFrame,
    out_path: Path,
    *,
    cfg: GeoRefConfig,
    x_col: str = "rts_x",
    y_col: str = "rts_y",
    fallback_x: str = "centroid_x",
    fallback_y: str = "centroid_y",
    keep_cols: Optional[List[str]] = None,
) -> None:
    """
    Writes a QGIS-friendly points CSV with lon/lat in EPSG:4326.
    """
    out = add_lonlat_columns(
        df,
        cfg=cfg,
        x_col=x_col,
        y_col=y_col,
        fallback_x=fallback_x,
        fallback_y=fallback_y,
        out_lon_col="lon",
        out_lat_col="lat",
    )

    # Default columns to keep (edit freely)
    if keep_cols is None:
        keep_cols = [
            "timestamp",
            "lat",
            "lon",
            "tracklet_id",
            "stitched_id",
            "predicted_class",
            "rts_vx",
            "rts_vy",
            "rts_speed",
            "entry_well",
            "exit_well",
        ]

    cols = [c for c in keep_cols if c in out.columns]
    out = out[cols].copy()

    # Optional metadata column for humans
    out["crs"] = "EPSG:4326"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
