from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence, Union

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# =============================================================================
# Raw log parsing (tracking topic -> flat detections DataFrame)
# =============================================================================


@dataclass(frozen=True)
class RawLogConfig:
    """
    Config for reading + parsing your raw CSV logs.

    The raw CSV is assumed to be row-wise messages with fields like:
      topic, payload, ..., timestamp, time_delta_prev

    You can adapt names / suffix / JSON paths per scene or dataset.
    """

    # CSV reading
    column_names: Sequence[str] = (
        "topic",
        "payload",
        "zero",
        "boolean",
        "timestamp",
        "time_delta_prev",
    )
    has_header: bool = False

    # Topic filtering
    topic_suffix: str = "tracking"  # keep rows where topic endswith this

    # JSON parsing
    objects_key: str = "objects"  # payload JSON contains a list under this key

    # Output / cleanup
    sort_by_timestamp: bool = True
    add_time_index: bool = True  # group timestamps and assign an integer frame id

    # Drop noisy columns by substring (as in your notebook)
    drop_substrings: Sequence[str] = (
        "bounding_box",
        "collision",
        "acceleration",
        "mass",
        "previous",
        "is_active",
    )


def read_raw_log_csv(
    path: Union[str, Path],
    *,
    cfg: RawLogConfig = RawLogConfig(),
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """
    Read your raw tracking log CSV into a DataFrame with standard column names.
    """
    path = Path(path)
    names = list(cfg.column_names)

    df = pd.read_csv(
        path,
        names=names if not cfg.has_header else None,
        header=None if not cfg.has_header else "infer",
        **read_csv_kwargs,
    )
    return df


def parse_tracking_payload(
    payload: Any,
    *,
    objects_key: str = "objects",
) -> List[Dict[str, Any]]:
    """
    Parse a JSON payload and return the list of objects.
    Returns [] on parse errors or if key is missing/empty.
    """
    if payload is None or (isinstance(payload, float) and np.isnan(payload)):
        return []

    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, json.JSONDecodeError):
        return []

    objs = data.get(objects_key, [])
    if not isinstance(objs, list):
        return []
    # ensure dict-like objects
    out: List[Dict[str, Any]] = []
    for o in objs:
        if isinstance(o, dict):
            out.append(o)
    return out


def parse_raw_tracking_df(
    raw_data: pd.DataFrame,
    *,
    cfg: RawLogConfig = RawLogConfig(),
) -> pd.DataFrame:
    """
    Convert raw message log DataFrame into flat per-detection rows.

    Output includes:
      - timestamp
      - time_delta_prev (if present)
      - flattened payload fields (json_normalize)
      - time_index (optional)

    Notes:
      - This does NOT rename fields like centroid_x/y; your payload keys dictate that.
      - We replace '.' with '_' to keep column access easy.
    """
    if "topic" not in raw_data.columns:
        raise ValueError("raw_data must contain column 'topic'")
    if "payload" not in raw_data.columns:
        raise ValueError("raw_data must contain column 'payload'")
    if "timestamp" not in raw_data.columns:
        raise ValueError("raw_data must contain column 'timestamp'")

    df = raw_data.dropna(subset=["topic"]).copy()

    # Filter to tracking topics
    topic = df["topic"].astype(str)
    tracking_mask = topic.str.endswith(str(cfg.topic_suffix))
    tracking_df = df.loc[tracking_mask].copy()
    if tracking_df.empty:
        return pd.DataFrame()

    tracking_df["parsed_payload"] = tracking_df["payload"].apply(
        lambda p: parse_tracking_payload(p, objects_key=str(cfg.objects_key))
    )
    tracking_df = tracking_df[tracking_df["parsed_payload"].map(len) > 0].copy()
    if tracking_df.empty:
        return pd.DataFrame()

    # Explode objects list -> one row per object
    tracking_df = tracking_df.explode("parsed_payload", ignore_index=True)

    payload_df = pd.json_normalize(tracking_df["parsed_payload"])
    base_cols = ["timestamp"]
    if "time_delta_prev" in tracking_df.columns:
        base_cols.append("time_delta_prev")

    final_df = pd.concat(
        [
            tracking_df[base_cols].reset_index(drop=True),
            payload_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Drop noisy columns by substring
    for sub in cfg.drop_substrings:
        cols = list(final_df.filter(like=sub).columns)
        if cols:
            final_df = final_df.drop(columns=cols)

    # Replace dots in column names
    final_df.rename(lambda x: str(x).replace(".", "_"), axis=1, inplace=True)

    if cfg.sort_by_timestamp:
        final_df.sort_values(by="timestamp", inplace=True)

    if cfg.add_time_index:
        final_df["time_index"] = final_df.groupby("timestamp").ngroup()

    final_df = final_df.reset_index(drop=True)
    return final_df


# ============================================================================
# Time utilities
# ============================================================================


def estimate_period_seconds(
    ts_diffs: Union[np.ndarray, List[float]],
    q_low: float = 0.1,
    q_high: float = 0.9,
) -> float:
    d = np.asarray(ts_diffs, dtype=float).reshape(-1)

    # keep only valid positive deltas
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return 0.0

    lo, hi = np.quantile(d, [q_low, q_high])
    d = d[(d >= lo) & (d <= hi)]
    if d.size == 0:
        return 0.0

    return float(np.median(d))


def bucket_timestamps(
    ts: np.ndarray,
    ts_diffs: np.ndarray,
    mode: str = "round",
    t0: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """
    Returns (t_bucketed, bucket_idx, period_seconds).
    """
    T = estimate_period_seconds(ts_diffs)
    ts = np.asarray(ts, dtype=float).reshape(-1)

    if T <= 0.0:
        return ts.astype(float), None, 0.0

    t0_used = float(ts.min() if t0 is None else t0)
    x = (ts - t0_used) / T

    if mode == "round":
        idx = np.rint(x).astype(np.int64)
    elif mode == "floor":
        idx = np.floor(x).astype(np.int64)
    else:
        raise ValueError("bucket mode must be 'round' or 'floor'")

    t_bucketed = t0_used + idx.astype(float) * T
    return t_bucketed, idx, T


# ============================================================================
# Union-Find
# ============================================================================


class UnionFind:
    """
    Simple union-find (disjoint set) structure.
    """

    def __init__(self, items: Iterable[int]):
        items = [int(i) for i in items]
        self.parent = {i: i for i in items}
        self.rank = {i: 0 for i in items}

    def find(self, a: int) -> int:
        a = int(a)
        p = self.parent[a]
        if p != a:
            self.parent[a] = self.find(p)
        return self.parent[a]

    def union(self, a: int, b: int) -> None:
        a = int(a)
        b = int(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self) -> List[List[int]]:
        comp: dict[int, List[int]] = {}
        for a in self.parent:
            r = self.find(a)
            comp.setdefault(r, []).append(a)
        return list(comp.values())


# ============================================================================
# Vector / geometry helpers (used by stitching + geofences)
# ============================================================================


def safe_unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=float)
    return v / n


def cos_between(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> float:
    uu = safe_unit(u, eps=eps)
    vv = safe_unit(v, eps=eps)
    return float(np.dot(uu, vv))


def polyline_length(xs: np.ndarray, ys: np.ndarray) -> float:
    xs = np.asarray(xs, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    if xs.size < 2:
        return 0.0
    dx = np.diff(xs)
    dy = np.diff(ys)
    return float(np.sum(np.hypot(dx, dy)))


def point_in_poly(x: float, y: float, poly_xy: np.ndarray) -> bool:
    """
    Ray casting point-in-polygon. poly_xy can be closed or open.
    """
    poly = np.asarray(poly_xy, dtype=float)
    if poly.shape[0] < 3:
        return False

    inside = False
    x0, y0 = float(poly[-1, 0]), float(poly[-1, 1])
    for i in range(poly.shape[0]):
        x1, y1 = float(poly[i, 0]), float(poly[i, 1])
        cond = ((y1 > y) != (y0 > y)) and (
            x < (x0 - x1) * (y - y1) / (y0 - y1 + 1e-12) + x1
        )
        if cond:
            inside = not inside
        x0, y0 = x1, y1
    return inside


def build_geofence_maps(
    geofence_list: List[Dict],
    area_name: str = "general_area",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Parse list-of-dicts geofences format:
      [{"name": ..., "vertices_xy": [[x,y],...]}]

    Returns:
      area_poly, well_polys (dict of name -> poly) for all non-area polygons
    """
    polys = {
        g["name"]: np.asarray(g["vertices_xy"], dtype=float) for g in geofence_list
    }
    if area_name not in polys:
        raise ValueError(f"area_name='{area_name}' not found in geofences")
    area_poly = polys[area_name]
    well_polys = {name: poly for name, poly in polys.items() if name != area_name}
    return area_poly, well_polys


def which_well(xy: np.ndarray, well_polys: Dict[str, np.ndarray]) -> Optional[str]:
    """
    Returns the name of the first well polygon containing xy, or None.
    """
    if not well_polys:
        return None
    x, y = float(xy[0]), float(xy[1])
    for name, poly in well_polys.items():
        if point_in_poly(x, y, poly):
            return name
    return None


# ============================================================================
# Class mapping + compatibility + conservative pairwise merges (stitching)
# ============================================================================


def canon_class(value: Any, class_params: Dict, default: str = "ambiguous") -> str:
    """
    Map arbitrary predicted_class strings to keys in class_params, else default.

    Strategy:
      - if any class key (except default) is a substring -> return that key
      - synonyms: ped/person/human -> pedestrian, vehicle/car/truck -> car
      - otherwise default
    """
    if value is None:
        return default if default in class_params else list(class_params.keys())[0]

    s = str(value).lower()

    # explicit contains checks for known keys (except default)
    for k in class_params.keys():
        if k != default and k in s:
            return k

    # common synonyms
    if ("ped" in s) or ("person" in s) or ("human" in s):
        if "pedestrian" in class_params:
            return "pedestrian"
        return default if default in class_params else list(class_params.keys())[0]

    if ("car" in s) or ("truck" in s) or ("vehicle" in s):
        if "car" in class_params:
            return "car"
        # allow truck key if present but no car
        if "truck" in class_params:
            return "truck"
        return default if default in class_params else list(class_params.keys())[0]

    return default if default in class_params else list(class_params.keys())[0]


def class_compatible(a_cls: str, b_cls: str, ambiguous: str = "ambiguous") -> bool:
    """
    Ambiguous matches anything; otherwise must match exactly.
    """
    a = str(a_cls)
    b = str(b_cls)
    if a == ambiguous or b == ambiguous:
        return True
    return a == b


def merge_sigma(
    a_cls: str, b_cls: str, class_params: Dict, key: str = "meas_sigma_m"
) -> float:
    """
    Conservative: use the larger measurement sigma.
    """
    A = class_params[str(a_cls)]
    B = class_params[str(b_cls)]
    return float(max(A.get(key, 0.0), B.get(key, 0.0)))


def merge_max_gap(
    a_cls: str, b_cls: str, class_params: Dict, key: str = "max_gap_s"
) -> float:
    """
    Conservative: use the smaller max gap.
    """
    A = class_params[str(a_cls)]
    B = class_params[str(b_cls)]
    return float(min(A.get(key, np.inf), B.get(key, np.inf)))


def merge_max_turn(
    a_cls: str, b_cls: str, class_params: Dict, key: str = "max_turn_deg"
) -> float:
    """
    Conservative: use the smaller max turn.
    """
    A = class_params[str(a_cls)]
    B = class_params[str(b_cls)]
    return float(min(A.get(key, 180.0), B.get(key, 180.0)))


def merge_speed_ellipse_min(
    a_cls: str, b_cls: str, class_params: Dict, key: str = "speed_ellipse_min_mps"
) -> float:
    """
    Conservative: use the larger threshold (harder to use ellipse when uncertain).
    """
    A = class_params[str(a_cls)]
    B = class_params[str(b_cls)]
    return float(max(A.get(key, 0.0), B.get(key, 0.0)))


def merge_pair_params_stitching(a_cls: str, b_cls: str, class_params: Dict) -> Dict:
    """
    Convenience helper used by stitch_tracklets_hungarian to merge a pair's gating parameters
    conservatively.

    Expects per-class dict to include keys used by your stitcher:
      - max_gap_s
      - max_turn_deg
      - meas_sigma_m
      - speed_ellipse_min_mps
    """
    a = str(a_cls)
    b = str(b_cls)
    return dict(
        meas_sigma_m=merge_sigma(a, b, class_params, key="meas_sigma_m"),
        max_gap_s=merge_max_gap(a, b, class_params, key="max_gap_s"),
        max_turn_deg=merge_max_turn(a, b, class_params, key="max_turn_deg"),
        speed_ellipse_min_mps=merge_speed_ellipse_min(
            a, b, class_params, key="speed_ellipse_min_mps"
        ),
    )


def merge_params_postlink(a_cls: str, b_cls: str, class_params: Dict) -> Dict:
    """
    Conservative merge for the *final post-stitching linker*.

    Expects (per your last code) per-class keys:
      - dt_max_s
      - overlap_allow_s
      - dist_max_m
      - pair_dt_thr_s
      - angle_thr_deg
      - speed_min_for_dir_mps
      - meas_sigma_m
      - rts_sigma_a
      - k_boundary
    """
    A = class_params[str(a_cls)]
    B = class_params[str(b_cls)]
    return dict(
        dt_max=float(min(A["dt_max_s"], B["dt_max_s"])),
        overlap_allow=float(min(A["overlap_allow_s"], B["overlap_allow_s"])),
        dist_max=float(min(A["dist_max_m"], B["dist_max_m"])),
        pair_dt_thr=float(min(A["pair_dt_thr_s"], B["pair_dt_thr_s"])),
        angle_thr_deg=float(min(A["angle_thr_deg"], B["angle_thr_deg"])),
        speed_min_for_dir=float(
            max(A["speed_min_for_dir_mps"], B["speed_min_for_dir_mps"])
        ),
        meas_sigma_m=float(max(A["meas_sigma_m"], B["meas_sigma_m"])),
        rts_sigma_a=float(max(A["rts_sigma_a"], B["rts_sigma_a"])),
        k_boundary=int(min(A["k_boundary"], B["k_boundary"])),
    )


def plot_track_length_distributions(
    df: pd.DataFrame,
    run_dir: str | Path,
    id_col: str = "stitched_id",
    class_col: str = "predicted_class",
    ignore_id: int = -1,
    bin_size: int = 1,
) -> Path:
    """
    Plot per-class track length histograms and save them under run_dir/plots.

    Returns the plots directory path.
    """
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if class_col not in df.columns:
        raise ValueError(f"Column '{class_col}' not found in dataframe")

    classes = df[class_col].dropna().astype(str).unique().tolist()

    for cls in classes:
        cls_ids = df.loc[df[class_col].eq(cls), id_col].dropna().unique()

        lengths = df[df[id_col].isin(cls_ids)].groupby(id_col).size()

        if ignore_id is not None:
            lengths = lengths[lengths.index != ignore_id]

        if lengths.empty:
            continue

        x = lengths.to_numpy()

        fig = go.Figure(
            go.Histogram(
                x=x,
                xbins=dict(size=bin_size),
                histnorm="percent",
            )
        )

        fig.update_layout(
            title=f"Track length distribution – class: {cls}",
            xaxis_title=f"Rows per {id_col}",
            yaxis_title="Percentage of tracks",
            bargap=0.05,
        )

        out_path = plots_dir / f"track_length_{cls}.png"
        fig.write_image(out_path)

    return plots_dir


def plot_entry_exit_matrix(
    df: pd.DataFrame,
    run_dir: str | Path,
    entry_col: str = "entry_well",
    exit_col: str = "exit_well",
) -> Path:
    """
    Plot entry–exit well count matrix and save it under run_dir/plots.

    Returns the plots directory path.
    """
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if entry_col not in df.columns or exit_col not in df.columns:
        raise ValueError(f"Missing required columns: {entry_col}, {exit_col}")

    # Keep only rows where both entry and exit wells are present
    pairs = df.loc[
        df[entry_col].notna() & df[exit_col].notna(),
        [entry_col, exit_col],
    ]

    if pairs.empty:
        return plots_dir

    # Entry–exit count matrix
    matrix = pairs.groupby([entry_col, exit_col]).size().reset_index(name="count")

    matrix_pivot = matrix.pivot(
        index=entry_col, columns=exit_col, values="count"
    ).fillna(0)

    fig = px.imshow(
        matrix_pivot,
        labels=dict(
            x="Exit well",
            y="Entry well",
            color="Count",
        ),
        aspect="auto",
        color_continuous_scale="Viridis",
    )

    fig.update_layout(
        title="Entry–Exit Well Pair Matrix",
        xaxis_title="Exit well",
        yaxis_title="Entry well",
    )

    out_path = plots_dir / "entry_exit_well_matrix.png"
    fig.write_image(out_path)

    return plots_dir
