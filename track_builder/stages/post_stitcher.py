# mot/stages/post_linker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment

from track_builder.kf.motion_model import rts_smooth_tracklet
from track_builder.utils import (
    UnionFind,
    canon_class,
    class_compatible,
    merge_params_postlink,
)


# -----------------------------------------------------------------------------
# Global configuration (requested)
# -----------------------------------------------------------------------------

POST_LINKER_CLASS_PARAMS_DEFAULT: Dict[str, Dict[str, Any]] = {
    "car": dict(
        dt_max_s=0.75,
        overlap_allow_s=0.75,
        dist_max_m=2.0,
        pair_dt_thr_s=0.3,  # within ~1-2 frames at 10 Hz
        angle_thr_deg=30.0,
        speed_min_for_dir_mps=1.0,
        meas_sigma_m=0.5,
        rts_sigma_a=3.0,
        k_boundary=3,
    ),
    "pedestrian": dict(
        dt_max_s=0.7,
        overlap_allow_s=0.75,
        dist_max_m=1.5,
        pair_dt_thr_s=0.3,
        angle_thr_deg=40.0,
        speed_min_for_dir_mps=0.6,
        meas_sigma_m=0.25,
        rts_sigma_a=3.0,
        k_boundary=3,
    ),
    "ambiguous": dict(
        dt_max_s=0.75,
        overlap_allow_s=0.75,
        dist_max_m=1.5,
        pair_dt_thr_s=0.3,
        angle_thr_deg=35.0,
        speed_min_for_dir_mps=0.8,
        meas_sigma_m=0.35,
        rts_sigma_a=3.0,
        k_boundary=3,
    ),
}


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _robust_window_velocity(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    first: bool,
    k: int = 6,
    min_dt: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(t)
    if n < 2:
        v = np.array([0.0, 0.0], float)
        return v, 0.0

    k = min(int(k), n)
    if first:
        tt, xx, yy = t[:k], x[:k], y[:k]
    else:
        tt, xx, yy = t[-k:], x[-k:], y[-k:]

    dt = float(tt[-1] - tt[0])
    if dt < min_dt:
        # fallback to local step
        if first:
            dt2 = float(t[1] - t[0]) if n >= 2 else 0.0
            if dt2 < min_dt:
                v = np.array([0.0, 0.0], float)
            else:
                v = np.array([(x[1] - x[0]) / dt2, (y[1] - y[0]) / dt2], float)
        else:
            dt2 = float(t[-1] - t[-2]) if n >= 2 else 0.0
            if dt2 < min_dt:
                v = np.array([0.0, 0.0], float)
            else:
                v = np.array([(x[-1] - x[-2]) / dt2, (y[-1] - y[-2]) / dt2], float)
        return v, float(np.hypot(v[0], v[1]))

    t0 = float(tt.mean())
    denom = float(np.sum((tt - t0) ** 2) + 1e-12)
    vx = float(np.sum((tt - t0) * (xx - xx.mean())) / denom)
    vy = float(np.sum((tt - t0) * (yy - yy.mean())) / denom)
    s = float(np.hypot(vx, vy))
    return np.array([vx, vy], float), s


def _angle_between(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> float:
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < eps or nv < eps:
        return 0.0
    c = float(np.clip(float(np.dot(u, v)) / (nu * nv), -1.0, 1.0))
    return float(np.arccos(c))


def _min_boundary_distance(
    tA: np.ndarray,
    xA: np.ndarray,
    yA: np.ndarray,
    tB: np.ndarray,
    xB: np.ndarray,
    yB: np.ndarray,
    *,
    pair_dt_thr: float,
) -> Tuple[float, float, float]:
    """
    Minimum measured-to-measured distance between last window of A and first window of B,
    constrained by |tB - tA| <= pair_dt_thr.
    Returns (min_dist, ta, tb). If none, returns (inf, nan, nan).
    """
    tA = np.asarray(tA, float)
    xA = np.asarray(xA, float)
    yA = np.asarray(yA, float)
    tB = np.asarray(tB, float)
    xB = np.asarray(xB, float)
    yB = np.asarray(yB, float)

    best = np.inf
    best_ta = np.nan
    best_tb = np.nan

    thr = float(pair_dt_thr)
    for i in range(len(tA)):
        for j in range(len(tB)):
            if abs(float(tB[j] - tA[i])) <= thr:
                dx = float(xB[j] - xA[i])
                dy = float(yB[j] - yA[i])
                d = float(np.hypot(dx, dy))
                if d < best:
                    best = d
                    best_ta = float(tA[i])
                    best_tb = float(tB[j])

    return best, best_ta, best_tb


@dataclass
class TrSummary:
    tid: int
    cls: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    idx: np.ndarray
    t0: float
    t1: float
    v_start: np.ndarray
    s_start: float
    v_end: np.ndarray
    s_end: float


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------


def final_post_stitch_linker(
    df: pd.DataFrame,
    class_params: Dict[str, Dict[str, Any]] = POST_LINKER_CLASS_PARAMS_DEFAULT,
    id_col: str = "stitched_id",
    class_col: str = "predicted_class",
    time_col: str = "timestamp",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    write_rts_cols: bool = True,
    require_margin: bool = True,
    margin_ratio: float = 1.4,
    time_eps: float = 1e-3,
    min_track_length_m: float = 5.0,
    min_track_displacement_m: float = 5.0,
) -> pd.DataFrame:
    """
    Fourth-pass linker (no prediction):
      - considers only end->start links (A ends, B starts)
      - allows small overlap: dt = t0_B - t1_A can be negative
      - NO prediction: uses min measured distance between boundary windows with time pairing
      - direction check uses robust velocities over boundary windows
      - resolves conflicts via Hungarian + optional margin
      - merges by union-find and re-RTS smooths merged groups on raw centroids

    NOTE: This operates on `id_col` (default 'tracklet_id'). If you want it to operate
    on 'stitched_id', pass id_col='stitched_id'.
    """

    out = df.copy()

    if id_col not in out.columns:
        raise ValueError(f"Missing id_col='{id_col}'")
    for c in (time_col, x_col, y_col):
        if c not in out.columns:
            raise ValueError(f"Missing required column '{c}'")

    ids = out[id_col].to_numpy()
    tids = np.unique(ids[ids >= 0])
    if len(tids) < 2:
        return out

    # build summaries
    summaries: List[TrSummary] = []
    for tid in tids:
        g = out[out[id_col] == tid].sort_values(time_col)
        t = g[time_col].to_numpy(float)
        x = g[x_col].to_numpy(float)
        y = g[y_col].to_numpy(float)
        idx = g.index.to_numpy()

        if len(t) < 2:
            continue

        cls_raw = None
        if class_col in g.columns and g[class_col].notna().any():
            cls_raw = g[class_col].mode().iat[0]
        cls = canon_class(cls_raw, class_params, default="ambiguous")

        k = int(class_params.get(cls, class_params["ambiguous"])["k_boundary"])
        v_start, s_start = _robust_window_velocity(t, x, y, first=True, k=k)
        v_end, s_end = _robust_window_velocity(t, x, y, first=False, k=k)

        summaries.append(
            TrSummary(
                tid=int(tid),
                cls=str(cls),
                t=t,
                x=x,
                y=y,
                idx=idx,
                t0=float(t[0]),
                t1=float(t[-1]),
                v_start=v_start,
                s_start=float(s_start),
                v_end=v_end,
                s_end=float(s_end),
            )
        )

    if len(summaries) < 2:
        return out

    ends = sorted(summaries, key=lambda s: s.t1)
    starts = sorted(summaries, key=lambda s: s.t0)

    E = len(ends)
    S = len(starts)
    BIG = 1e9

    cost = np.full((E, S), BIG, float)
    valid = np.zeros((E, S), bool)

    # candidate costs
    for i, A in enumerate(ends):
        for j, B in enumerate(starts):
            if A.tid == B.tid:
                continue
            if not class_compatible(A.cls, B.cls, ambiguous="ambiguous"):
                continue

            P = merge_params_postlink(A.cls, B.cls, class_params)

            dt = float(B.t0 - A.t1)
            if dt > float(P["dt_max"]) or dt < -float(P["overlap_allow"]):
                continue

            k = int(P["k_boundary"])
            tA = A.t[-k:]
            xA = A.x[-k:]
            yA = A.y[-k:]
            tB = B.t[:k]
            xB = B.x[:k]
            yB = B.y[:k]

            dmin, _, _ = _min_boundary_distance(
                tA,
                xA,
                yA,
                tB,
                xB,
                yB,
                pair_dt_thr=float(P["pair_dt_thr"]),
            )
            if not np.isfinite(dmin) or float(dmin) > float(P["dist_max"]):
                continue

            # direction gate only if both moving enough
            speed_min = float(P["speed_min_for_dir"])
            enforce_dir = (A.s_end >= speed_min) and (B.s_start >= speed_min)
            dir_pen = 0.0
            if enforce_dir:
                ang = _angle_between(A.v_end, B.v_start)
                if ang > np.deg2rad(float(P["angle_thr_deg"])):
                    continue
                dir_pen = float(ang / (np.deg2rad(float(P["angle_thr_deg"])) + 1e-9))

            cost[i, j] = float(dmin + 0.10 * abs(dt) + 0.25 * dir_pen)
            valid[i, j] = True

    if not np.any(valid):
        return out

    cost2 = cost.copy()
    cost2[~valid] = BIG
    row_ind, col_ind = linear_sum_assignment(cost2)

    accepted: List[Tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        if not valid[r, c] or float(cost2[r, c]) >= BIG:
            continue

        if require_margin:
            row_vals = np.sort(cost2[r, :])
            best = float(row_vals[0])
            second = float(row_vals[1]) if len(row_vals) > 1 else BIG

            col_vals = np.sort(cost2[:, c])
            bestc = float(col_vals[0])
            secondc = float(col_vals[1]) if len(col_vals) > 1 else BIG

            ok_row = best < (second / margin_ratio)
            ok_col = bestc < (secondc / margin_ratio)
            if not (ok_row and ok_col):
                continue

        accepted.append((ends[r].tid, starts[c].tid))

    if not accepted:
        return out

    # union merges
    uf = UnionFind([s.tid for s in summaries])
    for a, b in accepted:
        uf.union(a, b)

    comps = [c for c in uf.components() if len(c) > 1]
    if not comps:
        return out

    # ensure RTS cols exist if requested
    if write_rts_cols:
        for col in ("rts_x", "rts_y", "rts_vx", "rts_vy", "rts_speed"):
            if col not in out.columns:
                out[col] = np.nan

    # apply merges and re-smooth per component
    for comp in comps:
        rep = int(min(comp))

        spans = []
        has_entry = "entry_well" in out.columns
        has_exit = "exit_well" in out.columns

        if has_entry or has_exit:

            for tid in comp:
                g = out[out[id_col] == int(tid)]
                if len(g) == 0:
                    continue
                t0 = float(g[time_col].min())
                t1 = float(g[time_col].max())
                spans.append((int(tid), t0, t1))

            if spans:
                first_tid = min(spans, key=lambda a: a[1])[0]
                last_tid = max(spans, key=lambda a: a[2])[0]

                entry_val = (
                    out.loc[out[id_col] == first_tid, "entry_well"].iloc[0]
                    if has_entry
                    else None
                )
                exit_val = (
                    out.loc[out[id_col] == last_tid, "exit_well"].iloc[0]
                    if has_exit
                    else None
                )

        rows = out.index[out[id_col].isin(comp)].to_numpy()
        out.loc[rows, id_col] = rep

        if (has_entry or has_exit) and spans:
            if has_entry:
                out.loc[rows, "entry_well"] = entry_val
            if has_exit:
                out.loc[rows, "exit_well"] = exit_val

        if write_rts_cols:
            g = out.loc[rows].sort_values(time_col)
            t = g[time_col].to_numpy(float)
            z = g[[x_col, y_col]].to_numpy(float)

            cls_raw = None
            if class_col in g.columns and g[class_col].notna().any():
                cls_raw = g[class_col].mode().iat[0]
            cls_m = canon_class(cls_raw, class_params, default="ambiguous")

            meas_sigma = float(class_params[cls_m]["meas_sigma_m"])
            sigma_a = float(class_params[cls_m]["rts_sigma_a"])

            if len(t) >= 2:
                xs = rts_smooth_tracklet(
                    t,
                    z,
                    meas_noise_pos=meas_sigma,
                    sigma_a=sigma_a,
                    time_eps=time_eps,
                )
                out.loc[g.index, "rts_x"] = xs[:, 0]
                out.loc[g.index, "rts_y"] = xs[:, 1]
                out.loc[g.index, "rts_vx"] = xs[:, 2]
                out.loc[g.index, "rts_vy"] = xs[:, 3]
                out.loc[g.index, "rts_speed"] = np.hypot(xs[:, 2], xs[:, 3])

    min_len = float(min_track_length_m) if min_track_length_m is not None else 0.0
    if min_len > 0.0:
        keep_ids = []

        for tid, g in out.groupby(id_col):
            if tid < 0 or len(g) < 2:
                continue

            g = g.sort_values(time_col)

            # prefer RTS-smoothed if available
            if write_rts_cols and {"rts_x", "rts_y"}.issubset(g.columns):
                x = g["rts_x"].to_numpy(float)
                y = g["rts_y"].to_numpy(float)
            else:
                x = g[x_col].to_numpy(float)
                y = g[y_col].to_numpy(float)

            dist = np.sum(np.hypot(np.diff(x), np.diff(y)))
            if dist >= min_len:
                keep_ids.append(tid)

        out = out[out[id_col].isin(keep_ids)].copy()

    min_len = float(min_track_length_m) if min_track_length_m is not None else 0.0
    min_disp = (
        float(min_track_displacement_m) if min_track_displacement_m is not None else 0.0
    )

    if min_len > 0.0 or min_disp > 0.0:
        keep_ids = []

        for tid, g in out.groupby(id_col):
            if tid < 0 or len(g) < 2:
                continue

            g = g.sort_values(time_col)

            if write_rts_cols and {"rts_x", "rts_y"}.issubset(g.columns):
                x = g["rts_x"].to_numpy(float)
                y = g["rts_y"].to_numpy(float)
            else:
                x = g[x_col].to_numpy(float)
                y = g[y_col].to_numpy(float)

            # 1) total path length
            path_len = np.sum(np.hypot(np.diff(x), np.diff(y)))

            # 2) start → end displacement
            disp = np.hypot(x[-1] - x[0], y[-1] - y[0])

            if path_len >= min_len and disp >= min_disp:
                keep_ids.append(tid)

        out = out[out[id_col].isin(keep_ids)].copy()

    return out
