# mot/stages/pre_stitch.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from track_builder.kf.motion_model import rts_smooth_tracklet
from track_builder.utils import UnionFind


@dataclass
class TrackletRTS:
    tid: int
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    idx: np.ndarray

    @property
    def t0(self) -> float:
        return float(self.t[0])

    @property
    def t1(self) -> float:
        return float(self.t[-1])


def _slice_window(tr: TrackletRTS, t0: float, t1: float) -> Optional[TrackletRTS]:
    m = (tr.t >= t0) & (tr.t <= t1)
    if not np.any(m):
        return None
    return TrackletRTS(tr.tid, tr.t[m], tr.x[m], tr.y[m], tr.vx[m], tr.vy[m], tr.idx[m])


def _coverage_space_time(
    A: TrackletRTS, B: TrackletRTS, dt_thr: float, dist_thr: float
) -> float:
    orderA = np.argsort(A.t, kind="mergesort")
    orderB = np.argsort(B.t, kind="mergesort")
    tA = A.t[orderA]
    xA = A.x[orderA]
    yA = A.y[orderA]
    tB = B.t[orderB]
    xB = B.x[orderB]
    yB = B.y[orderB]

    usedB = np.zeros(len(tB), dtype=bool)
    j = 0
    hits = 0

    dt_thr = float(dt_thr)
    dist2 = float(dist_thr) ** 2

    for i in range(len(tA)):
        ta = float(tA[i])
        while j < len(tB) and tB[j] < ta - dt_thr:
            j += 1

        best_k = -1
        best_d2 = np.inf
        k = j
        while k < len(tB) and tB[k] <= ta + dt_thr:
            if not usedB[k]:
                dx = xB[k] - xA[i]
                dy = yB[k] - yA[i]
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_k = k
            k += 1

        if best_k >= 0 and best_d2 <= dist2:
            usedB[best_k] = True
            hits += 1

    return hits / max(1, len(tA))


def _slice_window_min_points(
    tr: TrackletRTS, t0: float, t1: float, min_points: int
) -> Optional[TrackletRTS]:
    t = tr.t
    n = len(t)
    if n == 0:
        return None

    in_win = (t >= t0) & (t <= t1)
    idx_in = np.flatnonzero(in_win)
    if len(idx_in) == 0:
        return None

    lo = int(idx_in[0])
    hi = int(idx_in[-1])
    count = hi - lo + 1

    take_left = True
    while count < int(min_points) and (lo > 0 or hi < n - 1):
        if take_left and lo > 0:
            lo -= 1
        elif (not take_left) and hi < n - 1:
            hi += 1
        else:
            if lo > 0:
                lo -= 1
            elif hi < n - 1:
                hi += 1
            else:
                break
        take_left = not take_left
        count = hi - lo + 1

    m = slice(lo, hi + 1)
    return TrackletRTS(tr.tid, tr.t[m], tr.x[m], tr.y[m], tr.vx[m], tr.vy[m], tr.idx[m])


def _window_vel_from_endpoints(tr: TrackletRTS) -> tuple[np.ndarray, float]:
    dt = float(tr.t[-1] - tr.t[0])
    if dt <= 1e-6:
        v = np.array([0.0, 0.0], dtype=float)
        return v, 0.0
    v = np.array([(tr.x[-1] - tr.x[0]) / dt, (tr.y[-1] - tr.y[0]) / dt], dtype=float)
    return v, float(np.hypot(v[0], v[1]))


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.hypot(u[0], u[1]))
    nv = float(np.hypot(v[0], v[1]))
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    c = float(np.clip((u @ v) / (nu * nv), -1.0, 1.0))
    return float(np.arccos(c))


def pre_stitch_tracklets(
    out: pd.DataFrame,
    dt_thr: float = 0.2,
    dist_thr: float = 0.30,
    prop_thr: float = 0.8,
    min_vel_points: int = 3,
    speed_thr: float = 2.0,
    angle_thr_deg: float = 25.0,
    min_overlap_points: int = 2,
    meas_noise_pos: float = 0.25,
    rts_sigma_a: float = 3.0,
    time_eps: float = 0.001,
) -> pd.DataFrame:
    """
    Merge overlapping tracklets (time overlap), then re-smooth merged groups.

    Input must contain:
      tracklet_id, timestamp, centroid_x, centroid_y, rts_x, rts_y, rts_vx, rts_vy
    """

    angle_thr = float(np.deg2rad(float(angle_thr_deg)))

    df = out.copy()

    required = {
        "tracklet_id",
        "timestamp",
        "centroid_x",
        "centroid_y",
        "rts_x",
        "rts_y",
        "rts_vx",
        "rts_vy",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Run builder with do_rts_smoothing=True."
        )

    valid = df["tracklet_id"].to_numpy()
    tids = np.unique(valid[valid >= 0])
    tracklets: list[TrackletRTS] = []

    for tid in tids:
        idx = df.index[df["tracklet_id"] == tid].to_numpy()
        t = df.loc[idx, "timestamp"].to_numpy(float)
        order = np.argsort(t, kind="mergesort")
        idx = idx[order]
        tr = TrackletRTS(
            tid=int(tid),
            t=df.loc[idx, "timestamp"].to_numpy(float),
            x=df.loc[idx, "rts_x"].to_numpy(float),
            y=df.loc[idx, "rts_y"].to_numpy(float),
            vx=df.loc[idx, "rts_vx"].to_numpy(float),
            vy=df.loc[idx, "rts_vy"].to_numpy(float),
            idx=idx,
        )
        if len(tr.t) >= 2:
            tracklets.append(tr)

    tracklets.sort(key=lambda tr: tr.t0)
    uf = UnionFind([tr.tid for tr in tracklets])

    active: list[TrackletRTS] = []
    for tr in tracklets:
        active = [a for a in active if a.t1 > tr.t0]

        for a in active:
            t0 = max(a.t0, tr.t0)
            t1 = min(a.t1, tr.t1)
            if t1 <= t0:
                continue

            Aov = _slice_window(a, t0, t1)
            Bov = _slice_window(tr, t0, t1)
            if Aov is None or Bov is None:
                continue
            if len(Aov.t) < min_overlap_points or len(Bov.t) < min_overlap_points:
                continue

            covA = _coverage_space_time(Aov, Bov, dt_thr=dt_thr, dist_thr=dist_thr)
            covB = _coverage_space_time(Bov, Aov, dt_thr=dt_thr, dist_thr=dist_thr)
            if min(covA, covB) < prop_thr:
                continue

            Avel = _slice_window_min_points(a, t0, t1, min_points=min_vel_points)
            Bvel = _slice_window_min_points(tr, t0, t1, min_points=min_vel_points)
            if Avel is None or Bvel is None:
                continue
            if len(Avel.t) < min_vel_points or len(Bvel.t) < min_vel_points:
                continue

            vA, sA = _window_vel_from_endpoints(Avel)
            vB, sB = _window_vel_from_endpoints(Bvel)

            if abs(sA - sB) > speed_thr:
                continue
            if _angle_between(vA, vB) > angle_thr:
                continue

            uf.union(a.tid, tr.tid)

        active.append(tr)

    comps = [c for c in uf.components() if len(c) > 1]
    if not comps:
        return df

    for comp in comps:
        rep = int(min(comp))
        rows = df.index[df["tracklet_id"].isin(comp)].to_numpy()

        t = df.loc[rows, "timestamp"].to_numpy(float)
        z = df.loc[rows, ["centroid_x", "centroid_y"]].to_numpy(float)

        order = np.argsort(t, kind="mergesort")
        rows = rows[order]
        t = t[order]
        z = z[order]

        # fuse near-same-time points within dt_thr
        fused_t = []
        fused_z = []
        start = 0
        n = len(t)
        while start < n:
            end = start + 1
            while end < n and (t[end] - t[start]) <= dt_thr:
                end += 1
            fused_t.append(float(np.mean(t[start:end])))
            fused_z.append(np.mean(z[start:end], axis=0))
            start = end

        fused_t = np.asarray(fused_t, float)
        fused_z = np.asarray(fused_z, float)

        if len(fused_t) >= 2:
            xs = rts_smooth_tracklet(
                fused_t,
                fused_z,
                meas_noise_pos=meas_noise_pos,
                sigma_a=rts_sigma_a,
                time_eps=time_eps,
            )

            nearest = np.searchsorted(fused_t, t, side="left")
            nearest = np.clip(nearest, 0, len(fused_t) - 1)
            prev = np.clip(nearest - 1, 0, len(fused_t) - 1)
            pick_prev = np.abs(fused_t[prev] - t) < np.abs(fused_t[nearest] - t)
            nearest[pick_prev] = prev[pick_prev]

            df.loc[rows, "rts_x"] = xs[nearest, 0]
            df.loc[rows, "rts_y"] = xs[nearest, 1]
            df.loc[rows, "rts_vx"] = xs[nearest, 2]
            df.loc[rows, "rts_vy"] = xs[nearest, 3]
            df.loc[rows, "rts_speed"] = np.hypot(xs[nearest, 2], xs[nearest, 3])

        df.loc[rows, "tracklet_id"] = rep

    return df
