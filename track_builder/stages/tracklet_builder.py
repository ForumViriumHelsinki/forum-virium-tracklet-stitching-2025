# mot/stages/tracklet_builder.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from track_builder.kf.motion_model import cv_predict, rts_smooth_tracklet
from track_builder.kf.tracks import Track
from track_builder.association.hungarian import solve_hungarian
from track_builder.utils import bucket_timestamps


def build_tracklets_hungarian(
    df: pd.DataFrame,
    v_max: float = 15.0,
    a_max: float = 5.0,
    max_gap_seconds: float = 0.25,
    min_tracklet_duration: float = 0.3,
    min_tracklet_points: int = 3,
    direction_weight: float = 1.5,
    min_speed_for_dir: float = 0.7,
    meas_noise_pos: float = 0.25,
    vel_gate_min_hits: int = 3,
    use_kinematic_gate: bool = True,
    use_mahalanobis_gate: bool = True,
    maha_gamma: float = 5.99,
    bucket_mode: str = "round",
    time_eps: float = 0.001,
    do_rts_smoothing: bool = True,
    rts_sigma_a: float = 3.0,
) -> pd.DataFrame:
    """
    Build short tracklets from per-frame detections using Hungarian assignment + KF gating.

    Required columns:
      - timestamp
      - centroid_x, centroid_y

    Output columns:
      - tracklet_id (int; -1 if unassigned)
      - optionally rts_* if do_rts_smoothing=True
    """

    if not {"timestamp", "centroid_x", "centroid_y"}.issubset(df.columns):
        raise ValueError("df must contain columns: timestamp, centroid_x, centroid_y")

    df_sorted = df.sort_values("timestamp").reset_index(drop=False)
    original_index = df_sorted["index"].to_numpy()

    t_raw = df_sorted["timestamp"].to_numpy(float)
    t_diffs = df_sorted["time_delta_prev"].to_numpy(float)

    t_bucketed, bucket_idx, period_seconds = bucket_timestamps(
        t_raw, t_diffs, mode=bucket_mode
    )
    df_sorted["_t_group_key"] = bucket_idx if bucket_idx is not None else t_raw
    df_sorted["_t_dyn"] = t_raw

    tracklet_ids_sorted = np.full(len(df_sorted), -1, dtype=int)

    grouped = df_sorted.groupby("_t_group_key").indices
    sorted_keys = sorted(grouped.keys())

    active_tracks: list[Track] = []
    next_internal_id = 0
    next_tracklet_id = 0

    # gating matrices
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
    R = np.eye(2, dtype=float) * (meas_noise_pos**2)

    def finalize_track(track: Track) -> None:
        nonlocal next_tracklet_id
        start_t = float(df_sorted.loc[track.row_indices[0], "_t_dyn"])
        end_t = float(track.last_meas_time)
        if track.confirmed and (end_t - start_t) >= min_tracklet_duration:
            tracklet_ids_sorted[track.row_indices] = next_tracklet_id
            next_tracklet_id += 1

    for key in sorted_keys:
        row_idxs = grouped[key]

        det_xy = df_sorted.loc[row_idxs, ["centroid_x", "centroid_y"]].to_numpy(
            dtype=float
        )
        det_t = df_sorted.loc[row_idxs, "_t_dyn"].to_numpy(dtype=float)

        # ensure increasing exact time within group
        order = np.argsort(det_t, kind="mergesort")
        row_idxs = np.asarray(row_idxs)[order]
        det_xy = det_xy[order]
        det_t = det_t[order]

        t_ref_min = float(det_t[0])
        t_ref_max = float(det_t[-1])

        # retire stale tracks
        still_active: list[Track] = []
        for tr in active_tracks:
            if (t_ref_min - float(tr.last_meas_time)) > max_gap_seconds:
                finalize_track(tr)
            else:
                still_active.append(tr)
        active_tracks = still_active

        T = len(active_tracks)
        D = len(det_xy)

        matches: list[Tuple[int, int]] = []
        unmatched_track = set(range(T))
        unmatched_det = set(range(D))

        if T > 0 and D > 0:
            BIG = 1e9
            cost = np.full((T, D), BIG, dtype=float)
            valid = np.ones((T, D), dtype=bool)

            for i, tr in enumerate(active_tracks):
                vdir = tr.velocity_dir
                spd = tr.speed

                for j in range(D):
                    t_j = float(det_t[j])

                    # predict to detection time for gating/cost (no mutation)
                    dt_state = t_j - float(tr.state_time)
                    if dt_state <= time_eps:
                        dt_state = 0.0
                    x_pred, P_pred = cv_predict(tr.x, tr.P, dt_state, rts_sigma_a)
                    pred_pos = x_pred[:2]

                    diff = det_xy[j] - pred_pos
                    dist = float(np.hypot(diff[0], diff[1]))

                    if use_kinematic_gate:
                        dt_meas = max(0.0, t_j - float(tr.last_meas_time))
                        dt_meas = float(max(dt_meas, 0.01))

                        if len(tr.row_indices) < vel_gate_min_hits:
                            max_dist = v_max * dt_meas + meas_noise_pos * 2.0
                        else:
                            max_dist = (
                                spd * dt_meas
                                + 0.5 * a_max * (dt_meas**2)
                                + meas_noise_pos * 2.0
                            )

                        if dist > max_dist:
                            valid[i, j] = False
                            continue

                    if use_mahalanobis_gate:
                        S = H @ P_pred @ H.T + R
                        try:
                            maha2 = float(diff.T @ np.linalg.solve(S, diff))
                        except np.linalg.LinAlgError:
                            valid[i, j] = False
                            continue
                        if maha2 > maha_gamma:
                            valid[i, j] = False
                            continue

                    # directional multiplier
                    if dist <= meas_noise_pos * 2.0:
                        mult = 1.0
                    elif spd > min_speed_for_dir:
                        unit = diff / (dist + 1e-6)
                        cos_sim = float(vdir @ unit)
                        angle_penalty = 1.0 - cos_sim
                        mult = 1.0 + direction_weight * angle_penalty
                    else:
                        mult = 1.0

                    cost[i, j] = dist * mult

            matches = solve_hungarian(cost, valid=valid, big=BIG)
            for r, c in matches:
                unmatched_track.discard(r)
                unmatched_det.discard(c)

        # commit matched updates at exact detection times
        for ti, di in matches:
            tr = active_tracks[ti]
            t_match = float(det_t[di])
            tr.predict(t_match, sigma_a=rts_sigma_a, time_eps=time_eps)
            tr.update(
                det_xy[di], t_match, int(row_idxs[di]), meas_noise_pos=meas_noise_pos
            )

            if (not tr.confirmed) and (len(tr.row_indices) >= min_tracklet_points):
                tr.confirmed = True

        # predict unmatched tracks forward to frame reference time
        for ti in list(unmatched_track):
            active_tracks[ti].predict(t_ref_max, sigma_a=rts_sigma_a, time_eps=time_eps)

        # spawn new tracks
        for di in unmatched_det:
            new_tr = Track(
                det_xy[di], float(det_t[di]), int(row_idxs[di]), next_internal_id
            )
            next_internal_id += 1
            active_tracks.append(new_tr)

    # finalize remaining
    for tr in active_tracks:
        finalize_track(tr)

    out = df.copy()
    out["tracklet_id"] = -1
    out.loc[original_index, "tracklet_id"] = tracklet_ids_sorted

    # optional RTS smoothing
    if do_rts_smoothing:
        for col in ("rts_x", "rts_y", "rts_vx", "rts_vy", "rts_speed"):
            out[col] = np.nan

        tmp = out[["tracklet_id", "timestamp", "centroid_x", "centroid_y"]].copy()

        tmp["_t_smooth"] = tmp["timestamp"].astype(float)

        ids = tmp["tracklet_id"].to_numpy()
        unique_ids = np.unique(ids[ids >= 0])

        for tid in unique_ids:
            idx = tmp.index[tmp["tracklet_id"] == tid].to_numpy()
            if len(idx) < 2:
                continue

            t = tmp.loc[idx, "_t_smooth"].to_numpy(float)
            z = tmp.loc[idx, ["centroid_x", "centroid_y"]].to_numpy(float)

            order = np.argsort(t, kind="mergesort")
            idx_sorted = idx[order]

            xs = rts_smooth_tracklet(
                t[order],
                z[order],
                meas_noise_pos=meas_noise_pos,
                sigma_a=rts_sigma_a,
                time_eps=time_eps,
            )

            out.loc[idx_sorted, "rts_x"] = xs[:, 0]
            out.loc[idx_sorted, "rts_y"] = xs[:, 1]
            out.loc[idx_sorted, "rts_vx"] = xs[:, 2]
            out.loc[idx_sorted, "rts_vy"] = xs[:, 3]
            out.loc[idx_sorted, "rts_speed"] = np.hypot(xs[:, 2], xs[:, 3])

    return out
