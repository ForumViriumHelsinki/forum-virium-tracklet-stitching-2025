from __future__ import annotations

import json
from typing import Dict, Optional, Any, List

import numpy as np
import pandas as pd

from track_builder.association.hungarian import (
    calculate_directional_costs,
    solve_hungarian,
)
from track_builder.association.gating import (
    angle_ok_between_dirs,
    circular_radius,
    ellipse_axes,
    mahalanobis2_xy,
    project_to_heading_frame,
    safe_unit,
)
from track_builder.kf.tracks import (
    StitchTrack,
    apply_tracklet_to_kf,
    backward_slack_dynamic,
)
from track_builder.utils import (
    build_geofence_maps,
    canon_class,
    class_compatible,
    merge_max_gap,
    merge_max_turn,
    merge_sigma,
    merge_speed_ellipse_min,
    point_in_poly,
    polyline_length,
)

# -----------------------------------------------------------------------------
# Global configuration (requested)
# -----------------------------------------------------------------------------

CLASS_PARAMS: Dict[str, Dict[str, float]] = {
    "car": dict(
        max_gap_s=3.0,
        v_cap=15.0,
        a_long=2.5,
        a_long_cap_m=8.0,
        lat_base_m=1.5,
        lat_rate_mps=0.4,
        backward_slack_m=0.5,
        max_turn_deg=90.0,
        meas_sigma_m=0.5,
        speed_ellipse_min_mps=2.0,
        long_margin_m=1.0,
        circ_margin_m=1.5,
    ),
    "truck": dict(
        max_gap_s=3.0,
        v_cap=15.0,
        a_long=2.5,
        a_long_cap_m=8.0,
        lat_base_m=1.5,
        lat_rate_mps=0.4,
        backward_slack_m=0.5,
        max_turn_deg=90.0,
        meas_sigma_m=0.5,
        speed_ellipse_min_mps=2.0,
        long_margin_m=1.0,
        circ_margin_m=1.5,
    ),
    "pedestrian": dict(
        max_gap_s=5.0,
        v_cap=3.0,
        a_long=1.2,
        a_long_cap_m=2.0,
        lat_base_m=0.8,
        lat_rate_mps=0.5,
        backward_slack_m=0.3,
        max_turn_deg=90.0,
        meas_sigma_m=0.25,
        speed_ellipse_min_mps=0.6,
        long_margin_m=0.6,
        circ_margin_m=1.0,
    ),
    "ambiguous": dict(
        max_gap_s=5.0,
        v_cap=8.0,
        a_long=1.8,
        a_long_cap_m=4.0,
        lat_base_m=1.0,
        lat_rate_mps=0.45,
        backward_slack_m=0.4,
        max_turn_deg=90.0,
        meas_sigma_m=0.30,
        speed_ellipse_min_mps=0.9,
        long_margin_m=0.8,
        circ_margin_m=1.2,
    ),
}


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _robust_end_velocity(
    times, xs, ys, k: int = 6, min_dt: float = 1e-3
) -> tuple[float, float]:
    n = len(times)
    if n < 2:
        return 0.0, 0.0
    k = min(int(k), n)
    t = np.asarray(times[-k:], float)
    x = np.asarray(xs[-k:], float)
    y = np.asarray(ys[-k:], float)

    dt = float(t[-1] - t[0])
    if dt < min_dt:
        dt2 = float(times[-1] - times[-2])
        if dt2 < min_dt:
            return 0.0, 0.0
        return float((xs[-1] - xs[-2]) / dt2), float((ys[-1] - ys[-2]) / dt2)

    t0 = float(t.mean())
    denom = float(np.sum((t - t0) ** 2) + 1e-12)
    vx = float(np.sum((t - t0) * (x - x.mean())) / denom)
    vy = float(np.sum((t - t0) * (y - y.mean())) / denom)
    return vx, vy


def _robust_start_velocity(
    times, xs, ys, k: int = 6, min_dt: float = 1e-3
) -> tuple[float, float]:
    n = len(times)
    if n < 2:
        return 0.0, 0.0
    k = min(int(k), n)
    t = np.asarray(times[:k], float)
    x = np.asarray(xs[:k], float)
    y = np.asarray(ys[:k], float)

    dt = float(t[-1] - t[0])
    if dt < min_dt:
        dt2 = float(times[1] - times[0])
        if dt2 < min_dt:
            return 0.0, 0.0
        return float((xs[1] - xs[0]) / dt2), float((ys[1] - ys[0]) / dt2)

    t0 = float(t.mean())
    denom = float(np.sum((t - t0) ** 2) + 1e-12)
    vx = float(np.sum((t - t0) * (x - x.mean())) / denom)
    vy = float(np.sum((t - t0) * (y - y.mean())) / denom)
    return vx, vy


def _get_tracklet_series(
    g: pd.DataFrame,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]
]:
    """
    Returns (t, x, y, vx, vy) where x,y are positions (RTS if available else centroid),
    vx,vy may be RTS velocity arrays or None.
    """
    t = g["timestamp"].to_numpy(float)

    use_rts = ("rts_x" in g.columns) and g["rts_x"].notna().all()
    if use_rts:
        x = g["rts_x"].to_numpy(float)
        y = g["rts_y"].to_numpy(float)
        if ("rts_vx" in g.columns) and g["rts_vx"].notna().all():
            vx = g["rts_vx"].to_numpy(float)
            vy = g["rts_vy"].to_numpy(float)
        else:
            vx = vy = None
    else:
        x = g["centroid_x"].to_numpy(float)
        y = g["centroid_y"].to_numpy(float)
        vx = vy = None

    return t, x, y, vx, vy


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------


def stitch_tracklets_hungarian(
    df: pd.DataFrame,
    class_params: Dict[str, Dict[str, float]] = CLASS_PARAMS,
    geofence_path: Any = None,
    area_name: str = "general_area",
    tracklet_id_col: str = "tracklet_id",
    max_stitch_length: float = 90.0,
    direction_weight: float = 1.5,
    min_speed_for_dir: float = 0.5,
    use_kinematic_gate: bool = True,
    use_mahalanobis_gate: bool = True,
    maha_gamma: float = 9.21,
    sigma_a: float = 3.0,
    replay_stride: int = 1,
    terminate_on_leave_area: bool = True,
) -> pd.DataFrame:
    """
    Stitch tracklets end->start using Hungarian assignment among active stitched tracks.

    Inputs expected:
      - tracklet_id (tracklet_id_col)
      - timestamp, centroid_x, centroid_y
      - predicted_class (recommended; else treated ambiguous)
      - optionally rts_x,rts_y,rts_vx,rts_vy (improves summaries + replay)

    Outputs added:
      - stitched_id
      - entry_well, exit_well (if geofence wells provided)
    """

    id_col = str(tracklet_id_col)
    if id_col not in df.columns:
        raise ValueError(f"Missing tracklet id column '{id_col}'")

    required = {"timestamp", "centroid_x", "centroid_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["stitched_id"] = -1
    out["entry_well"] = None
    out["exit_well"] = None

    valid_df = out[out[id_col] >= 0].copy()
    if len(valid_df) == 0:
        return out

    # --- Geofences (optional) ---
    area_poly = None
    well_polys: Dict[str, np.ndarray] = {}
    if geofence_path is not None:
        with open(str(geofence_path), "r") as f:
            geofences = json.load(f)
        area_poly, well_polys = build_geofence_maps(geofences, area_name=area_name)

    # --- Build per-tracklet summaries (RTS-first) ---
    summaries: List[Dict[str, Any]] = []
    for tid, g in valid_df.sort_values("timestamp").groupby(id_col, sort=False):
        g = g.sort_values("timestamp")
        t, x, y, vx, vy = _get_tracklet_series(g)

        if len(t) < 2:
            continue

        t0, t1 = float(t[0]), float(t[-1])
        p0 = np.array([x[0], y[0]], dtype=float)
        p1 = np.array([x[-1], y[-1]], dtype=float)

        if vx is not None and vy is not None and len(vx) > 0:
            v_start = np.array([vx[0], vy[0]], dtype=float)
            v_end = np.array([vx[-1], vy[-1]], dtype=float)
        else:
            vx_end, vy_end = _robust_end_velocity(t, x, y, k=6)
            vx_start, vy_start = _robust_start_velocity(t, x, y, k=6)
            v_start = np.array([vx_start, vy_start], dtype=float)
            v_end = np.array([vx_end, vy_end], dtype=float)

        length = polyline_length(x, y)

        cls_raw = None
        if "predicted_class" in g.columns and g["predicted_class"].notna().any():
            cls_raw = g["predicted_class"].mode().iat[0]
        cls = canon_class(cls_raw, class_params, default="ambiguous")

        disp_dir = safe_unit(p1 - p0)

        summaries.append(
            dict(
                tracklet_id=int(tid),
                t0=t0,
                t1=t1,
                p0=p0,
                p1=p1,
                v_start=v_start,
                v_end=v_end,
                length=float(length),
                cls=str(cls),
                disp_dir=disp_dir,
            )
        )

    if not summaries:
        return out

    summaries.sort(key=lambda s: s["t0"])
    start_times = np.array(sorted({s["t0"] for s in summaries}), dtype=float)
    by_t0: Dict[float, List[Dict[str, Any]]] = {}
    for s in summaries:
        by_t0.setdefault(float(s["t0"]), []).append(s)

    active: List[StitchTrack] = []
    next_stitch_id = 0
    stitch_assignment: Dict[int, int] = {}
    stitch_meta: Dict[int, Dict[str, Any]] = {}

    def finalize_stitch(tr: StitchTrack) -> None:
        if tr.has_left_entry:
            tr.exit_well = tr.last_visited_non_entry_well
        else:
            tr.exit_well = None

        for tid in tr.members:
            stitch_assignment[int(tid)] = int(tr.id)

        stitch_meta[int(tr.id)] = {
            "entry_well": tr.entry_well,
            "exit_well": tr.exit_well,
        }

    # ---- Main loop over candidate start times ----
    for t0 in start_times:
        cand_list = by_t0[float(t0)]

        # 1) Retire + predict + optional geofence termination
        still_active: List[StitchTrack] = []
        for tr in active:
            gap = float(t0) - float(tr.last_time)
            if gap < 0.0:
                still_active.append(tr)
                continue

            tr_cls = canon_class(tr.obj_class, class_params, default="ambiguous")
            if (
                gap > float(class_params[tr_cls]["max_gap_s"])
                or tr.length_so_far > max_stitch_length
            ):
                finalize_stitch(tr)
                continue

            tr.predict(float(t0), sigma_a=sigma_a)

            if well_polys:
                tr.update_wells_from_predicted_pos(well_polys)

            if terminate_on_leave_area and area_poly is not None:
                now_in = point_in_poly(float(tr.x[0]), float(tr.x[1]), area_poly)
                tr.entered_area = tr.entered_area or now_in
                if tr.entered_area and (not now_in):
                    finalize_stitch(tr)
                    continue
                tr.in_area = now_in

            still_active.append(tr)
        active = still_active

        # 2) Association
        T = len(active)
        D = len(cand_list)
        matches: List[tuple[int, int]] = []
        unmatched_track_indices = set(range(T))
        unmatched_cand_indices = set(range(D))

        if T > 0 and D > 0:
            detections = np.stack([c["p0"] for c in cand_list], axis=0)  # (D,2)

            cost_matrix, raw_dists = calculate_directional_costs(
                active,
                detections,
                angle_weight=direction_weight,
                speed_threshold=min_speed_for_dir,
            )

            BIG = 1e9
            valid = np.ones((T, D), dtype=bool)

            # temporal gate (candidate start must be after last track time)
            track_last = np.array([float(tr.last_time) for tr in active], dtype=float)
            gap_raw = float(t0) - track_last
            valid &= gap_raw[:, None] >= 0.0
            gap_vec = np.maximum(gap_raw, 0.0)

            # class compatibility + per-pair max gap
            track_cls = [
                canon_class(tr.obj_class, class_params, default="ambiguous")
                for tr in active
            ]
            cand_cls = [
                canon_class(c["cls"], class_params, default="ambiguous")
                for c in cand_list
            ]

            for i in range(T):
                for j in range(D):
                    if not class_compatible(
                        track_cls[i], cand_cls[j], ambiguous="ambiguous"
                    ):
                        valid[i, j] = False
                        continue
                    max_gap = merge_max_gap(
                        track_cls[i], cand_cls[j], class_params, key="max_gap_s"
                    )
                    if float(gap_vec[i]) > float(max_gap):
                        valid[i, j] = False

            # geometry diffs
            track_pos = np.array(
                [[float(tr.x[0]), float(tr.x[1])] for tr in active], dtype=float
            )
            diff = detections[None, :, :] - track_pos[:, None, :]  # (T,D,2)

            # --- Kinematic gate: ellipse (moving) or circle (slow) ---
            if use_kinematic_gate:
                track_speed = np.array([float(tr.speed) for tr in active], dtype=float)

                cand_vstart = np.stack(
                    [c["v_start"] for c in cand_list], axis=0
                ).astype(float)
                cand_speed = np.linalg.norm(cand_vstart, axis=1)

                for i in range(T):
                    # per-track heading
                    h = safe_unit(active[i].heading_dir())
                    heading_ok = float(np.linalg.norm(h)) > 1e-6
                    tc = track_cls[i]

                    for j in range(D):
                        if not valid[i, j]:
                            continue

                        cc = cand_cls[j]
                        dt = float(gap_vec[i])

                        speed_min_for_ellipse = merge_speed_ellipse_min(
                            tc, cc, class_params, key="speed_ellipse_min_mps"
                        )
                        max_turn_deg = merge_max_turn(
                            tc, cc, class_params, key="max_turn_deg"
                        )

                        # decide ellipse vs circle
                        use_ellipse = heading_ok and (
                            track_speed[i] >= float(speed_min_for_ellipse)
                        )

                        # reachability uses track class params (conservative enough because ellipse/circle is gated)
                        params = class_params[tc]
                        v_eff = float(min(track_speed[i], float(params["v_cap"])))

                        if use_ellipse:
                            # project displacement into heading frame
                            d_par, d_per, _ = project_to_heading_frame(h, diff[i, j])

                            # backward slack gate
                            base_back = float(params["backward_slack_m"])
                            slack = backward_slack_dynamic(
                                active[i], h, dt, base_back, k_back=3.0, max_back=6.0
                            )
                            if d_par < -float(slack):
                                valid[i, j] = False
                                continue

                            a, b = ellipse_axes(v_eff, dt, params)
                            if a <= 1e-6 or b <= 1e-6:
                                valid[i, j] = False
                                continue

                            ell = (d_par / a) ** 2 + (d_per / b) ** 2
                            if float(ell) > 1.0:
                                valid[i, j] = False
                                continue

                            # optional direction agreement gate if both moving enough
                            if (cand_speed[j] >= float(speed_min_for_ellipse)) and (
                                track_speed[i] >= float(speed_min_for_ellipse)
                            ):
                                if not angle_ok_between_dirs(
                                    h, cand_vstart[j], max_turn_deg=float(max_turn_deg)
                                ):
                                    valid[i, j] = False
                                    continue
                        else:
                            r = circular_radius(v_eff, dt, params)
                            if float(raw_dists[i, j]) > float(r):
                                valid[i, j] = False
                                continue

            # --- Mahalanobis gate (pairwise sigma) ---
            if use_mahalanobis_gate:
                for i, tr in enumerate(active):
                    tc = track_cls[i]
                    for j in range(D):
                        if not valid[i, j]:
                            continue
                        cc = cand_cls[j]
                        sigma = merge_sigma(tc, cc, class_params, key="meas_sigma_m")
                        m2 = mahalanobis2_xy(diff[i, j], tr.P, meas_sigma=float(sigma))
                        if not np.isfinite(m2) or float(m2) > float(maha_gamma):
                            valid[i, j] = False

            # optional: if already entered the area, don't link to a candidate starting outside
            if area_poly is not None and terminate_on_leave_area:
                cand_in = np.array(
                    [
                        point_in_poly(float(c["p0"][0]), float(c["p0"][1]), area_poly)
                        for c in cand_list
                    ],
                    dtype=bool,
                )
                for i, tr in enumerate(active):
                    if tr.entered_area:
                        valid[i, :] &= cand_in

            matches = solve_hungarian(cost_matrix, valid=valid, big=1e9)
            for r, c in matches:
                unmatched_track_indices.discard(r)
                unmatched_cand_indices.discard(c)

        # 3) Apply matches: replay candidate tracklet points into KF, update metadata
        for ti, ci in matches:
            tr = active[ti]
            cand = cand_list[ci]

            # update class: ambiguous adopts concrete
            cand_c = canon_class(cand["cls"], class_params, default="ambiguous")
            tr_c = canon_class(tr.obj_class, class_params, default="ambiguous")
            if tr_c == "ambiguous" and cand_c != "ambiguous":
                tr.obj_class = cand_c
                tr_c = cand_c

            # replay points (RTS if present)
            cand_tid = int(cand["tracklet_id"])
            rows = valid_df[valid_df[id_col] == cand_tid].sort_values("timestamp")

            # merged sigma for update
            sigma = merge_sigma(tr_c, cand_c, class_params, key="meas_sigma_m")

            cols = ["timestamp", "centroid_x", "centroid_y"]
            if ("rts_x" in rows.columns) and ("rts_y" in rows.columns):
                cols += ["rts_x", "rts_y"]

            apply_tracklet_to_kf(
                tr,
                rows[cols].copy(),
                meas_sigma_m=float(sigma),
                sigma_a=sigma_a,
                stride=replay_stride,
                update_wells=True,
                well_polys=well_polys,
            )

            # update heading fallback using candidate displacement
            if (
                cand.get("disp_dir", None) is not None
                and float(np.linalg.norm(cand["disp_dir"])) > 1e-6
            ):
                tr.last_disp_dir = safe_unit(cand["disp_dir"])

            tr.length_so_far += float(cand["length"])
            tr.members.append(cand_tid)

            # geofence termination at end
            if terminate_on_leave_area and area_poly is not None:
                end_in = point_in_poly(float(tr.x[0]), float(tr.x[1]), area_poly)
                tr.entered_area = tr.entered_area or end_in
                tr.in_area = end_in
                if tr.entered_area and (not end_in):
                    finalize_stitch(tr)
                    tr._terminated = True

        active = [tr for tr in active if not tr._terminated]

        # 4) Spawn new stitches from unmatched candidates
        for ci in unmatched_cand_indices:
            cand = cand_list[ci]
            cand_c = canon_class(cand["cls"], class_params, default="ambiguous")

            # area flags (keep your existing logic)
            if area_poly is not None:
                in_area_end = point_in_poly(
                    float(cand["p1"][0]), float(cand["p1"][1]), area_poly
                )
                in_area_start = point_in_poly(
                    float(cand["p0"][0]), float(cand["p0"][1]), area_poly
                )
                entered = bool(in_area_start or in_area_end)
            else:
                in_area_end = False
                entered = False

            x0 = np.array(
                [
                    float(cand["p0"][0]),
                    float(cand["p0"][1]),
                    float(cand["v_start"][0]),
                    float(cand["v_start"][1]),
                ],
                dtype=float,
            )

            tr = StitchTrack(
                x_end=x0,
                t_end=float(cand["t0"]),  # start time
                length_so_far=float(cand["length"]),
                member_tracklets=[int(cand["tracklet_id"])],
                stitch_id=next_stitch_id,
                obj_class=str(cand_c),
                P_init=np.diag([1.0, 1.0, 9.0, 9.0]),
                in_area=in_area_end,
                entered_area=entered,
                entry_well=None,
                exit_well=None,
                entry_locked=False,
                has_left_entry=False,
                last_visited_non_entry_well=None,
                last_disp_dir=cand.get("disp_dir", None),
            )

            cand_tid = int(cand["tracklet_id"])
            rows = valid_df[valid_df[id_col] == cand_tid].sort_values("timestamp")

            sigma = float(class_params[cand_c]["meas_sigma_m"])

            cols = ["timestamp", "centroid_x", "centroid_y"]
            if "rts_x" in rows.columns:
                cols += ["rts_x", "rts_y"]

            apply_tracklet_to_kf(
                tr,
                rows[cols].copy(),
                meas_sigma_m=float(sigma),
                sigma_a=sigma_a,
                stride=replay_stride,
                update_wells=True,
                well_polys=well_polys,
            )

            # ensure clocks consistent after replay
            if len(rows) > 0:
                tr.last_time = float(rows["timestamp"].iloc[-1])
                tr.state_time = tr.last_time

            next_stitch_id += 1
            active.append(tr)

    # finalize remaining
    for tr in active:
        finalize_stitch(tr)

    # write outputs back
    mapped_sid = valid_df[id_col].map(lambda tid: stitch_assignment.get(int(tid), -1))
    out.loc[valid_df.index, "stitched_id"] = mapped_sid.to_numpy()
    out.loc[valid_df.index, "entry_well"] = mapped_sid.map(
        lambda sid: stitch_meta.get(int(sid), {}).get("entry_well")
    )
    out.loc[valid_df.index, "exit_well"] = mapped_sid.map(
        lambda sid: stitch_meta.get(int(sid), {}).get("exit_well")
    )

    return out
