from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from track_builder.utils import which_well
from .motion_model import cv_predict, kf_update_joseph


def _safe_unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=float)
    return v / n


class Track:
    """
    Online track used during initial tracklet building.

    State: [x, y, vx, vy]
    """

    def __init__(self, x0, t0, row_idx: int, track_id: int):
        self.id = int(track_id)

        x0 = np.asarray(x0, dtype=float).reshape(2)
        self.x = np.array([x0[0], x0[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([1.0, 1.0, 10.0, 10.0]).astype(float)

        self.state_time = float(t0)
        self.last_meas_time = float(t0)
        self.confirmed = False
        self.row_indices = [int(row_idx)]

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))

    @property
    def velocity_dir(self) -> np.ndarray:
        s = self.speed
        if s < 0.1:
            return np.array([0.0, 0.0], dtype=float)
        return np.array([self.x[2] / s, self.x[3] / s], dtype=float)

    def predict(
        self, t_now: float, sigma_a: float = 3.0, time_eps: float = 1e-4
    ) -> None:
        """Commit prediction to time t_now."""
        t_now = float(t_now)
        dt = t_now - float(self.state_time)
        if dt <= float(time_eps):
            return
        self.x, self.P = cv_predict(self.x, self.P, dt, sigma_a)
        self.state_time = t_now

    def update(
        self, z_xy, t_now: float, row_idx: int, meas_noise_pos: float = 0.25
    ) -> None:
        """KF position update (Joseph form)."""
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        R = np.eye(2, dtype=float) * (float(meas_noise_pos) ** 2)

        x_new, P_new = kf_update_joseph(self.x, self.P, np.asarray(z_xy, float), H, R)
        self.x, self.P = x_new, P_new

        self.last_meas_time = float(t_now)
        self.row_indices.append(int(row_idx))


class StitchTrack:
    """
    KF-backed stitched track that "replays" tracklet points (RTS if available).

    State: [x, y, vx, vy]
    """

    def __init__(
        self,
        x_end,  # [x,y,vx,vy]
        t_end,
        length_so_far,
        member_tracklets,
        stitch_id,
        obj_class: str = "ambiguous",
        P_init: Optional[np.ndarray] = None,
        in_area: bool = False,
        entered_area: bool = False,
        entry_well=None,
        exit_well=None,
        entry_locked: bool = False,
        has_left_entry: bool = False,
        last_visited_non_entry_well=None,
        last_disp_dir=None,  # optional heading fallback
    ):
        self.id = int(stitch_id)
        self.obj_class = str(obj_class)

        x_end = np.asarray(x_end, dtype=float).reshape(4)
        self.x = x_end.copy()
        self.P = (
            np.diag([1.0, 1.0, 9.0, 9.0])
            if P_init is None
            else np.asarray(P_init, float).reshape(4, 4)
        )

        self.state_time = float(t_end)
        self.last_time = float(t_end)

        self.length_so_far = float(length_so_far)
        self.members = list(member_tracklets)

        self.in_area = bool(in_area)
        self.entered_area = bool(entered_area)
        self.entry_well = entry_well
        self.exit_well = exit_well
        self.entry_locked = bool(entry_locked)
        self.has_left_entry = bool(has_left_entry)
        self.last_visited_non_entry_well = last_visited_non_entry_well

        self.last_disp_dir = (
            None if last_disp_dir is None else _safe_unit(last_disp_dir)
        )

        self._terminated = False

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))

    @property
    def velocity_dir(self) -> np.ndarray:
        return _safe_unit(self.x[2:4])

    def _update_exit_state_from_well(self, w):
        """
        Applies the canonical rules:
        - has_left_entry becomes True once we are no longer in entry_well (if entry_well is not None)
        - last_visited_non_entry_well updates to the last non-entry well visited (prediction or measurement),
            but ONLY after has_left_entry is True.
        """
        # If entry_well is None, we treat "left entry" as already satisfied (set at lock time).
        if self.entry_well is not None and (w != self.entry_well):
            self.has_left_entry = True

        if self.has_left_entry and (w is not None) and (w != self.entry_well):
            self.last_visited_non_entry_well = w

    def update_wells_from_measured_pos(self, z_xy, well_polys):
        """Call after update_pos(): uses measured/replayed point."""
        if not well_polys:
            return

        # lock entry from FIRST detection only
        self.maybe_set_entry_from_first_measurement(z_xy, well_polys)

        w_meas = which_well(np.asarray(z_xy, float), well_polys)
        self._update_exit_state_from_well(w_meas)

    def update_wells_from_predicted_pos(self, well_polys):
        """Call after predict(): uses current tr.x[:2] as predicted position."""
        if not well_polys:
            return
        z = np.array([float(self.x[0]), float(self.x[1])], dtype=float)
        w_pred = which_well(z, well_polys)
        self._update_exit_state_from_well(w_pred)

    def maybe_set_entry_from_first_measurement(self, z_xy, well_polys):
        """
        Locks entry_well based ONLY on the first measurement ever replayed into this stitch track.
        If first measurement is not in a well -> entry_well stays None (locked).
        """
        if self.entry_locked or (not well_polys):
            return

        w0 = which_well(np.asarray(z_xy, float), well_polys)
        self.entry_well = w0  # may be None
        self.entry_locked = True

        # RULESET: if entry_well is None, exit-well eligibility is allowed immediately
        if self.entry_well is None:
            self.has_left_entry = True

    def heading_dir(self) -> np.ndarray:
        """
        Prefer velocity direction; fall back to last displacement direction; otherwise zero.
        """
        vdir = self.velocity_dir
        if float(np.linalg.norm(vdir)) > 1e-6:
            return vdir
        if (
            self.last_disp_dir is not None
            and float(np.linalg.norm(self.last_disp_dir)) > 1e-6
        ):
            return self.last_disp_dir
        return np.array([0.0, 0.0], dtype=float)

    def predict(self, t_now: float, sigma_a: float = 3.0) -> None:
        dt = float(t_now) - float(self.state_time)
        if dt <= 0.0:
            return
        self.x, self.P = cv_predict(self.x, self.P, dt, sigma_a)
        self.state_time = float(t_now)

    def update_pos(self, z_xy, meas_sigma_m: float = 0.25) -> None:
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        R = np.eye(2, dtype=float) * (float(meas_sigma_m) ** 2)

        # robust solve guard
        try:
            x_new, P_new = kf_update_joseph(
                self.x, self.P, np.asarray(z_xy, float), H, R
            )
        except np.linalg.LinAlgError:
            return
        self.x, self.P = x_new, P_new


def apply_tracklet_to_kf(
    tr: StitchTrack,
    tracklet_rows: pd.DataFrame,
    meas_sigma_m: float = 0.25,
    sigma_a: float = 3.0,
    stride: int = 1,
    *,
    update_wells: bool = False,
    well_polys=None,
) -> None:
    """
    Replay a tracklet into an existing StitchTrack KF.

    If update_wells=True and well_polys provided, updates:
      - appends measured well to wells_crossed (unique consecutive)
    """
    if len(tracklet_rows) == 0:
        return

    ts = tracklet_rows["timestamp"].to_numpy(dtype=float)

    use_rts = (
        ("rts_x" in tracklet_rows.columns)
        and ("rts_y" in tracklet_rows.columns)
        and tracklet_rows["rts_x"].notna().all()
        and tracklet_rows["rts_y"].notna().all()
    )
    if use_rts:
        xs = tracklet_rows["rts_x"].to_numpy(dtype=float)
        ys = tracklet_rows["rts_y"].to_numpy(dtype=float)
    else:
        xs = tracklet_rows["centroid_x"].to_numpy(dtype=float)
        ys = tracklet_rows["centroid_y"].to_numpy(dtype=float)

    stride = max(1, int(stride))
    for k in range(0, len(ts), stride):
        t = float(ts[k])
        z = np.array([float(xs[k]), float(ys[k])], dtype=float)

        tr.predict(t, sigma_a=sigma_a)
        tr.update_pos(z, meas_sigma_m=meas_sigma_m)
        tr.last_time = t

        if update_wells:
            tr.update_wells_from_measured_pos(z, well_polys)


def backward_slack_dynamic(
    tr: StitchTrack,
    heading_unit: np.ndarray,
    dt: float,
    base_back: float,
    k_back: float = 3.0,
    back_rate: float = 0.0,
    min_back: float = 0.0,
    max_back: float = 5.0,
) -> float:
    """
    Backward slack based on projected position uncertainty along heading direction.

    Uses P[:2,:2] and the heading vector; increases slack with uncertainty + optional back_rate*dt.
    """
    h = _safe_unit(heading_unit)
    if float(np.linalg.norm(h)) < 1e-6:
        return float(base_back)

    Pxy = np.asarray(tr.P[:2, :2], dtype=float)
    sigma_par = float(np.sqrt(max(0.0, h.T @ Pxy @ h)))

    slack = float(base_back) + float(k_back) * sigma_par + float(back_rate) * float(dt)
    return float(np.clip(slack, float(min_back), float(max_back)))
