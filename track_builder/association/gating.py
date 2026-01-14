from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def safe_unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=float)
    return v / n


def angle_ok_between_dirs(
    u: np.ndarray, v: np.ndarray, max_turn_deg: float, eps: float = 1e-9
) -> bool:
    """
    True if angle(u, v) <= max_turn_deg. If either vector is near-zero, returns True.
    """
    u = np.asarray(u, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < eps or nv < eps:
        return True
    c = float(np.clip(float(np.dot(u, v)) / (nu * nv), -1.0, 1.0))
    return c >= float(np.cos(np.deg2rad(float(max_turn_deg))))


def ellipse_axes(v_eff: float, dt: float, params: dict) -> Tuple[float, float]:
    """
    Forward ellipse semi-axes (a: along heading, b: lateral).

    Uses:
      v_cap, a_long, a_long_cap_m, long_margin_m, lat_base_m, lat_rate_mps
    """
    v_cap = float(params["v_cap"])
    v = float(min(max(v_eff, 0.0), v_cap))

    a_long = float(params["a_long"])
    a_cap = float(params["a_long_cap_m"])
    long_margin = float(params.get("long_margin_m", 0.0))

    dt = float(max(dt, 0.0))
    a_acc = min(0.5 * a_long * (dt**2), a_cap)
    a = v * dt + a_acc + long_margin

    lat_base = float(params["lat_base_m"])
    lat_rate = float(params["lat_rate_mps"])
    b = lat_base + lat_rate * dt

    return float(a), float(b)


def circular_radius(v_eff: float, dt: float, params: dict) -> float:
    """
    Circular fallback radius.
    Uses: v_cap, a_long, a_long_cap_m, circ_margin_m
    """
    v_cap = float(params["v_cap"])
    v = float(min(max(v_eff, 0.0), v_cap))

    a_long = float(params["a_long"])
    a_cap = float(params["a_long_cap_m"])
    circ_margin = float(params.get("circ_margin_m", 0.0))

    dt = float(max(dt, 0.0))
    a_acc = min(0.5 * a_long * (dt**2), a_cap)
    r = v * dt + a_acc + circ_margin
    return float(r)


def mahalanobis2_xy(
    diff_xy: np.ndarray,
    P: np.ndarray,
    meas_sigma: float,
    H: Optional[np.ndarray] = None,
) -> float:
    """
    Compute squared Mahalanobis distance for a 2D position residual:
        maha2 = diff^T S^{-1} diff
    where S = H P H^T + R, R = sigma^2 I2.

    Args:
        diff_xy: (2,) residual in measurement space (z - Hx) or (det - pred_pos)
        P: (4,4) state covariance
        meas_sigma: measurement std (meters)
        H: optional (2,4) measurement matrix; default maps [x,y] from state

    Returns:
        maha2 (float). Returns inf if solve fails.
    """
    d = np.asarray(diff_xy, dtype=float).reshape(2)
    P = np.asarray(P, dtype=float).reshape(4, 4)
    if H is None:
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
    else:
        H = np.asarray(H, dtype=float).reshape(2, 4)

    R = np.eye(2, dtype=float) * (float(meas_sigma) ** 2)
    S = H @ P @ H.T + R
    try:
        return float(d.T @ np.linalg.solve(S, d))
    except np.linalg.LinAlgError:
        return float("inf")


def project_to_heading_frame(
    heading_unit: np.ndarray,
    diff_xy: np.ndarray,
    eps: float = 1e-9,
) -> Tuple[float, float, bool]:
    """
    Project a 2D displacement into a heading-aligned frame.

    Returns:
        d_par: scalar along heading
        d_per: scalar along left-normal
        ok: False if heading is near-zero
    """
    h = safe_unit(heading_unit, eps=eps).reshape(2)
    if float(np.linalg.norm(h)) < eps:
        return 0.0, 0.0, False
    n = np.array([-h[1], h[0]], dtype=float)
    d = np.asarray(diff_xy, dtype=float).reshape(2)
    d_par = float(h @ d)
    d_per = float(n @ d)
    return d_par, d_per, True
