from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def cv_F_Q(dt: float, sigma_a: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constant-velocity (CV) model for state [x, y, vx, vy].

    Returns:
        F: (4,4) state transition
        Q: (4,4) process noise covariance (white accel with std sigma_a)
    """
    dt = float(max(dt, 0.0))

    F = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    q = float(sigma_a) ** 2

    Q = q * np.array(
        [
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ],
        dtype=float,
    )

    return F, Q


def cv_predict(
    x: np.ndarray,
    P: np.ndarray,
    dt: float,
    sigma_a: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict state/cov forward by dt under CV model (no mutation).
    """
    F, Q = cv_F_Q(dt, sigma_a)
    x = np.asarray(x, dtype=float).reshape(4)
    P = np.asarray(P, dtype=float).reshape(4, 4)
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def kf_update_joseph(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic Kalman measurement update using Joseph stabilized covariance form.

    Args:
        x: (n,) state
        P: (n,n) covariance
        z: (m,) measurement
        H: (m,n) measurement matrix
        R: (m,m) measurement covariance

    Returns:
        (x_upd, P_upd) (no mutation)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    P = np.asarray(P, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    H = np.asarray(H, dtype=float)
    R = np.asarray(R, dtype=float)

    y = z - (H @ x)  # innovation
    S = H @ P @ H.T + R  # innovation covariance

    PHt = P @ H.T
    # K = P H^T S^{-1}, avoid explicit inverse
    K = np.linalg.solve(S, PHt.T).T

    x_new = x + K @ y

    I = np.eye(P.shape[0], dtype=float)
    A = I - K @ H
    P_new = A @ P @ A.T + K @ R @ K.T

    # enforce symmetry (guards tiny numeric drift)
    P_new = 0.5 * (P_new + P_new.T)
    return x_new, P_new


def rts_smooth_tracklet(
    t: np.ndarray,
    z_xy: np.ndarray,
    meas_noise_pos: float = 0.25,
    sigma_a: float = 3.0,
    P0: Optional[np.ndarray] = None,
    time_eps: float = 1e-3,
) -> np.ndarray:
    """
    Rauch-Tung-Striebel smoothing for 2D positions under a CV model.

    State: [x, y, vx, vy]
    Measurements: [x, y]

    Args:
        t: (N,) timestamps (must be sortable)
        z_xy: (N,2) measured positions
        meas_noise_pos: measurement std (meters)
        sigma_a: process accel std
        P0: optional initial covariance (4,4)
        time_eps: small dt treated as 0 (numerical stability)

    Returns:
        x_s: (N,4) smoothed states
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    z = np.asarray(z_xy, dtype=float)
    n = int(t.shape[0])

    if n == 0:
        return np.empty((0, 4), dtype=float)
    if z.ndim != 2 or z.shape[0] != n or z.shape[1] != 2:
        raise ValueError(f"z_xy must be shape (N,2). Got {z.shape}, expected ({n},2).")

    # Measurement model
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
    R = np.eye(2, dtype=float) * (float(meas_noise_pos) ** 2)

    # Init
    x0 = np.array([z[0, 0], z[0, 1], 0.0, 0.0], dtype=float)
    P = (
        np.diag([1.0, 1.0, 10.0, 10.0]).astype(float)
        if P0 is None
        else np.asarray(P0, dtype=float).reshape(4, 4).copy()
    )

    x_f = np.zeros((n, 4), dtype=float)
    P_f = np.zeros((n, 4, 4), dtype=float)
    x_p = np.zeros((n, 4), dtype=float)
    P_p = np.zeros((n, 4, 4), dtype=float)
    F_list = np.zeros((n, 4, 4), dtype=float)

    x_f[0] = x0
    P_f[0] = P

    # Forward KF
    for k in range(1, n):
        dt = float(t[k] - t[k - 1])
        if dt <= float(time_eps):
            dt = 0.0
        F, Q = cv_F_Q(dt, sigma_a)

        x_pred = F @ x_f[k - 1]
        P_pred = F @ P_f[k - 1] @ F.T + Q

        x_p[k] = x_pred
        P_p[k] = P_pred
        F_list[k] = F

        # update with z[k]
        y = z[k] - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        PHt = P_pred @ H.T
        K = np.linalg.solve(S, PHt.T).T

        x_f[k] = x_pred + K @ y

        I = np.eye(4, dtype=float)
        A = I - K @ H
        P_f[k] = A @ P_pred @ A.T + K @ R @ K.T
        P_f[k] = 0.5 * (P_f[k] + P_f[k].T)

    # Backward RTS
    x_s = x_f.copy()
    P_s = P_f.copy()

    for k in range(n - 2, -1, -1):
        F = F_list[k + 1]
        P_pred = P_p[k + 1]

        # robust guard against nasty conditioning
        if np.linalg.cond(P_pred) > 1e12:
            continue

        PFt = P_f[k] @ F.T
        Ck = np.linalg.solve(P_pred, PFt.T).T

        x_s[k] = x_f[k] + Ck @ (x_s[k + 1] - x_p[k + 1])
        P_s[k] = P_f[k] + Ck @ (P_s[k + 1] - P_pred) @ Ck.T
        P_s[k] = 0.5 * (P_s[k] + P_s[k].T)

    return x_s
