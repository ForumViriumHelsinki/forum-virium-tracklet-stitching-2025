from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def solve_hungarian(
    cost: np.ndarray,
    valid: Optional[np.ndarray] = None,
    big: float = 1e9,
) -> List[Tuple[int, int]]:
    """
    Solve linear assignment with optional validity mask.

    Args:
        cost: (T,D) cost matrix.
        valid: optional (T,D) boolean mask. Invalid entries are set to `big`.
        big: sentinel cost for invalid edges.

    Returns:
        List of (row_idx, col_idx) matches. Only returns pairs that are valid and < big.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2D, got shape {cost.shape}")

    T, D = cost.shape
    if T == 0 or D == 0:
        return []

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != cost.shape:
            raise ValueError(f"valid must have shape {cost.shape}, got {valid.shape}")
        cost2 = cost.copy()
        cost2[~valid] = float(big)
    else:
        cost2 = cost

    row_ind, col_ind = linear_sum_assignment(cost2)

    matches: List[Tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        if r < 0 or c < 0:
            continue
        if valid is not None and not bool(valid[r, c]):
            continue
        if float(cost2[r, c]) >= float(big):
            continue
        matches.append((int(r), int(c)))
    return matches


def calculate_directional_costs(
    tracks: Sequence,
    detections_xy: np.ndarray,
    angle_weight: float = 2.0,
    speed_threshold: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distance cost with directional multiplier.

    Expects each `track` to have:
      - tr.x with position at [0],[1]
      - tr.velocity_dir property -> unit-ish (2,)
      - tr.speed property -> float

    Args:
        tracks: list of track-like objects
        detections_xy: (D,2) array
        angle_weight: multiplier strength on direction disagreement
        speed_threshold: only apply angular multiplier when track speed > threshold

    Returns:
        (costs, raw_dists)
          - costs: (T,D)
          - raw_dists: (T,D)
    """
    det_pos = np.asarray(detections_xy, dtype=float)
    if det_pos.ndim != 2 or det_pos.shape[1] != 2:
        raise ValueError(f"detections_xy must be shape (D,2), got {det_pos.shape}")

    T = len(tracks)
    D = int(det_pos.shape[0])
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=float), np.zeros((T, D), dtype=float)

    track_pos = np.array(
        [[float(tr.x[0]), float(tr.x[1])] for tr in tracks], dtype=float
    )  # (T,2)
    diff = det_pos[None, :, :] - track_pos[:, None, :]  # (T,D,2)
    dists = np.linalg.norm(diff, axis=2)  # (T,D)

    track_vels = np.array(
        [np.asarray(tr.velocity_dir, dtype=float).reshape(2) for tr in tracks],
        dtype=float,
    )  # (T,2)
    track_speeds = np.array([float(tr.speed) for tr in tracks], dtype=float)  # (T,)

    diff_norm = diff / (dists[:, :, None] + 1e-6)
    cos_sim = np.einsum("ti,tdi->td", track_vels, diff_norm)  # (T,D)

    # penalty factor: aligned -> 0, opposite -> 2
    angle_penalty = 1.0 - cos_sim
    multipliers = np.ones_like(dists, dtype=float)

    moving = track_speeds > float(speed_threshold)
    if np.any(moving):
        multipliers[moving, :] += float(angle_weight) * angle_penalty[moving, :]

    return dists * multipliers, dists
