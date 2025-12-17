import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def split_tracklets(df: pd.DataFrame, val_frac=0.2, seed=123):
    rng = np.random.default_rng(seed)
    tids = df["tracklet_id"].unique()
    rng.shuffle(tids)
    n_val = int(len(tids) * val_frac)
    val_ids = set(tids[:n_val])
    train_ids = set(tids[n_val:])
    return train_ids, val_ids


def build_tracklet_index(df: pd.DataFrame, tracklet_ids, min_len: int):
    """
    Returns dict: tid -> numpy array of row indices sorted by timestamp
    Filters out tracklets with length < min_len
    """
    sub = df[df["tracklet_id"].isin(tracklet_ids)].copy()
    sub = sub.sort_values("timestamp")
    groups = {}
    for tid, g in sub.groupby("tracklet_id", sort=False):
        idx = g.index.to_numpy()
        if len(idx) >= min_len:
            groups[int(tid)] = idx
    return groups


class StitchingDataset(Dataset):
    """
    Produces:
      - anchor segment A (T points)
      - positive continuation segment B (T points) from same long tracklet, B after A
      - negative segment N (T points) from different tracklet
      - link feats for A->B and A->N
      - labels for BCE: y_pos=1, y_neg=0
    """

    def __init__(
        self,
        df: pd.DataFrame,
        groups: dict,  # tid -> sorted row indices
        T: int = 5,
        stride_choices=(1, 2, 3),
        same_sensor_neg: bool = True,
        same_class_neg: bool = False,
        pos_jitter_std: float = 0.01,
        scale_jitter_std: float = 0.05,
        drop_point_prob: float = 0.05,
        seed: int = 0,
    ):
        self.df = df
        self.groups = groups
        self.T = T
        self.stride_choices = stride_choices
        self.same_sensor_neg = same_sensor_neg
        self.same_class_neg = same_class_neg
        self.pos_jitter_std = pos_jitter_std
        self.scale_jitter_std = scale_jitter_std
        self.drop_point_prob = drop_point_prob
        self.rng = np.random.default_rng(seed)

        self.tids = np.array(list(self.groups.keys()), dtype=np.int64)

        # per-tracklet attributes to help negative sampling
        meta = df.groupby("tracklet_id").agg(
            sensor_id=("sensor_id", "first"),
            predicted_class=("predicted_class", "first"),
        )
        self.tid_to_sensor = meta["sensor_id"].to_dict()
        self.tid_to_class = meta["predicted_class"].to_dict()

    def __len__(self):
        # arbitrary epoch length; using number of long tracklets is fine
        return len(self.tids)

    def _get_rows(self, idxs):
        g = self.df.loc[idxs]
        # xy only
        xy = g[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)  # (T,2)
        ts = g["timestamp"].to_numpy(dtype=np.float32)  # (T,)
        return xy, ts

    def _sample_segment(self, tid):
        idx = self.groups[tid]
        L = len(idx)

        k = int(self.rng.choice(self.stride_choices))
        # need room for T points spaced by k
        max_start = L - (self.T - 1) * k
        if max_start <= 0:
            k = 1
            max_start = L - self.T + 1
        start = int(self.rng.integers(0, max_start))
        sel = idx[start : start + self.T * k : k]
        return self._get_rows(sel)

    def _sample_positive_pair(self, tid):
        """
        Sample A then sample B such that B starts after A ends, with a random gap.
        """
        idx = self.groups[tid]
        L = len(idx)

        # We will sample A by picking its indices, then force B to start later.
        # Use stride k for both for simplicity.
        k = int(self.rng.choice(self.stride_choices))
        # ensure A fits
        max_start_A = L - (self.T - 1) * k
        if max_start_A <= 0:
            k = 1
            max_start_A = L - self.T + 1

        start_A = int(self.rng.integers(0, max_start_A))
        sel_A = idx[start_A : start_A + self.T * k : k]

        # choose a B start index after the end of A, with a random gap in index units
        end_A_pos = start_A + (self.T - 1) * k
        # require at least one step forward; allow big gaps too (unbounded)
        min_start_B = end_A_pos + 1
        max_start_B = L - (self.T - 1) * k
        if min_start_B > max_start_B:
            # fallback: resample if this tracklet can't provide a forward continuation
            return self._sample_positive_pair(tid)

        start_B = int(self.rng.integers(min_start_B, max_start_B + 1))
        sel_B = idx[start_B : start_B + self.T * k : k]

        A = self._get_rows(sel_A)
        B = self._get_rows(sel_B)
        return A, B

    def _normalize_segment(self, xy, ts):
        """
        Normalize translation (subtract first point). Keep orientation.
        Also scale by path length for numeric stability (optional but recommended).
        Meta returned: [duration, mean_dt, path_len]
        """
        xy = xy - xy[0:1]  # translation removal

        dxy = np.diff(xy, axis=0)
        step = np.linalg.norm(dxy, axis=1)
        path_len = float(step.sum()) + 1e-6
        xy = xy / path_len

        duration = float(ts[-1] - ts[0])
        mean_dt = float(np.mean(np.diff(ts))) if len(ts) > 1 else 0.0
        meta = np.array([duration, mean_dt, path_len], dtype=np.float32)
        return xy, meta, ts

    def _augment_xy(self, xy):
        # small scale jitter (keeps orientation)
        if self.scale_jitter_std > 0:
            s = float(np.exp(self.rng.normal(0.0, self.scale_jitter_std)))
            xy = xy * np.float32(s)

        # small position jitter
        if self.pos_jitter_std > 0:
            xy = xy + self.rng.normal(0.0, self.pos_jitter_std, size=xy.shape).astype(
                np.float32
            )

        # occasionally drop a point (copy previous)
        if self.drop_point_prob > 0 and self.rng.random() < self.drop_point_prob:
            j = int(self.rng.integers(1, xy.shape[0]))
            xy[j] = xy[j - 1]

        return xy

    def _link_features(self, A_xy, A_ts, B_xy, B_ts):
        """
        Compute stitch link feats from the *end of A* to the *start of B* in normalized coords.
        """
        dt = float(B_ts[0] - A_ts[-1])
        # dt should be > 0 for valid stitching; but we'll be robust:
        dt_eps = dt if dt > 1e-6 else 1e-6

        dp = (B_xy[0] - A_xy[-1]).astype(np.float32)  # (2,)
        dist = float(np.linalg.norm(dp))
        implied_speed = dist / dt_eps

        vxvy = dp / np.float32(dt_eps)

        link = np.array(
            [
                np.log1p(max(dt, 0.0)),
                dp[0],
                dp[1],
                implied_speed,
                vxvy[0],
                vxvy[1],
            ],
            dtype=np.float32,
        )
        return link

    def _sample_negative_tid(self, anchor_tid):
        if not self.same_sensor_neg and not self.same_class_neg:
            while True:
                tid = int(self.rng.choice(self.tids))
                if tid != anchor_tid:
                    return tid

        anchor_sensor = self.tid_to_sensor[anchor_tid]
        anchor_class = self.tid_to_class[anchor_tid]

        candidates = []
        for tid in self.tids:
            tid = int(tid)
            if tid == anchor_tid:
                continue
            if self.same_sensor_neg and self.tid_to_sensor[tid] != anchor_sensor:
                continue
            if self.same_class_neg and self.tid_to_class[tid] != anchor_class:
                continue
            candidates.append(tid)

        if not candidates:
            # fallback: ignore constraints
            return self._sample_negative_tid(anchor_tid)

        return int(self.rng.choice(candidates))

    def __getitem__(self, i):
        tid = int(self.tids[i])

        # Positive A->B from same long tracklet
        (A_xy, A_ts), (B_xy, B_ts) = self._sample_positive_pair(tid)

        # Normalize
        A_xy, A_meta, A_ts = self._normalize_segment(A_xy, A_ts)
        B_xy, B_meta, B_ts = self._normalize_segment(B_xy, B_ts)

        # Augment (after normalization so noise is scale-consistent)
        A_xy = self._augment_xy(A_xy)
        B_xy = self._augment_xy(B_xy)

        link_pos = self._link_features(A_xy, A_ts, B_xy, B_ts)

        # Negative: A -> N from different tracklet
        neg_tid = self._sample_negative_tid(tid)
        N_xy, N_ts = self._sample_segment(neg_tid)
        N_xy, N_meta, N_ts = self._normalize_segment(N_xy, N_ts)
        N_xy = self._augment_xy(N_xy)

        link_neg = self._link_features(A_xy, A_ts, N_xy, N_ts)

        return {
            "loc_a": torch.from_numpy(A_xy),
            "meta_a": torch.from_numpy(A_meta),
            "loc_b": torch.from_numpy(B_xy),
            "meta_b": torch.from_numpy(B_meta),
            "loc_n": torch.from_numpy(N_xy),
            "meta_n": torch.from_numpy(N_meta),
            "link_pos": torch.from_numpy(link_pos),
            "link_neg": torch.from_numpy(link_neg),
        }
