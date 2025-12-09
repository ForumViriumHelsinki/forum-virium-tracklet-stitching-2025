import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, Polygon, MultiPolygon
from scipy.optimize import linear_sum_assignment


class TrackletStitcher:
    """
    Stitches short LiDAR tracklets into longer tracks using:
      - constant-velocity Kalman Filter in 2D
      - Hungarian algorithm for data association
      - termination based on max time gap, distance, and boundary polygon.

    Parameters
    ----------
    start_ts_col : str
        Column name for tracklet start timestamp (ms or seconds).
    end_ts_col : str
        Column name for tracklet end timestamp (ms or seconds).
    class_col : str
        Column for predicted class label (e.g. 'human', 'car').
    id_col : str
        Column for original tracklet ID.
    max_time_gap_s : float
        Maximum allowed gap (seconds) between end of one tracklet and start of
        a candidate continuation.
    max_dist_m : float
        Maximum Euclidean distance (meters) allowed between predicted position
        and candidate tracklet start position.
    process_noise_sigma : float
        Std dev of process acceleration noise (m/s^2) for the KF.
    meas_noise_sigma : float
        Std dev of measurement noise (m) for the KF.
    class_mismatch_penalty : float
        Cost penalty added if predicted_class differs between track and candidate.
    cost_threshold : float
        Maximum allowed association cost; above this, the match is rejected.
    time_bin_s : float
        Time bin size (seconds) used to group tracklets starting at similar times
        for Hungarian assignment.
    target_crs : int or str or None
        Target CRS EPSG code (or proj string) with metric units. If None,
        the CRS of the input GeoDataFrame is used as-is. If the input is
        geographic (degrees) and target_crs is None, an error is raised.
    """

    def __init__(
        self,
        start_ts_col="start_timestamp",
        end_ts_col="end_timestamp",
        class_col="predicted_class",
        id_col="id",
        max_time_gap_s=2.0,
        max_dist_m=5.0,
        process_noise_sigma=2.0,
        meas_noise_sigma=0.2,
        class_mismatch_penalty=50.0,
        cost_threshold=30.0,
        time_bin_s=0.5,
        target_crs=3067,  # ETRS-TM35FIN by default (good for Helsinki)
    ):
        self.start_ts_col = start_ts_col
        self.end_ts_col = end_ts_col
        self.class_col = class_col
        self.id_col = id_col

        self.max_time_gap_s = max_time_gap_s
        self.max_dist_m = max_dist_m
        self.process_noise_sigma = process_noise_sigma
        self.meas_noise_sigma = meas_noise_sigma
        self.class_mismatch_penalty = class_mismatch_penalty
        self.cost_threshold = cost_threshold
        self.time_bin_s = time_bin_s
        self.target_crs = target_crs

        # Measurement matrix: we observe position only: z = [px, py]
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def stitch(self, tracklets_gdf: gpd.GeoDataFrame, boundary_polygon=None):
        """
        Main entry point.

        Parameters
        ----------
        tracklets_gdf : GeoDataFrame
            GeoDataFrame of tracklets (one row per tracklet).
        boundary_polygon : shapely Polygon or MultiPolygon or GeoSeries or GeoDataFrame, optional
            Boundary of the junction in the same CRS (or reprojectable).
            If provided, candidate predictions/starts outside this polygon
            are rejected.

        Returns
        -------
        stitched_tracks : GeoDataFrame
            GeoDataFrame of stitched tracks with columns:
                - 'track_id' : new stitched track ID
                - 'source_ids' : list of original tracklet IDs
                - 'start_time', 'end_time' : seconds since epoch
                - 'predicted_class' : majority class along the track
                - 'n_tracklets' : number of tracklets merged
                - 'n_points' : sum of point_count (if present)
                - 'geometry' : merged LineString
        mapping_df : DataFrame
            Pandas DataFrame mapping original tracklet IDs to stitched track_id.
        """
        gdf = tracklets_gdf.copy()

        if gdf.crs is None:
            raise ValueError("Input GeoDataFrame must have a CRS set.")

        # Reproject to metric CRS if needed
        if self.target_crs is not None:
            gdf = gdf.to_crs(self.target_crs)
        else:
            if not gdf.crs.is_projected:
                raise ValueError(
                    "Input CRS is geographic (degrees). "
                    "Please set 'target_crs' to a metric CRS (e.g. 3067)."
                )

        # Prepare boundary polygon in the same CRS
        boundary_geom = None
        if boundary_polygon is not None:
            if isinstance(boundary_polygon, (Polygon, MultiPolygon)):
                boundary_geom = boundary_polygon
            elif isinstance(boundary_polygon, gpd.GeoSeries):
                boundary_geom = boundary_polygon.unary_union
                if boundary_polygon.crs != gdf.crs:
                    boundary_geom = (
                        gpd.GeoSeries([boundary_geom], crs=boundary_polygon.crs)
                        .to_crs(gdf.crs)
                        .iloc[0]
                    )
            elif isinstance(boundary_polygon, gpd.GeoDataFrame):
                boundary_geom = boundary_polygon.unary_union
                if boundary_polygon.crs != gdf.crs:
                    boundary_geom = (
                        gpd.GeoSeries([boundary_geom], crs=boundary_polygon.crs)
                        .to_crs(gdf.crs)
                        .iloc[0]
                    )
            else:
                raise ValueError("Unsupported type for boundary_polygon.")

        # Extract numeric features per tracklet
        feats = self._extract_tracklet_features(gdf)

        # Group tracklets by start time bin
        feats = feats.sort_values("start_time").reset_index(drop=True)
        feats["start_bin"] = (
            np.floor(feats["start_time"] / self.time_bin_s) * self.time_bin_s
        )

        # We will build tracks as a list of dicts
        tracks = []
        # Active tracks: dict track_id -> state
        active_tracks = {}

        next_track_id = 0

        # Helper: end times for gating / cleanup
        # (Not strictly needed in a separate structure, but clarifies logic.)

        for bin_val in sorted(feats["start_bin"].unique()):
            group = feats[feats["start_bin"] == bin_val]
            if group.empty:
                continue

            current_time = group["start_time"].min()

            # Clean up active tracks that are too old (time gap exceeded)
            to_remove = []
            for tid, tstate in active_tracks.items():
                gap = current_time - tstate["end_time"]
                if gap > self.max_time_gap_s:
                    to_remove.append(tid)
            for tid in to_remove:
                active_tracks.pop(tid, None)

            # If no active tracks, each tracklet in this group starts a new track
            if not active_tracks:
                for _, row in group.iterrows():
                    new_tid = next_track_id
                    next_track_id += 1
                    track = self._init_track(new_tid, row)
                    tracks.append(track)
                    active_tracks[new_tid] = track
                continue

            # Build cost matrix between active tracks and new group
            active_ids = list(active_tracks.keys())
            new_indices = list(group.index)

            n_active = len(active_ids)
            n_new = len(new_indices)

            cost_matrix = np.full((n_active, n_new), fill_value=1e9, dtype=float)

            for i_a, tid in enumerate(active_ids):
                track_state = active_tracks[tid]
                x = track_state["x"]
                P = track_state["P"]
                t_end = track_state["end_time"]
                track_class = track_state["class"]

                for j_n, idx in enumerate(new_indices):
                    row = feats.loc[idx]
                    dt = row["start_time"] - t_end
                    if dt <= 0 or dt > self.max_time_gap_s:
                        continue

                    # KF prediction to candidate start time
                    F = self._F(dt)
                    Q = self._Q(dt)
                    x_pred = F @ x
                    P_pred = F @ P @ F.T + Q

                    pred_px, pred_py = x_pred[0], x_pred[1]
                    start_px, start_py = row["start_x"], row["start_y"]

                    # Distance gate
                    dx = start_px - pred_px
                    dy = start_py - pred_py
                    dist = np.hypot(dx, dy)
                    if dist > self.max_dist_m:
                        continue

                    # Boundary gate (prediction AND start must be inside)
                    if boundary_geom is not None:
                        if not boundary_geom.contains(Point(pred_px, pred_py)):
                            continue
                        if not boundary_geom.contains(Point(start_px, start_py)):
                            continue

                    # Mahalanobis distance of innovation
                    z = np.array([start_px, start_py])
                    y = z - self.H @ x_pred  # innovation
                    S = self.H @ P_pred @ self.H.T + self._R()
                    try:
                        Sinv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        continue

                    d2 = float(y.T @ Sinv @ y)  # squared Mahalanobis

                    cost = d2

                    # Class mismatch penalty
                    if (track_class is not None) and (row["class"] is not None):
                        if str(track_class) != str(row["class"]):
                            cost += self.class_mismatch_penalty

                    cost_matrix[i_a, j_n] = cost

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Tracklets that are already matched
            matched_new = set()
            matched_active = set()

            for r, c in zip(row_ind, col_ind):
                c_cost = cost_matrix[r, c]
                if c_cost > self.cost_threshold:
                    continue  # treat as no match

                tid = active_ids[r]
                idx = new_indices[c]
                row = feats.loc[idx]
                matched_new.add(idx)
                matched_active.add(tid)

                # Update track with this tracklet using KF predict + update
                track_state = active_tracks[tid]

                # Predict to start of new tracklet
                dt1 = row["start_time"] - track_state["end_time"]
                F1 = self._F(dt1)
                Q1 = self._Q(dt1)
                x_pred = F1 @ track_state["x"]
                P_pred = F1 @ track_state["P"] @ F1.T + Q1

                # Update with measurement at start point
                z = np.array([row["start_x"], row["start_y"]])
                H = self.H
                S = H @ P_pred @ H.T + self._R()
                K = P_pred @ H.T @ np.linalg.inv(S)
                y = z - H @ x_pred
                x_upd = x_pred + K @ y
                P_upd = (np.eye(4) - K @ H) @ P_pred

                # Propagate to end_time of this new tracklet
                dt2 = row["end_time"] - row["start_time"]
                if dt2 < 0:
                    dt2 = 0.0
                F2 = self._F(dt2)
                Q2 = self._Q(dt2)
                x_end = F2 @ x_upd
                P_end = F2 @ P_upd @ F2.T + Q2

                # Update track state
                track_state["x"] = x_end
                track_state["P"] = P_end
                track_state["end_time"] = row["end_time"]
                track_state["tracklet_indices"].append(int(idx))
                track_state["source_ids"].append(row["orig_id"])
                track_state["classes"].append(row["class"])

            # Any new tracklet not matched -> start a new track
            for _, row in group.iterrows():
                idx = row.name
                if idx in matched_new:
                    continue
                new_tid = next_track_id
                next_track_id += 1
                track = self._init_track(new_tid, row)
                tracks.append(track)
                active_tracks[new_tid] = track

        stitched_records = []
        mapping_records = []

        for track in tracks:
            indices = track["tracklet_indices"]
            if not indices:
                continue
            sub = gdf.iloc[indices].sort_values(self.start_ts_col)

            # Merge geometries into one LineString (concatenate coordinates)
            coords = []
            for geom in sub.geometry:
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == "LineString":
                    xs, ys = geom.coords.xy
                    pts = list(zip(xs, ys))
                elif geom.geom_type == "Point":
                    pts = [(geom.x, geom.y)]
                else:
                    # Take first line of MultiLineString, etc.
                    try:
                        line = list(geom.geoms)[0]
                        xs, ys = line.coords.xy
                        pts = list(zip(xs, ys))
                    except Exception:
                        continue

                # Avoid duplicating the first point when concatenating
                if coords and pts:
                    if coords[-1] == pts[0]:
                        coords.extend(pts[1:])
                    else:
                        coords.extend(pts)
                else:
                    coords.extend(pts)

            if len(coords) < 2:
                # Fallback to a point track
                geom_out = Point(coords[0]) if coords else sub.geometry.iloc[0]
            else:
                from shapely.geometry import LineString

                geom_out = LineString(coords)

            start_time = sub[self.start_ts_col].min()
            end_time = sub[self.end_ts_col].max()

            # Convert to seconds if needed
            factor = 1000.0 if start_time > 1e10 else 1.0
            start_time_s = float(start_time / factor)
            end_time_s = float(end_time / factor)

            # Majority class
            classes = [c for c in track["classes"] if c is not None]
            if classes:
                pred_class = pd.Series(classes).mode().iloc[0]
            else:
                pred_class = None

            # Points count
            if "point_count" in sub.columns:
                n_points = int(sub["point_count"].sum())
            else:
                n_points = int(len(indices))  # fallback

            stitched_records.append(
                {
                    "track_id": track["track_id"],
                    "source_ids": track["source_ids"],
                    "start_time": start_time_s,
                    "end_time": end_time_s,
                    "predicted_class": pred_class,
                    "n_tracklets": len(indices),
                    "n_points": n_points,
                    "geometry": geom_out,
                }
            )

            for orig in track["source_ids"]:
                mapping_records.append(
                    {
                        "orig_id": orig,
                        "track_id": track["track_id"],
                    }
                )

        stitched_gdf = gpd.GeoDataFrame(
            stitched_records, geometry="geometry", crs=gdf.crs
        )
        mapping_df = pd.DataFrame(mapping_records).drop_duplicates()

        return stitched_gdf, mapping_df

    def _extract_tracklet_features(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Extract start/end positions, times, velocities, class from each row."""
        rows = []

        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            if geom.geom_type == "LineString":
                xs, ys = geom.coords.xy
                start_x, start_y = xs[0], ys[0]
                end_x, end_y = xs[-1], ys[-1]
            elif geom.geom_type == "Point":
                start_x = end_x = geom.x
                start_y = end_y = geom.y
            else:
                # e.g. MultiLineString: take first component
                try:
                    line = list(geom.geoms)[0]
                    xs, ys = line.coords.xy
                    start_x, start_y = xs[0], ys[0]
                    end_x, end_y = xs[-1], ys[-1]
                except Exception:
                    continue

            t_start = float(row[self.start_ts_col])
            t_end = float(row[self.end_ts_col])

            # Convert ms -> seconds if needed
            factor = 1000.0 if t_start > 1e10 else 1.0
            t_start_s = t_start / factor
            t_end_s = t_end / factor

            dt = max(t_end_s - t_start_s, 1e-3)
            vx = (end_x - start_x) / dt
            vy = (end_y - start_y) / dt

            rows.append(
                {
                    "idx": int(idx),
                    "orig_id": row.get(self.id_col, idx),
                    "start_time": t_start_s,
                    "end_time": t_end_s,
                    "start_x": start_x,
                    "start_y": start_y,
                    "end_x": end_x,
                    "end_y": end_y,
                    "vx": vx,
                    "vy": vy,
                    "class": row.get(self.class_col, None),
                }
            )

        return pd.DataFrame(rows)

    def _init_track(self, track_id: int, row: pd.Series):
        """Initialize a new track from a single tracklet row (from feats)."""
        # State at end of tracklet
        px, py = row["end_x"], row["end_y"]
        vx, vy = row["vx"], row["vy"]
        x0 = np.array([px, py, vx, vy], dtype=float)

        # Initial covariance: position quite certain, velocity less so
        P0 = np.diag([1.0, 1.0, 10.0, 10.0])

        return {
            "track_id": track_id,
            "x": x0,
            "P": P0,
            "end_time": row["end_time"],
            "tracklet_indices": [int(row["idx"])],
            "source_ids": [row["orig_id"]],
            "class": row["class"],
            "classes": [row["class"]],
        }

    def _F(self, dt: float) -> np.ndarray:
        """State transition matrix for constant-velocity model."""
        return np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    def _Q(self, dt: float) -> np.ndarray:
        """Process noise covariance for constant-acceleration driving constant-velocity state."""
        a2 = self.process_noise_sigma**2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        q11 = 0.25 * dt4 * a2
        q13 = 0.5 * dt3 * a2
        q33 = dt2 * a2

        Q = np.array(
            [
                [q11, 0, q13, 0],
                [0, q11, 0, q13],
                [q13, 0, q33, 0],
                [0, q13, 0, q33],
            ],
            dtype=float,
        )
        return Q

    def _R(self) -> np.ndarray:
        """Measurement noise covariance."""
        r2 = self.meas_noise_sigma**2
        return np.array([[r2, 0], [0, r2]], dtype=float)
