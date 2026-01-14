# Forum Virium Tracklet Stitching 2025

## Recommended structure

```sh
├── data
│   ├── fences
│       └── <location_folders>
│   ├── fgb
│       └── <FGB_FILES>
│   └── raw
│       └── <CSV_FILES>
├── track_builder
│   ├── __init__.py
│   ├── __pycache__
│   ├── association
│   ├── geo
│   ├── kf
│   ├── stages
│   ├── main.py
│   ├── geo_fence_drawer.py
│   ├── track_timeline_visualiser.py
│   ├── track_visualiser.py
│   └── utils.py
├── track_builder_outputs
│   └── <output_folders>
├── pyproject.toml
├── uv.lock
├── README.md
├── __init__.py
```

## Setup


- Python (version specified in `pyproject.toml`)
- uv

Install uv if needed:

```sh
pip install uv
```

Sync uv. This will setup virtual env and install dependencies.
```sh
uv sync
```


## Usage

This project provides an offline multi-object tracking (MOT) pipeline that takes raw per-frame detections, builds short tracklets, stitches them into long trajectories, and produces analysis-ready outputs and visualizations.

The pipeline is designed to be run end-to-end from the command line and writes all results into a self-contained, timestamped run directory.

### 1. Required inputs

You need two inputs:

1. Raw log CSV  
   A CSV containing per-frame detections. After parsing, it must contain at least:
   - `timestamp` (float, UNIX seconds)
   - `time_delta_prev`(float, UNIX seconds)
   - `centroid_x`, `centroid_y` (positions in sensor / LiDAR coordinates)

   The provided loader expects a log-style CSV and parses it using `RawLogConfig`.

2. Correct Geo-information **IMPORTANT**
   In the `main.py` inside `track_builder` you must modify this object for your scene
   ```sh
   GEOREF_PARAMS = GeoRefConfig(
        sensor_lat=60.196490,
        sensor_lon=24.960735,
        rotation_deg=0.0,
        swap_xy=False,
        flip_x=False,
        flip_y=False,
        earth_radius_m=6378137.0,  # WGS84 sphere approx (good enough for small areas)
    )
    ```

   The provided loader expects a log-style CSV and parses it using `RawLogConfig`.

3. Geofence JSON  
   A JSON file defining polygons in sensor coordinates. It must contain:
   - A polygon named `general_area` (used to crop data and terminate tracks).
   - Optional additional polygons (“wells”) used for entry/exit labeling.

   Polygons are defined as lists of `[x, y]` vertices.


### 2. Running the pipeline

The main entry point is `mot.main`.

```bash
uv run python -m track_builder.main \
  --input-csv path/to/raw_log.csv \
  --geofence-json path/to/geofence.json \
  --output-root track_builder_outputs
```
Arguments:

- `--input-csv` (required)  
  Path to the raw log CSV file.

- `--geofence-json` (required)  
  Path to the geofence JSON file.

\
**NOTE: DO NOT CHANGE WITHOUT PURPOSE** 
- `--output-root` (optional)\
  Directory under which a timestamped run folder is created.  
  Default: `track_builder_outputs/`

Each execution creates a new run directory.

### 3. What the pipeline does (stages)

The pipeline executes four sequential stages:

1. **Stage 1 – Tracklet building**
   - Associates per-frame detections into short, high-confidence tracklets.
   - Uses Kalman filtering, kinematic and Mahalanobis gating, and Hungarian assignment.
   - Optionally applies RTS smoothing per tracklet.

2. **Stage 2 – Pre-stitching**
   - Merges overlapping tracklets that represent the same object.
   - Uses time overlap, spatial consistency, and velocity agreement.
   - Re-smooths merged tracklets.

3. **Stage 3 – Main stitching**
   - Links tracklets end-to-start into long trajectories.
   - Uses class-aware motion models, gating, and Hungarian assignment.
   - Applies geofence logic (area containment, entry/exit wells).

4. **Stage 4 – Final post-stitch linking**
   - A conservative final pass without prediction.
   - Links remaining fragmented tracks based on boundary distances and direction.
   - Re-smooths merged results and filters very short or low-displacement tracks.

All parameters for these stages are centrally defined in `PIPELINE_PARAMS` and can be tuned per scene.

### 4. Outputs

Each run directory contains a complete, reproducible record of the run:

```
run_YYYYMMDD_HHMMSS/
├── run.log
├── run_config.json
├── output_tracks.csv
├── output_tracks_qgis_points.csv
├── tracklets_timeline.html
├── tracks.html
├── entry_exit_matrix.png
├── track_length_distributions.png
```

- **`run.log`**  
  Human-readable log of every pipeline step, including counts and timing.

- **`run_config.json`**  
  Full snapshot of:
  - Input paths
  - All stage parameters
  - Georeferencing configuration  
  This guarantees reproducibility.

- **`output_tracks.csv`**  
  Main machine-readable result. Contains:
  - Original detections
  - `tracklet_id` (Stage 1 / 2)
  - `stitched_id` (final trajectory ID)
  - Optional RTS-smoothed columns:  
    `rts_x`, `rts_y`, `rts_vx`, `rts_vy`, `rts_speed`
  - Entry/exit well labels (if defined)

- **`output_tracks_qgis_points.csv`**  
  Point-level CSV converted to WGS84 latitude/longitude.  
  Intended for direct import into QGIS.

- **`tracklets_timeline.html`**  
  Interactive Folium timeline visualization:
  - Tracks animated over time
  - Color-coded by ID
  - Inspectable metadata per track

- **`tracks.html`**  
  Static map visualization:
  - Full trajectories as polylines
  - Single-point and unmatched detections shown separately

- **Plots (`*.png`)**
  - Entry–exit matrix
  - Track length distributions


### 5. Geofence drawing tool

Typical usage:
```
uv run python -m bokeh serve --show ./track_builder/geo_fence_drawer.py
```

**Features:**
- Load raw CSV point clouds
- Draw, edit, and name polygons
- Save directly to the JSON format expected by the pipeline

### 6. Typical workflow

1. Prepare raw detection CSV.
2. Create or edit geofences using the drawing tool.
3. Run the pipeline with `python -m mot.main`.
4. Inspect results in:
   - `output_tracks.csv` for analysis
   - HTML visualizations for qualitative validation
   - QGIS for spatial analysis

This design allows fully offline, reproducible MOT processing with strong introspection at every stage.

### 7. Pipeline parameters (overview)

All stage parameters are centrally defined and managed in `main.py` via the
`PIPELINE_PARAMS` object. Each stage has its own parameter dataclass, and values
can be tuned per scene before running the pipeline.

---

#### Stage 1 – Tracklet building (`Stage1Params`)

Controls how per-frame detections are associated into short tracklets.

- `v_max`  
  Maximum expected speed (m/s) used for kinematic gating.

- `a_max`  
  Maximum expected acceleration (m/s²) used for kinematic gating.

- `max_gap_seconds`  
  Maximum allowed time gap between detections within a tracklet.

- `min_tracklet_duration`  
  Minimum duration (seconds) required for a tracklet to be kept.

- `min_tracklet_points`  
  Minimum number of detections required to confirm a tracklet.

- `direction_weight`  
  Weight applied to directional disagreement in the assignment cost.

- `min_speed_for_dir`  
  Minimum speed required before directional penalties are applied.

- `meas_noise_pos`  
  Measurement noise standard deviation (meters) for position updates.

- `vel_gate_min_hits`  
  Number of points required before velocity-based gating is enabled.

- `use_kinematic_gate`  
  Enable or disable kinematic distance gating.

- `use_mahalanobis_gate`  
  Enable or disable Mahalanobis-distance gating.

- `maha_gamma`  
  Mahalanobis distance threshold (χ² value for 2D).

- `time_eps`  
  Small epsilon to stabilize time-difference computations.

- `bucket_mode`  
  Timestamp bucketing mode (e.g. `round`) for grouping detections.

- `do_rts_smoothing`  
  If true, applies RTS smoothing to each finalized tracklet.

- `rts_sigma_a`  
  Process noise (acceleration) used during RTS smoothing.

---

#### Stage 2 – Pre-stitching (`Stage2Params`)

Controls merging of overlapping tracklets before long-range stitching.

- `dt_thr`  
  Maximum allowed time difference between overlapping points.

- `dist_thr`  
  Maximum spatial distance (meters) for overlap matching.

- `prop_thr`  
  Minimum proportion of overlapping points required to merge tracklets.

- `min_vel_points`  
  Minimum number of points used to estimate velocities for comparison.

- `speed_thr`  
  Maximum allowed speed difference (m/s) between overlapping tracklets.

- `angle_thr_deg`  
  Maximum allowed direction angle difference (degrees).

- `min_overlap_points`  
  Minimum number of overlapping points required to consider a merge.

- `meas_noise_pos`  
  Measurement noise used when re-smoothing merged tracklets.

- `rts_sigma_a`  
  Process noise (acceleration) for RTS re-smoothing.

- `time_eps`  
  Numerical stability epsilon for time comparisons.

---

#### Stage 3 – Main stitching (`Stage3Params`)

Controls linking of tracklets into long trajectories.

- `geofence_path`  
  Path to geofence JSON file (injected at runtime).

- `area_name`  
  Name of the main area polygon used for containment checks.

- `tracklet_id_col`  
  Column name containing tracklet IDs.

- `max_stitch_length`  
  Maximum allowed trajectory length (meters) before termination.

- `direction_weight`  
  Weight applied to directional disagreement in stitching costs.

- `min_speed_for_dir`  
  Minimum speed required to enforce directional constraints.

- `use_kinematic_gate`  
  Enable or disable kinematic reachability gating.

- `use_mahalanobis_gate`  
  Enable or disable Mahalanobis-distance gating.

- `maha_gamma`  
  Mahalanobis distance threshold for stitching (χ² value).

- `sigma_a`  
  Process noise (acceleration) for Kalman filter replay.

- `replay_stride`  
  Subsampling stride when replaying tracklet points into the filter.

- `terminate_on_leave_area`  
  If true, tracks are terminated when leaving the defined area polygon.

---

#### Stage 4 – Final post-stitch linking (`Stage4Params`)

Controls the final conservative linking and filtering pass.

- `id_col`  
  Column name containing trajectory IDs to be post-linked.

- `class_col`  
  Column name containing object class labels.

- `time_col`  
  Timestamp column name.

- `x_col`, `y_col`  
  Position column names used for distance checks.

- `write_rts_cols`  
  If true, writes RTS-smoothed columns for merged tracks.

- `require_margin`  
  Require a clear cost margin between best and second-best matches.

- `margin_ratio`  
  Margin ratio used when `require_margin` is enabled.

- `time_eps`  
  Numerical stability epsilon for time comparisons.

- `min_track_length_m`  
  Minimum total path length (meters) required to keep a track.

- `min_track_displacement_m`  
  Minimum start-to-end displacement (meters) required to keep a track.

---

#### Global pipeline parameters

- `csv_write_kwargs`  
  Keyword arguments passed to `DataFrame.to_csv` when writing outputs.

All parameters can be modified in `PIPELINE_PARAMS` before running the pipeline,
making scene-specific tuning explicit and reproducible.