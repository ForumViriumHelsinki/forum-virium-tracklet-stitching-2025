from __future__ import annotations

import json
import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# ---- Stage 1: initial tracklet builder ----
from .stages.tracklet_builder import (
    build_tracklets_hungarian,
)

# ---- Stage 2: pre-stitch (overlap-based union + re-smooth) ----
from .stages.pre_stitcher import pre_stitch_tracklets

# ---- Stage 3: main stitcher ----
from .stages.main_stitcher import (
    stitch_tracklets_hungarian,
    CLASS_PARAMS,
)

# ---- Stage 4: final post stitching linker ----
from .stages.post_stitcher import (
    final_post_stitch_linker,
    POST_LINKER_CLASS_PARAMS_DEFAULT,
)

from .geo.utils import (
    GeoRefConfig,
    export_qgis_points_csv,
)
from .utils import (
    read_raw_log_csv,
    parse_raw_tracking_df,
    RawLogConfig,
    plot_entry_exit_matrix,
    plot_track_length_distributions,
    build_geofence_maps,
    point_in_poly,
)
from .track_timeline_visualiser import track_timeline_visualizer
from .track_visualiser import track_visualizer

OUTPUT_ROOT = "track_builder_outputs"

REQUIRED_INPUT_COLUMNS = {"timestamp", "centroid_x", "centroid_y", "time_delta_prev"}


# -------- Stage 1: Tracklet building --------
@dataclass
class Stage1Params:
    v_max: float = 15.0
    a_max: float = 5.0
    max_gap_seconds: float = 0.25
    min_tracklet_duration: float = 0.3
    min_tracklet_points: int = 3
    direction_weight: float = 1.5
    min_speed_for_dir: float = 0.7
    meas_noise_pos: float = 0.25
    vel_gate_min_hits: int = 3
    use_kinematic_gate: bool = True
    use_mahalanobis_gate: bool = True
    maha_gamma: float = 5.99  # 2.30, 5.99, 9.21 for 68%, 95%, 99% in 2D
    time_eps: float = 0.001
    bucket_mode: str = "round"
    do_rts_smoothing: bool = True
    rts_sigma_a: float = 3.0


# -------- Stage 2: Pre-stitch (overlap-time merge) --------
@dataclass
class Stage2Params:
    dt_thr: float = 0.25
    dist_thr: float = 0.50
    prop_thr: float = 0.75
    min_vel_points: int = 3
    speed_thr: float = 2.0
    angle_thr_deg: float = 25.0
    min_overlap_points: int = 2
    meas_noise_pos: float = 0.25
    rts_sigma_a: float = 3.0
    time_eps: float = 0.001


# -------- Stage 3: Main stitch (KF replay + gates) --------
@dataclass
class Stage3Params:
    geofence_path: Optional[str] = None  # geofence_path injected at runtime
    area_name: str = "general_area"
    tracklet_id_col: str = "tracklet_id"
    max_stitch_length: float = 90.0
    direction_weight: float = 1.5
    min_speed_for_dir: float = 0.5
    use_kinematic_gate: bool = True
    use_mahalanobis_gate: bool = True
    maha_gamma: float = 9.21  # 2.30, 5.99, 9.21 for 68%, 95%, 99% in 2D
    sigma_a: float = 3.0
    replay_stride: int = 1
    terminate_on_leave_area: bool = True


# -------- Stage 4: Final post linker (no prediction) --------
@dataclass
class Stage4Params:
    id_col: str = "stitched_id"
    class_col: str = "predicted_class"
    time_col: str = "timestamp"
    x_col: str = "centroid_x"
    y_col: str = "centroid_y"
    write_rts_cols: bool = True
    require_margin: bool = True
    margin_ratio: float = 1.4
    time_eps: float = 1e-3
    min_track_length_m: float = 5.0
    min_track_displacement_m: float = 5.0


@dataclass
class PipelineParams:
    csv_write_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(index=False))
    stage1: Stage1Params = field(default_factory=Stage1Params)
    stage2: Stage2Params = field(default_factory=Stage2Params)
    stage3: Stage3Params = field(default_factory=Stage3Params)
    stage4: Stage4Params = field(default_factory=Stage4Params)


PIPELINE_PARAMS = PipelineParams()

# Ratastie: sensor at (60.197547, 24.907931), rotated -45 deg
# Kontula: sensor at (60.239518, 25.080891), rotated -243 deg
# Runeberg-Dobeln: sensor at (60.178113, 24.922863), rotated -180 deg
# Sturenkatu: sensor at (60.196490, 24.960735), rotated -35 deg
# Katajanokka: sensor at (60.166190, 24.964635), rotated -45 deg

# NOTE: TUNE THIS PER SCENE
GEOREF_PARAMS = GeoRefConfig(
    sensor_lat=60.196490,
    sensor_lon=24.960735,
    rotation_deg=0.0,
    swap_xy=False,
    flip_x=False,
    flip_y=False,
    earth_radius_m=6378137.0,  # WGS84 sphere approx (good enough for small areas)
)


def _validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_INPUT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


def _make_run_dir(base_dir: Path) -> Path:
    # second-level timestamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def _log_line(fp, msg: str) -> None:
    print(msg)
    fp.write(msg + "\n")
    fp.flush()


def run_pipeline(
    input_path: Path,
    output_root: Path,
    geofence_path: Optional[Path] = None,
    params: PipelineParams = PIPELINE_PARAMS,
) -> pd.DataFrame:
    def _ts_range(d: pd.DataFrame, col: str = "timestamp") -> str:
        if col not in d.columns or len(d) == 0:
            return "n/a"
        t0 = float(pd.to_numeric(d[col], errors="coerce").min())
        t1 = float(pd.to_numeric(d[col], errors="coerce").max())
        if not (pd.notna(t0) and pd.notna(t1)):
            return "n/a"
        return f"{t0:.3f} → {t1:.3f} (Δ {t1 - t0:.3f}s)"

    def _n_ids(d: pd.DataFrame, col: str) -> int:
        if col not in d.columns or len(d) == 0:
            return 0
        v = pd.to_numeric(d[col], errors="coerce")
        v = v[(v.notna()) & (v >= 0)]
        return int(v.nunique())

    run_dir = _make_run_dir(output_root)
    log_path = run_dir / "run.log"
    cfg_path = run_dir / "run_config.json"

    # Define all output paths INSIDE run_dir
    output_csv_path = run_dir / "output_tracks.csv"
    qgis_points_csv_path = run_dir / "output_tracks_qgis_points.csv"

    # materialize the actual kwargs that will be used (including injected geofence_path)
    stage3 = asdict(params.stage3)
    stage3["geofence_path"] = None if geofence_path is None else str(geofence_path)

    used_config = {
        "input_path": str(input_path),
        "geofence_path": None if geofence_path is None else str(geofence_path),
        "outputs": {
            "run_dir": str(run_dir),
            "output_csv": str(output_csv_path),
            "qgis_points_csv": str(qgis_points_csv_path),
        },
        "csv_write_kwargs": params.csv_write_kwargs,
        "stage1_kwargs": asdict(params.stage1),
        "stage2_kwargs": asdict(params.stage2),
        "stage3_kwargs": stage3,
        "stage4_kwargs": asdict(params.stage4),
        "georef_params": asdict(GEOREF_PARAMS),
        "class_params_name": "CLASS_PARAMS",
        "post_linker_class_params_name": "POST_LINKER_CLASS_PARAMS_DEFAULT",
    }
    _write_json(cfg_path, used_config)

    with open(log_path, "w", encoding="utf-8") as lf:
        _log_line(lf, "=" * 80)
        _log_line(lf, "[pipeline] start")
        _log_line(lf, f"[pipeline] run_dir={run_dir}")
        _log_line(lf, f"[pipeline] input_path={input_path}")
        _log_line(
            lf,
            f"[pipeline] geofence_path={None if geofence_path is None else str(geofence_path)}",
        )
        _log_line(lf, f"[pipeline] wrote config: {cfg_path}")
        _log_line(lf, "=" * 80)

        # ---- Load ----
        _log_line(lf, "[load] reading raw log csv...")
        cfg = RawLogConfig(topic_suffix="tracking", objects_key="objects")
        df_raw = read_raw_log_csv(input_path, cfg=cfg)
        _log_line(lf, f"[load] df_raw: rows={len(df_raw):,} cols={len(df_raw.columns)}")

        _log_line(lf, "[load] parsing tracking df...")

        df = parse_raw_tracking_df(df_raw, cfg=cfg)
        # Crop to general_area if geofence provided
        if geofence_path is not None:
            with open(str(geofence_path), "r", encoding="utf-8") as f:
                geofences = json.load(f)

            area_poly, _ = build_geofence_maps(
                geofences, area_name=params.stage3.area_name  # e.g. "general_area"
            )

            xs = pd.to_numeric(df["centroid_x"], errors="coerce")
            ys = pd.to_numeric(df["centroid_y"], errors="coerce")

            in_area = [
                (
                    point_in_poly(float(x), float(y), area_poly)
                    if pd.notna(x) and pd.notna(y)
                    else False
                )
                for x, y in zip(xs, ys)
            ]
            df = df[pd.Series(in_area, index=df.index)].copy()

        _log_line(
            lf, f"[load] df: rows={len(df):,} cols={len(df.columns)} ts={_ts_range(df)}"
        )

        _log_line(lf, "[validate] checking required columns...")
        _validate_input(df)
        _log_line(lf, "[validate] ok")

        # ---- Stage 1 ----
        _log_line(lf, "-" * 80)
        _log_line(lf, "[stage1] build_tracklets_hungarian: start")
        _log_line(lf, f"[stage1] kwargs={asdict(params.stage1)}")
        df1 = build_tracklets_hungarian(df, **asdict(params.stage1))
        n_tr = _n_ids(df1, "tracklet_id")
        assigned = (
            int((pd.to_numeric(df1["tracklet_id"], errors="coerce") >= 0).sum())
            if "tracklet_id" in df1.columns
            else 0
        )
        _log_line(
            lf,
            f"[stage1] done: rows={len(df1):,} tracklets={n_tr:,} assigned_rows={assigned:,} ts={_ts_range(df1)}",
        )

        # ---- Stage 2 ----
        _log_line(lf, "-" * 80)
        _log_line(lf, "[stage2] pre_stitch_tracklets: start")
        _log_line(lf, f"[stage2] kwargs={asdict(params.stage2)}")
        df2 = pre_stitch_tracklets(df1, **asdict(params.stage2))
        n_tr2 = _n_ids(df2, "tracklet_id")
        _log_line(
            lf,
            f"[stage2] done: rows={len(df2):,} tracklets={n_tr2:,} delta_tracklets={n_tr2 - n_tr:+,} ts={_ts_range(df2)}",
        )

        # ---- Stage 3 ----
        _log_line(lf, "-" * 80)
        _log_line(lf, "[stage3] stitch_tracklets_hungarian: start")
        _log_line(lf, f"[stage3] kwargs={stage3}")
        df3 = stitch_tracklets_hungarian(
            df2,
            class_params=CLASS_PARAMS,
            **stage3,
        )
        n_st = _n_ids(df3, "stitched_id")
        _log_line(
            lf,
            f"[stage3] done: rows={len(df3):,} stitched_tracks={n_st:,} ts={_ts_range(df3)}",
        )

        # ---- Stage 4 ----
        _log_line(lf, "-" * 80)
        _log_line(lf, "[stage4] final_post_stitch_linker: start")
        _log_line(lf, f"[stage4] kwargs={asdict(params.stage4)}")
        df4 = final_post_stitch_linker(
            df3,
            class_params=POST_LINKER_CLASS_PARAMS_DEFAULT,
            **asdict(params.stage4),
        )
        n_final = _n_ids(df4, params.stage4.id_col)
        _log_line(
            lf,
            f"[stage4] done: rows={len(df4):,} final_ids({params.stage4.id_col})={n_final:,} ts={_ts_range(df4)}",
        )

        # ---- QGIS export ----
        _log_line(lf, "-" * 80)
        _log_line(lf, "[export] qgis points csv: start")
        _log_line(lf, f"[export] path={qgis_points_csv_path}")
        export_qgis_points_csv(
            df4,
            qgis_points_csv_path,
            cfg=GEOREF_PARAMS,
            x_col="rts_x",
            y_col="rts_y",
            fallback_x="centroid_x",
            fallback_y="centroid_y",
        )
        _log_line(lf, "[export] done")

        # ---- Save ----
        _log_line(lf, "-" * 80)
        _log_line(lf, "[save] writing output csv...")
        df4.to_csv(output_csv_path, **params.csv_write_kwargs)
        _log_line(lf, f"[save] wrote {output_csv_path}")

        # ---- Visualize ----
        _log_line(lf, "-" * 80)
        _log_line(lf, "[visualize] creating timeline visualization HTML...")
        track_timeline_visualizer(run_dir=run_dir)
        _log_line(
            lf,
            f"[visualize] wrote timeline visualization HTML to {run_dir / 'tracklets_timeline.html'}",
        )
        _log_line(lf, "[visualize] creating visualization HTML...")
        track_visualizer(run_dir=run_dir)
        _log_line(
            lf,
            f"[visualize] wrote visualization HTML to {run_dir / 'tracklets_timeline.html'}",
        )

        # ---- Plotting ----
        _log_line(lf, "-" * 80)
        _log_line(lf, "[plot] entry-exit matrix...")
        plot_entry_exit_matrix(df4, run_dir=run_dir)
        _log_line(lf, "[plot] track length distributions...")
        plot_track_length_distributions(df4, run_dir=run_dir)
        _log_line(lf, "[plot] done")

        _log_line(lf, "=" * 80)
        _log_line(lf, "[pipeline] done")
        _log_line(lf, "=" * 80)

    return df4


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-csv", type=Path, required=True, help="Raw log CSV to process"
    )
    p.add_argument(
        "--geofence-json", type=Path, required=True, help="Geofence JSON file"
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path(OUTPUT_ROOT),
        help=f"Output directory root (default: {OUTPUT_ROOT})",
    )
    args = p.parse_args(argv)

    run_pipeline(
        input_path=args.input_csv,
        output_root=args.output_root,
        geofence_path=args.geofence_json,
        params=PIPELINE_PARAMS,  # edit PIPELINE_PARAMS centrally above
    )


if __name__ == "__main__":
    main()
