import json
from pathlib import Path

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColumnDataSource,
    Div,
    PolyDrawTool,
    PolyEditTool,
    HoverTool,
    Select,
    TextInput,
)
from bokeh.plotting import figure

from utils import read_raw_log_csv, parse_raw_tracking_df, RawLogConfig

# ---- config ----
CFG = RawLogConfig()
DATA_ROOT = Path("data/raw")
DEFAULT_CSV = Path("data/raw/ratsastie_20250813_0706.csv")
DEFAULT_JSON = Path("data/fences/ratsastie/base.json")
N_PLOT, POINT_SIZE, PAD = 100_000, 2, 5.0


def list_data_files():
    if not DATA_ROOT.exists():
        return []
    return [str(p.as_posix()) for p in sorted(DATA_ROOT.rglob("*.csv"))]


csv_opts = list_data_files()
initial_csv = (
    str(DEFAULT_CSV.as_posix())
    if DEFAULT_CSV.exists()
    else (csv_opts[0] if csv_opts else str(DEFAULT_CSV))
)

# ---- widgets ----
csv_select = Select(
    title="CSV (under data/raw/):",
    value=initial_csv,
    options=csv_opts if csv_opts else [initial_csv],
    width=900,
)
csv_input = TextInput(title="CSV override (optional):", value="", width=900)

json_input = TextInput(
    title="Polygons JSON path:", value=str(DEFAULT_JSON.as_posix()), width=900
)

reload_btn = Button(label="Reload", button_type="primary", width=120)
save_btn = Button(label="Save", button_type="success", width=120)
load_btn = Button(label="Load", width=120)

poly_select = Select(title="Polygon:", value="(none)", options=["(none)"], width=300)
name_input = TextInput(title="Polygon name:", value="", width=300)
set_name_btn = Button(label="Set name", width=120)

msg = Div(width=900)
status = Div(width=900)

# ---- sources ----
pts = ColumnDataSource(data=dict(centroid_x=[], centroid_y=[]))
poly = ColumnDataSource(data=dict(xs=[], ys=[], name=[]))
verts = ColumnDataSource(data=dict(x=[], y=[]))

# ---- plot ----
p = figure(
    width=900,
    height=900,
    x_range=(-10, 10),
    y_range=(-10, 10),
    match_aspect=True,
    title="Draw geofences",
)
p.scatter("centroid_x", "centroid_y", source=pts, size=POINT_SIZE, alpha=0.35)
patches = p.patches(xs="xs", ys="ys", source=poly, fill_alpha=0.15, line_width=2)
patches = p.patches(xs="xs", ys="ys", source=poly, fill_alpha=0.15, line_width=2)

hover = HoverTool(
    renderers=[patches],
    tooltips=[
        ("name", "@name"),
        ("index", "$index"),
    ],
)
p.add_tools(hover)
vr = p.scatter("x", "y", source=verts, size=10, alpha=0.9)

draw = PolyDrawTool(renderers=[patches])
edit = PolyEditTool(renderers=[patches], vertex_renderer=vr)
p.add_tools(draw, edit)
p.toolbar.active_tap = draw


def csv_path():
    t = csv_input.value.strip()
    return Path(t) if t else Path(csv_select.value)


def json_path():
    t = json_input.value.strip()
    return Path(t) if t else DEFAULT_JSON


def update_poly_selector():
    n = len(poly.data.get("xs", []))
    opts = ["(none)"] + [f"{i}" for i in range(n)]
    poly_select.options = opts
    if poly_select.value not in opts:
        poly_select.value = "(none)"


def update_status():
    n = len(poly.data.get("xs", []))
    status.text = (
        f"<b>Polygons:</b> {n} &nbsp;&nbsp; <code>{json_path().as_posix()}</code>"
    )
    update_poly_selector()


def set_name():
    if poly_select.value == "(none)":
        msg.text = "ℹ️ Select a polygon index first."
        return
    i = int(poly_select.value)

    data = dict(poly.data)
    names = list(data.get("name", []))
    while len(names) < len(data.get("xs", [])):
        names.append("")  # explicit: blank until you set it
    names[i] = name_input.value
    data["name"] = names
    poly.data = data
    msg.text = f"✅ Set name for polygon {i} to <code>{name_input.value}</code>"
    update_status()


def save_json():
    xs, ys = poly.data.get("xs", []), poly.data.get("ys", [])
    names = list(poly.data.get("name", []))
    while len(names) < len(xs):
        names.append("")  # keep explicit blanks if user didn't name

    zones = []
    for i in range(len(xs)):
        verts_xy = [[float(x), float(y)] for x, y in zip(xs[i], ys[i])]
        zones.append({"id": i, "name": names[i], "vertices_xy": verts_xy})

    out = json_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(zones, indent=2), encoding="utf-8")
    msg.text = f"✅ Saved {len(zones)} polygon(s) to <code>{out.as_posix()}</code>"
    update_status()


def load_json():
    out = json_path()
    if not out.exists():
        msg.text = f"ℹ️ No polygons at <code>{out.as_posix()}</code>"
        update_status()
        return

    zones = json.loads(out.read_text(encoding="utf-8"))
    xs, ys, names = [], [], []
    for z in zones:
        v = z.get("vertices_xy", [])
        xs.append([pt[0] for pt in v])
        ys.append([pt[1] for pt in v])
        names.append(z.get("name", ""))  # explicit: empty if missing

    poly.data = dict(xs=xs, ys=ys, name=names)
    msg.text = f"✅ Loaded {len(xs)} polygon(s) from <code>{out.as_posix()}</code>"
    update_status()


def reload():
    path = csv_path()
    if not path.exists():
        msg.text = f"❌ Missing CSV: <code>{path.as_posix()}</code>"
        return

    df = read_raw_log_csv(path, cfg=CFG)
    df = parse_raw_tracking_df(df, cfg=CFG)
    need = {"centroid_x", "centroid_y"}
    if not need.issubset(df.columns):
        msg.text = f"❌ CSV missing columns: {', '.join(sorted(need))}"
        return

    use = df[["centroid_x", "centroid_y"]].dropna()
    samp = use.sample(min(N_PLOT, len(use)), random_state=0) if len(use) else use

    pts.data = dict(
        centroid_x=samp["centroid_x"].tolist(),
        centroid_y=samp["centroid_y"].tolist(),
    )

    if len(samp):
        p.x_range.start, p.x_range.end = (
            float(samp.centroid_x.min()) - PAD,
            float(samp.centroid_x.max()) + PAD,
        )
        p.y_range.start, p.y_range.end = (
            float(samp.centroid_y.min()) - PAD,
            float(samp.centroid_y.max()) + PAD,
        )

    load_json()  # load polygons (if exist) from current json_input path
    msg.text = f"✅ Reloaded <code>{path.as_posix()}</code> ({len(samp):,} points)"
    update_status()


def on_poly_select_change(attr, old, new):
    if new == "(none)":
        name_input.value = ""
        return
    i = int(new)
    names = list(poly.data.get("name", []))
    name_input.value = names[i] if i < len(names) else ""


# ---- wire up ----
reload_btn.on_click(reload)
save_btn.on_click(save_json)
load_btn.on_click(load_json)
set_name_btn.on_click(set_name)

poly_select.on_change("value", on_poly_select_change)
poly.on_change("data", lambda attr, old, new: update_status())

update_status()

curdoc().add_root(
    column(
        csv_select,
        csv_input,
        json_input,
        row(reload_btn, save_btn, load_btn),
        row(poly_select, name_input, set_name_btn),
        msg,
        status,
        p,
    )
)
curdoc().title = "LiDAR Geofence Drawer"
