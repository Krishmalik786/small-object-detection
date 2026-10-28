"""
annotate_masks.py - local browser tool to draw binary masks for small infrared targets.

    python annotate_masks.py --inspect      # read-only report on the folders, writes nothing
    python annotate_masks.py                # start the tool, opens http://127.0.0.1:8770/

Runs only on this PC (127.0.0.1), uses no internet, no CDN, no external files.
Needs only numpy and Pillow. Original images and masks are only read, never written.

How a mask is made: every polygon / brush stroke / eraser stroke is stored as an action.
The SERVER replays the actions at the image's original resolution with Pillow, with no
anti-aliasing, so the saved PNG is strictly 0 (background) / 255 (target). The browser shows
exactly the mask the server produced, so what you see is what is saved.
"""

# =====================================================================================
# CONFIG - change these (or override them on the command line, e.g. --images_dir ...)
# =====================================================================================
CONFIG = dict(
    # folder with the images to annotate. Typical choices:
    #   "output_gt_check/review/both_failed/images"   (from the review page's "Both fail")
    #   "manual_annotation/images"
    images_dir="output_gt_check/review/both_failed/images",

    # optional folders with existing masks, shown as overlays / usable as a starting point.
    # "" = not used. Mask name = <image stem><mask_suffix>.<png/bmp/tif/jpg>
    gt_masks_dir="",
    pred_masks_dir="",
    mask_suffix="",

    # optional: the GT-check output folder. Old GT / prediction masks are found there
    # automatically (manual/<category>/gt_masks and pred_masks). "" = not used.
    source_output="output_gt_check",

    # where new masks are saved. "" = <annotation folder>/../new_masks, where the annotation
    # folder is the parent of images_dir if images_dir is called "images", else images_dir.
    output_dir="",

    port=8770,
)
# =====================================================================================

import argparse
import io
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_DIR = Path(__file__).resolve().parent
IMG_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
HIGH_BIT_MODES = {"I;16", "I;16B", "I;16L", "I;16N", "I", "F"}
RESAMPLE = getattr(Image, "Resampling", Image)
STATUSES = ("not_started", "in_progress", "done", "no_target")
MAX_BRUSH = 64


# ------------------------------------------------------------------------------------- setup
def parse_args():
    p = argparse.ArgumentParser(description="Local binary-mask annotation tool.")
    for k, v in CONFIG.items():
        p.add_argument("--" + k, type=type(v), default=v)
    p.add_argument("--inspect", action="store_true", help="print a read-only report and exit")
    p.add_argument("--no_browser", action="store_true", help="do not open the browser automatically")
    return p.parse_args()


def resolve(path):
    if not path:
        return None
    path = Path(path).expanduser()
    return (path if path.is_absolute() else REPO_DIR / path).resolve()


def finish_config(cfg):
    cfg.images_dir = resolve(cfg.images_dir)
    cfg.gt_masks_dir = resolve(cfg.gt_masks_dir)
    cfg.pred_masks_dir = resolve(cfg.pred_masks_dir)
    cfg.source_output = resolve(cfg.source_output)
    if cfg.output_dir:
        cfg.output_dir = resolve(cfg.output_dir)
    else:
        ann = cfg.images_dir.parent if cfg.images_dir.name == "images" else cfg.images_dir
        cfg.output_dir = (ann.parent / "new_masks").resolve()
    return cfg


def protected_dirs(cfg):
    dirs = [d for d in (cfg.images_dir, cfg.gt_masks_dir, cfg.pred_masks_dir) if d]
    if cfg.source_output and (cfg.source_output / "manual").is_dir():
        dirs.append(cfg.source_output / "manual")
    return dirs


def check_output_location(cfg):
    out = cfg.output_dir
    for d in protected_dirs(cfg):
        if out == d or d in out.parents:
            raise SystemExit("output_dir %s is inside the input folder %s - choose another output_dir" % (out, d))
        if out in d.parents:
            raise SystemExit("input folder %s is inside output_dir %s - choose another output_dir" % (d, out))


# ------------------------------------------------------------------------------------- files
def list_images(folder):
    files = {}
    if folder and folder.is_dir():
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS and not f.name.startswith("."):
                files.setdefault(f.stem, f)
    return files


def index_masks(folder, suffix):
    out = {}
    for stem, f in list_images(folder).items():
        if suffix:
            if stem.endswith(suffix):
                out[stem[: -len(suffix)]] = f
        else:
            out[stem] = f
    return out


def find_existing_masks(cfg, ids):
    gt, pred = {}, {}
    if cfg.source_output and (cfg.source_output / "manual").is_dir():
        for cat in sorted((cfg.source_output / "manual").iterdir()):
            gt.update(index_masks(cat / "gt_masks", ""))
            pred.update(index_masks(cat / "pred_masks", ""))
        if not pred:
            pred.update(index_masks(cfg.source_output / "pred_masks", ""))
    if cfg.gt_masks_dir:                            # explicit folders win
        gt.update(index_masks(cfg.gt_masks_dir, cfg.mask_suffix))
    if cfg.pred_masks_dir:
        pred.update(index_masks(cfg.pred_masks_dir, cfg.mask_suffix))
    return {i: gt[i] for i in ids if i in gt}, {i: pred[i] for i in ids if i in pred}


# ------------------------------------------------------------------------------------- reading
def stretch_to_8bit(a, lo_pct=0.5, hi_pct=99.5):
    a = a.astype(np.float64)
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def load_image(path):
    """Return (display uint8 HxWx3 for the screen, raw numpy array for pixel values, PIL mode)."""
    im = Image.open(path)
    mode = im.mode
    if mode in HIGH_BIT_MODES:
        raw = np.array(im)
        g = stretch_to_8bit(raw)                    # 16-bit -> 8-bit for DISPLAY only
        return np.stack([g, g, g], -1), raw, mode
    if mode == "L":
        raw = np.array(im)
        return np.stack([raw, raw, raw], -1), raw, mode
    rgb = np.array(im.convert("RGB"))
    return rgb, rgb, mode


def read_binary_mask(path, shape=None):
    """Any mask format -> bool HxW (resized with NEAREST to `shape` if given). Returns (mask, format)."""
    im = Image.open(path)
    mode = im.mode
    if mode == "1":
        m, fmt = np.array(im).astype(bool), "1-bit"
    elif mode == "P":
        m, fmt = np.array(im) > 0, "palette"
    elif mode in ("RGB", "RGBA"):
        rgb = np.array(im.convert("RGB")).astype(np.int32)
        if path.suffix.lower() in (".jpg", ".jpeg"):
            m, fmt = rgb.max(-1) > 127, "jpg"
        else:
            m, fmt = rgb.sum(-1) > 0, "colour"
    else:
        a = np.array(im.convert("L") if mode == "LA" else im)
        vals = np.unique(a)
        if path.suffix.lower() in (".jpg", ".jpeg"):
            m, fmt = a > 127, "jpg"
        elif len(vals) <= 2:
            m, fmt = a > 0, "0/%s" % (vals.max() if len(vals) else 0) if vals.max() > 0 else "empty"
        else:
            m, fmt = a > (127 if a.max() <= 255 else 32767), "grey"
    if shape is not None and m.shape != shape:
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((shape[1], shape[0]), RESAMPLE.NEAREST)) > 127
        fmt += " (resized)"
    return m, fmt


# ------------------------------------------------------------------------------------- masks
def stroke_cells(pts):
    """Integer pixels along a stroke. Same rounding (floor(v + 0.5)) as the browser."""
    cells = []
    prev = None
    for x, y in pts:
        x, y = int(x), int(y)
        if prev is None:
            cells.append((x, y))
        else:
            x0, y0 = prev
            dx, dy = x - x0, y - y0
            steps = max(abs(dx), abs(dy))
            for s in range(1, steps + 1):
                cells.append((int(np.floor(x0 + dx * s / steps + 0.5)), int(np.floor(y0 + dy * s / steps + 0.5))))
        prev = (x, y)
    return cells


def paint(mask, cells, size, value):
    h, w = mask.shape
    a = (size - 1) // 2                             # square brush: size 1 = exactly one pixel
    for x, y in cells:
        x0, y0 = max(0, x - a), max(0, y - a)
        x1, y1 = min(w, x - a + size), min(h, y - a + size)
        if x0 < x1 and y0 < y1:
            mask[y0:y1, x0:x1] = value


def validate_ops(ops):
    if not isinstance(ops, list):
        raise ValueError("ops must be a list")
    for op in ops:
        t = op.get("t") if isinstance(op, dict) else None
        if t in ("poly", "brush", "erase"):
            pts = op.get("pts")
            if not isinstance(pts, list) or not pts or not all(
                    isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p) for p in pts):
                raise ValueError("bad points in %s action" % t)
            if t == "poly" and len(pts) < 3:
                raise ValueError("a polygon needs at least 3 points")
            if t != "poly" and not (isinstance(op.get("size"), int) and 1 <= op["size"] <= MAX_BRUSH):
                raise ValueError("bad brush size")
        elif t == "load":
            if op.get("src") not in ("gt", "pred", "saved"):
                raise ValueError("bad load source")
        elif t != "clear":
            raise ValueError("unknown action %r" % t)


def render_ops(ops, h, w, sources):
    """Replay the actions at original resolution. Polygon vertices are pixel centres; Pillow fills
    every pixel whose centre is inside or on the outline (outline=1 includes the boundary pixels)."""
    m = np.zeros((h, w), dtype=bool)
    for op in ops:
        t = op["t"]
        if t == "clear":
            m[:] = False
        elif t == "load":
            src = sources(op["src"])
            if src is None:
                raise ValueError("no %s mask exists for this image" % op["src"])
            m = src.copy()
        elif t == "poly":
            im = Image.new("L", (w, h), 0)
            ImageDraw.Draw(im).polygon([(int(x), int(y)) for x, y in op["pts"]], fill=1, outline=1)
            m |= np.array(im, dtype=bool)
        else:
            paint(m, stroke_cells(op["pts"]), op["size"], t == "brush")
    return m


def count_blobs(mask):
    """8-connected target count without scipy (targets are small, so a simple flood fill is fine)."""
    ys, xs = np.nonzero(mask)
    if len(ys) > 200000:
        return None
    todo = set(zip(ys.tolist(), xs.tolist()))
    n = 0
    while todo:
        n += 1
        stack = [todo.pop()]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in todo:
                        todo.remove(q)
                        stack.append(q)
    return n


# ------------------------------------------------------------------------------------- state
class Project:
    def __init__(self, cfg):
        self.cfg = cfg
        if not cfg.images_dir.is_dir():
            raise SystemExit("images_dir does not exist: %s\nSet images_dir in CONFIG or use --images_dir." % cfg.images_dir)
        check_output_location(cfg)
        self.images = list_images(cfg.images_dir)
        if not self.images:
            raise SystemExit("No images found in %s" % cfg.images_dir)
        self.ids = sorted(self.images)
        if "progress" in self.ids:                  # its shapes file would collide with progress.json
            raise SystemExit("An image is called 'progress'. Rename it (in a copy of the folder) before annotating.")
        self.gt, self.pred = find_existing_masks(cfg, self.ids)
        self.out = cfg.output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.out / "progress.json"
        self.progress = {"last_index": 0, "status": {}}
        if self.progress_path.is_file():
            self.progress.update(json.loads(self.progress_path.read_text()))
        self.sizes = {}
        for i in self.ids:
            with Image.open(self.images[i]) as im:
                self.sizes[i] = (im.size[1], im.size[0], im.mode)
        self.lock = threading.Lock()
        self.cache = {}

    # -- helpers
    def image(self, idx):
        i = self.ids[idx]
        if i not in self.cache:
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = load_image(self.images[i])
        return self.cache[i]

    def source_mask(self, idx, src):
        i = self.ids[idx]
        h, w, _ = self.sizes[i]
        path = {"gt": self.gt.get(i), "pred": self.pred.get(i), "saved": self.out / (i + ".png")}[src]
        if path is None or not Path(path).is_file():
            return None
        return read_binary_mask(Path(path), (h, w))[0]

    def status(self, i):
        return self.progress["status"].get(i, "not_started")

    def write_progress(self):
        self.progress["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.progress_path.with_name("progress.json.tmp")
        tmp.write_text(json.dumps(self.progress, indent=1))
        os.replace(tmp, self.progress_path)

    # -- API
    def listing(self):
        items = []
        for i in self.ids:
            h, w, mode = self.sizes[i]
            items.append(dict(id=i, file=self.images[i].name, w=w, h=h, mode=mode, has_gt=i in self.gt,
                              has_pred=i in self.pred, status=self.status(i)))
        return dict(items=items, last_index=min(int(self.progress.get("last_index", 0)), len(self.ids) - 1),
                    images_dir=str(self.cfg.images_dir), output_dir=str(self.out))

    def display_png(self, idx):
        buf = io.BytesIO()
        Image.fromarray(self.image(idx)[0]).save(buf, "PNG")
        return buf.getvalue()

    def raw_values(self, idx):
        raw = self.image(idx)[1]
        code = {"uint8": "u1", "uint16": "u2", "int32": "i4", "float32": "f4"}.get(str(raw.dtype))
        if code is None:
            raw, code = raw.astype(np.float32), "f4"
        c = 1 if raw.ndim == 2 else raw.shape[2]
        return raw.astype("<" + code).tobytes(), dict(h=raw.shape[0], w=raw.shape[1], c=c, dtype=code)

    def mask_bits(self, idx, src):
        m = self.source_mask(idx, src)
        return None if m is None else m.astype(np.uint8).tobytes()

    def annotation(self, idx):
        i = self.ids[idx]
        js = self.out / (i + ".json")
        if js.is_file():
            return json.loads(js.read_text())
        if (self.out / (i + ".png")).is_file():     # a mask without its JSON: start from the saved mask
            return dict(ops=[{"t": "load", "src": "saved"}], status=self.status(i))
        return dict(ops=[], status=self.status(i))

    def render(self, idx, ops):
        validate_ops(ops)
        h, w, _ = self.sizes[self.ids[idx]]
        return render_ops(ops, h, w, lambda s: self.source_mask(idx, s))

    def save(self, idx, ops, kind):
        i = self.ids[idx]
        h, w, _ = self.sizes[i]
        with self.lock:
            mask = self.render(idx, ops)
            prev = self.status(i)
            if kind == "auto":
                status = prev if prev in ("done", "no_target") else "in_progress"
            elif kind in ("done", "no_target"):
                status = kind
            else:
                raise ValueError("bad save kind %r" % kind)
            if status == "no_target" and mask.any():
                status = "done"                     # the mask has pixels now -> it is not "no target" any more
            if kind == "no_target" and mask.any():
                raise ValueError("'No target' needs an empty mask")
            png = self.out / (i + ".png")
            tmp = self.out / (i + ".png.tmp")
            Image.fromarray(mask.astype(np.uint8) * 255, "L").save(tmp, "PNG")
            os.replace(tmp, png)
            check = Image.open(png)                 # verify what is on disk
            vals = np.unique(np.array(check))
            problems = []
            if check.size != (w, h):
                problems.append("size %dx%d differs from image %dx%d" % (check.size[0], check.size[1], w, h))
            if not set(vals.tolist()) <= {0, 255}:
                problems.append("values %s are not only 0/255" % vals.tolist())
            if check.mode != "L":
                problems.append("mode %s is not L" % check.mode)
            if problems:
                raise ValueError("saved mask failed verification: " + "; ".join(problems))
            meta = dict(image=self.images[i].name, width=w, height=h, status=status, ops=ops,
                        saved=time.strftime("%Y-%m-%d %H:%M:%S"), tool="annotate_masks.py")
            jtmp = self.out / (i + ".json.tmp")
            jtmp.write_text(json.dumps(meta))
            os.replace(jtmp, self.out / (i + ".json"))
            self.progress["status"][i] = status
            self.write_progress()
            return dict(ok=True, status=status, file=str(png), width=w, height=h, values=vals.tolist(),
                        target_pixels=int(mask.sum()), targets=count_blobs(mask))

    def set_last(self, idx):
        with self.lock:
            self.progress["last_index"] = int(idx)
            self.write_progress()


# ------------------------------------------------------------------------------------- inspect
def inspect(cfg):
    print("== Python packages")
    print("   python %s, numpy %s, Pillow %s" % (sys.version.split()[0], np.__version__, Image.__version__))
    try:
        import flask  # noqa: F401
        print("   flask: installed (not needed; this tool uses the standard library)")
    except ImportError:
        print("   flask: not installed (not needed)")
    root = cfg.images_dir.parent if cfg.images_dir.name == "images" else cfg.images_dir
    print("\n== Annotation folder: %s" % root)
    if not root.is_dir():
        print("   does NOT exist - set images_dir in CONFIG or use --images_dir")
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        depth = len(Path(dirpath).relative_to(root).parts)
        files = [f for f in filenames if not f.startswith(".")]
        exts = Counter(Path(f).suffix.lower() or "(none)" for f in files)
        print("   %s%s/  %d files  %s" % ("    " * depth, Path(dirpath).name, len(files),
                                         ", ".join("%s x%d" % kv for kv in exts.most_common())))
        if depth >= 2:
            dirnames[:] = []
    images = list_images(cfg.images_dir)
    print("\n== Images in %s: %d" % (cfg.images_dir, len(images)))
    names = [f.name for f in images.values()]
    print("   names: %s%s" % (names[:15], " ..." if len(names) > 15 else ""))
    sizes, modes = Counter(), Counter()
    for f in images.values():
        try:
            with Image.open(f) as im:
                sizes["%dx%d" % im.size] += 1
                modes[im.mode] += 1
        except Exception as e:
            print("   unreadable: %s (%s)" % (f.name, e))
    for k, n in sizes.most_common():
        print("   size %-12s x%d" % (k, n))
    for k, n in modes.most_common():
        print("   PIL mode %-7s x%d%s" % (k, n, "   <- 16-bit: shown stretched, saved masks unaffected" if k in HIGH_BIT_MODES else ""))
    ids = sorted(images)
    gt, pred = find_existing_masks(cfg, ids)
    print("\n== Existing masks found (from %s)" % ", ".join(str(x) for x in (cfg.gt_masks_dir, cfg.pred_masks_dir, cfg.source_output) if x))
    for name, found in (("GT", gt), ("prediction", pred)):
        fmts = Counter()
        for i, p in found.items():
            try:
                h_w = Image.open(images[i]).size[::-1]
                fmts[read_binary_mask(p, h_w)[1]] += 1
            except Exception as e:
                fmts["unreadable (%s)" % type(e).__name__] += 1
        print("   %-10s masks for %d of %d images   formats: %s" % (name, len(found), len(ids),
              ", ".join("%s x%d" % kv for kv in fmts.most_common()) or "-"))
        missing = [i for i in ids if i not in found]
        if found and missing:
            print("              no %s mask for: %s%s" % (name, missing[:10], " ..." if len(missing) > 10 else ""))
    print("\n== Output folder: %s" % cfg.output_dir)
    try:
        check_output_location(cfg)
        print("   location OK (not inside any input folder)")
    except SystemExit as e:
        print("   PROBLEM: %s" % e)
    prog = cfg.output_dir / "progress.json"
    if prog.is_file():
        st = Counter(json.loads(prog.read_text()).get("status", {}).values())
        print("   existing progress: %s" % dict(st))
    else:
        print("   no progress yet (folder is created when the tool starts)")
    print("\nNothing was written (inspect).")


# ------------------------------------------------------------------------------------- page
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Mask annotation</title>
<style>
:root{--bg:#1b1c1e;--panel:#26272b;--line:#3a3b40;--text:#e6e6e6;--dim:#9aa0a6;--acc:#4c8bf5}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:13px system-ui,Segoe UI,Arial,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
#top{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;padding:6px 10px;background:var(--panel);border-bottom:1px solid var(--line)}
.grp{display:flex;gap:4px;align-items:center} .grp b{color:var(--dim);font-weight:600;margin-right:2px}
button{background:#3a3b40;color:var(--text);border:1px solid #4a4b50;border-radius:4px;padding:4px 9px;cursor:pointer;font-size:13px}
button:hover{background:#45464c} button.on{background:var(--acc);border-color:var(--acc);color:#fff}
button.good{background:#1e7d39;border-color:#1e7d39} button.warn{background:#8a3a2e;border-color:#8a3a2e}
input[type=range]{width:90px} label{color:var(--dim)}
#main{flex:1;display:flex;min-height:0}
#side{width:230px;background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column}
#prog{padding:8px 10px;border-bottom:1px solid var(--line)} #bar{height:6px;background:#3a3b40;border-radius:3px;margin-top:5px;overflow:hidden} #bar div{height:100%;background:#1e7d39}
#list{overflow:auto;flex:1} .it{padding:4px 10px;cursor:pointer;display:flex;gap:6px;align-items:center;white-space:nowrap;overflow:hidden}
.it:hover{background:#303136} .it.cur{background:#36435a} .dot{width:9px;height:9px;border-radius:50%;flex:none}
.not_started{background:#666}.in_progress{background:#e3a008}.done{background:#2ea043}.no_target{background:#4c8bf5}
#wrap{flex:1;position:relative;min-width:0} canvas{display:block;width:100%;height:100%;cursor:crosshair;background:#111}
#status{padding:4px 10px;background:var(--panel);border-top:1px solid var(--line);color:var(--dim);white-space:nowrap;overflow:hidden}
#status span{color:var(--text)}
#help{position:absolute;top:12px;right:12px;background:rgba(20,20,22,.96);border:1px solid var(--line);border-radius:6px;padding:12px 16px;max-height:90%;overflow:auto;display:none}
#help table{border-collapse:collapse} #help td{padding:2px 10px 2px 0} #help td:first-child{color:#ffd54f;font-family:monospace}
#toast{position:absolute;left:50%;bottom:14px;transform:translateX(-50%);padding:7px 14px;border-radius:5px;background:#2b2c30;border:1px solid var(--line);display:none}
</style></head><body>
<div id="top">
 <div class="grp"><b>Tool</b><button id="t_poly">1 Polygon</button><button id="t_brush">2 Brush</button><button id="t_erase">3 Eraser</button>
  <label>size <input type="number" id="bsize" min="1" max="64" value="1" style="width:46px"></label></div>
 <div class="grp"><b>Edit</b><button id="undo">Undo</button><button id="redo">Redo</button>
  <button id="fromgt">Start from GT</button><button id="frompred">Start from pred</button><button id="clear" class="warn">Clear</button><button id="notarget">No target</button></div>
 <div class="grp"><b>Show</b><button id="s_mask" class="on">M mask</button><button id="s_gt">G GT</button><button id="s_pred">R pred</button><button id="s_saved">V saved</button><button id="s_grid" class="on">X grid</button>
  <label>opacity <input type="range" id="op" min="10" max="100" value="45"></label></div>
 <div class="grp"><b>Display</b><label>bright <input type="range" id="br" min="-100" max="100" value="0"></label><label>contrast <input type="range" id="co" min="-100" max="100" value="0"></label><label>gamma <input type="range" id="ga" min="-100" max="100" value="0"></label>
  <button id="auto">A auto stretch</button><button id="reset">reset</button><button id="fit">0 fit</button>
  <button id="scrollmode" title="Mouse: keep 'zoom'. Trackpad: 'pan' = two-finger scroll moves the image, pinch zooms.">T scroll: zoom</button></div>
 <div class="grp"><button id="prev">&larr; P</button><button id="save" class="good">Save (Ctrl+S)</button><button id="next">N &rarr;</button><button id="helpb">H help</button></div>
</div>
<div id="main"><div id="side"><div id="prog"><div id="progtxt"></div><div id="bar"><div></div></div></div><div id="list"></div></div>
 <div id="wrap"><canvas id="cv"></canvas>
  <div id="help"><b>Keyboard shortcuts</b><table>
   <tr><td>1 / 2 / 3</td><td>polygon / brush / eraser</td></tr><tr><td>[ ]</td><td>brush size smaller / bigger</td></tr>
   <tr><td>click</td><td>polygon: add point (points are pixel centres)</td></tr><tr><td>click the green ring, Enter, double-click</td><td>close polygon (the green ring is the first point; it fills when you are close enough)</td></tr>
   <tr><td>drag a point</td><td>move it (also on finished polygons)</td></tr><tr><td>Backspace</td><td>remove last polygon point</td></tr><tr><td>Esc</td><td>cancel the polygon being drawn</td></tr>
   <tr><td>drag</td><td>brush / eraser: paint</td></tr><tr><td>Space+drag, right or middle drag, arrow keys</td><td>pan</td></tr><tr><td>mouse wheel, + / -</td><td>zoom at the cursor (up to 64x)</td></tr>
   <tr><td>pinch (trackpad)</td><td>zoom at the cursor</td></tr><tr><td>T</td><td>two-finger scroll: zoom or pan (use "pan" on a trackpad)</td></tr>
   <tr><td>0</td><td>fit to screen</td></tr><tr><td>Ctrl+Z / Ctrl+Y</td><td>undo / redo</td></tr><tr><td>Ctrl+S</td><td>save and mark done</td></tr>
   <tr><td>N / P</td><td>next / previous image (auto-saves)</td></tr><tr><td>M / G / R</td><td>show my mask / GT / prediction</td></tr>
   <tr><td>V</td><td>show the SAVED mask as black/white</td></tr><tr><td>X</td><td>pixel grid (from 8x zoom)</td></tr><tr><td>A</td><td>auto stretch display</td></tr><tr><td>H</td><td>this help</td></tr></table>
   <p style="color:#9aa0a6;max-width:430px">Display sliders change only what you see, never the saved image or mask.
   The mask is computed by the server at the image's real resolution, exactly as it is saved.</p></div>
  <div id="toast"></div></div></div>
<div id="status"></div>
<script>
"use strict";
const $=id=>document.getElementById(id), cv=$('cv'), ctx=cv.getContext('2d');
const S={items:[],idx:-1,w:0,h:0,base:null,raw:null,rawC:1,ops:[],undo:[],redo:[],mask:null,dirty:false,
 tool:'poly',size:1,poly:[],z:1,ox:0,oy:0,fitZ:1,show:{mask:true,gt:false,pred:false,saved:false,grid:true},
 lo:0,hi:255,imgC:null,maskC:null,gtC:null,predC:null,savedC:null,gtBits:null,predBits:null,mx:-1,my:-1,renderSeq:0,busy:false};
const COL={mask:[255,214,0],gt:[0,230,90],pred:[255,60,255]};
let drag=null, space=false;

function esc(s){return String(s).replace(/[&<>"']/g,c=>'&#'+c.charCodeAt(0)+';');}
function toast(t,ms){const e=$('toast');e.textContent=t;e.style.display='block';clearTimeout(e._t);e._t=setTimeout(()=>e.style.display='none',ms||1800);}
async function api(path,opt){const r=await fetch(path,opt);if(!r.ok){let t=await r.text();try{t=JSON.parse(t).error||t}catch(e){}throw new Error(t);}return r;}
function post(path,body){return api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}

// ---------- list / progress
function drawList(){
 const L=$('list');L.innerHTML='';
 S.items.forEach((it,i)=>{const d=document.createElement('div');d.className='it'+(i===S.idx?' cur':'');d.title=it.file+'  '+it.w+'x'+it.h+'  '+it.mode;
  d.innerHTML='<span class="dot '+it.status+'"></span><span></span>';d.lastChild.textContent=it.id;d.onclick=()=>openImg(i);L.appendChild(d);});
 const done=S.items.filter(x=>x.status==='done'||x.status==='no_target').length;
 $('progtxt').textContent=done+' / '+S.items.length+' done';$('bar').firstChild.style.width=(100*done/Math.max(1,S.items.length))+'%';
 const c=L.children[S.idx]; if(c) c.scrollIntoView({block:'nearest'});
}

// ---------- open an image
async function openImg(i){
 if(i<0||i>=S.items.length||S.busy) return;
 S.busy=true;
 try{
  if(S.idx>=0) await leaveImage();
  S.idx=i; const it=S.items[i]; S.w=it.w; S.h=it.h; S.poly=[]; S.undo=[]; S.redo=[]; S.dirty=false;
  const [img,raw,ann]=await Promise.all([loadPng('/api/image/'+i),api('/api/raw/'+i),api('/api/annotation/'+i).then(r=>r.json())]);
  const t=document.createElement('canvas');t.width=S.w;t.height=S.h;const tc=t.getContext('2d');tc.drawImage(img,0,0);S.base=tc.getImageData(0,0,S.w,S.h);
  const buf=await raw.arrayBuffer(),dt=raw.headers.get('X-Dtype');S.rawC=+raw.headers.get('X-C');
  S.raw=dt==='u2'?new Uint16Array(buf):dt==='i4'?new Int32Array(buf):dt==='f4'?new Float32Array(buf):new Uint8Array(buf);
  S.gtBits=it.has_gt?await bits('/api/maskbits/'+i+'/gt'):null; S.predBits=it.has_pred?await bits('/api/maskbits/'+i+'/pred'):null;
  S.gtC=S.gtBits?bitsCanvas(S.gtBits,COL.gt):null; S.predC=S.predBits?bitsCanvas(S.predBits,COL.pred):null;
  S.ops=ann.ops||[]; await loadSaved(); S.lo=0;S.hi=255; applyDisplay(); await serverRender(); fit();
  post('/api/progress',{last_index:i}).catch(()=>{});
  drawList(); draw();
 }catch(e){alert('Could not open image: '+e.message);}
 S.busy=false;
}
function loadPng(url){return new Promise((res,rej)=>{const im=new Image();im.onload=()=>res(im);im.onerror=()=>rej(new Error('image load failed'));im.src=url+'?t='+Date.now();});}
async function bits(url){const r=await fetch(url);if(!r.ok)return null;return new Uint8Array(await r.arrayBuffer());}
async function loadSaved(){const it=S.items[S.idx];const b=(it.status!=='not_started')?await bits('/api/maskbits/'+S.idx+'/saved'):null;S.savedC=b?bitsCanvas(b,null):null;}
function bitsCanvas(b,col){const c=document.createElement('canvas');c.width=S.w;c.height=S.h;const x=c.getContext('2d'),d=x.createImageData(S.w,S.h);
 for(let k=0;k<b.length;k++){const o=4*k;if(col){if(b[k]){d.data[o]=col[0];d.data[o+1]=col[1];d.data[o+2]=col[2];d.data[o+3]=255;}}else{const v=b[k]?255:0;d.data[o]=d.data[o+1]=d.data[o+2]=v;d.data[o+3]=255;}}
 x.putImageData(d,0,0);return c;}

// ---------- save
function finishPolyIfAny(){if(S.poly.length>=3){pushOp({t:'poly',pts:S.poly});}S.poly=[];}
async function leaveImage(){finishPolyIfAny(); if(S.dirty) await doSave('auto');}
async function doSave(kind){
 const r=await post('/api/save/'+S.idx,{ops:S.ops,kind}); const j=await r.json();
 S.items[S.idx].status=j.status; S.dirty=false; await loadSaved(); drawList(); draw(); return j;
}
async function saveDone(){
 finishPolyIfAny();
 if(!maskCount()){ if(!confirm('The mask is empty. Save it as "no target"?')) return; return noTarget(); }
 try{const j=await doSave('done');toast('Saved '+j.width+'x'+j.height+', values '+JSON.stringify(j.values)+', '+j.target_pixels+' target px in '+j.targets+' target(s)',3000);}
 catch(e){alert('SAVE FAILED: '+e.message);}
}
async function noTarget(){
 S.poly=[]; if(maskCount()) pushOp({t:'clear'},true);
 try{await serverRender(); await doSave('no_target'); toast('Saved as "no target" (empty mask)');}catch(e){alert('SAVE FAILED: '+e.message);}
}

// ---------- ops / undo
function pushOp(op,noRender){S.undo.push(JSON.stringify(S.ops));S.redo=[];S.ops.push(op);S.dirty=true;if(!noRender)serverRender();}
function undo(){if(S.poly.length){S.poly.pop();draw();return;} if(!S.undo.length)return; S.redo.push(JSON.stringify(S.ops));S.ops=JSON.parse(S.undo.pop());S.dirty=true;serverRender();}
function redo(){if(!S.redo.length)return;S.undo.push(JSON.stringify(S.ops));S.ops=JSON.parse(S.redo.pop());S.dirty=true;serverRender();}
async function serverRender(){
 const seq=++S.renderSeq;
 try{const r=await post('/api/render/'+S.idx,{ops:S.ops});const b=new Uint8Array(await r.arrayBuffer());
  if(seq!==S.renderSeq)return; S.mask=b; S.maskC=bitsCanvas(b,COL.mask); draw();}
 catch(e){alert('Mask error: '+e.message); if(S.undo.length){S.ops=JSON.parse(S.undo.pop());serverRender();}}
}
function maskCount(){let n=0;if(S.mask)for(let k=0;k<S.mask.length;k++)n+=S.mask[k];return n;}

// ---------- brush (same pixel rule as the server: square, floor(v+0.5) interpolation)
function strokeCells(pts){const out=[];let p=null;for(const [x,y] of pts){if(!p){out.push([x,y]);}else{const dx=x-p[0],dy=y-p[1],st=Math.max(Math.abs(dx),Math.abs(dy));
 for(let s=1;s<=st;s++)out.push([Math.floor(p[0]+dx*s/st+0.5),Math.floor(p[1]+dy*s/st+0.5)]);}p=[x,y];}return out;}
function paintLocal(cells,size,val){if(!S.mask)return;const a=Math.floor((size-1)/2),x=S.maskC.getContext('2d');
 for(const [cx,cy] of cells){const x0=Math.max(0,cx-a),y0=Math.max(0,cy-a),x1=Math.min(S.w,cx-a+size),y1=Math.min(S.h,cy-a+size);
  for(let yy=y0;yy<y1;yy++)for(let xx=x0;xx<x1;xx++)S.mask[yy*S.w+xx]=val?1:0;
  if(x1>x0&&y1>y0){if(val){x.fillStyle='rgb('+COL.mask+')';x.fillRect(x0,y0,x1-x0,y1-y0);}else x.clearRect(x0,y0,x1-x0,y1-y0);}}}

// ---------- display adjustments (screen only)
function applyDisplay(){
 if(!S.base)return; const b=+$('br').value/100, c=Math.pow(4,+$('co').value/100), g=Math.pow(4,-$('ga').value/100), lo=S.lo, hi=Math.max(S.hi,S.lo+1);
 const lut=new Uint8ClampedArray(256);
 for(let v=0;v<256;v++){let t=(v-lo)/(hi-lo);t=(t-0.5)*c+0.5+b;t=Math.min(1,Math.max(0,t));lut[v]=255*Math.pow(t,g);}
 const c2=document.createElement('canvas');c2.width=S.w;c2.height=S.h;const x=c2.getContext('2d'),d=x.createImageData(S.w,S.h),s=S.base.data;
 for(let k=0;k<s.length;k+=4){d.data[k]=lut[s[k]];d.data[k+1]=lut[s[k+1]];d.data[k+2]=lut[s[k+2]];d.data[k+3]=255;}
 x.putImageData(d,0,0);S.imgC=c2;draw();
}
function autoStretch(){const h=new Uint32Array(256),s=S.base.data;for(let k=0;k<s.length;k+=4)h[s[k]]++;const n=s.length/4;
 const pct=p=>{let a=0;for(let v=0;v<256;v++){a+=h[v];if(a>=p*n)return v;}return 255;};S.lo=pct(0.005);S.hi=pct(0.995);
 $('br').value=0;$('co').value=0;$('ga').value=0;applyDisplay();toast('display stretched '+S.lo+'..'+S.hi);}

// ---------- view
function resize(){cv.width=cv.clientWidth;cv.height=cv.clientHeight;draw();}
function fit(){if(!S.w)return;S.fitZ=Math.min(cv.width/S.w,cv.height/S.h)*0.96;S.z=S.fitZ;S.ox=(cv.width-S.w*S.z)/2;S.oy=(cv.height-S.h*S.z)/2;draw();}
function toImg(sx,sy){return [(sx-S.ox)/S.z,(sy-S.oy)/S.z];}
function toScr(px,py){return [S.ox+(px+0.5)*S.z,S.oy+(py+0.5)*S.z];}
function draw(){
 ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#111';ctx.fillRect(0,0,cv.width,cv.height); if(!S.imgC){status();return;}
 ctx.imageSmoothingEnabled=false; const W=S.w*S.z,H=S.h*S.z;
 if(S.show.saved){ if(S.savedC) ctx.drawImage(S.savedC,S.ox,S.oy,W,H); else {ctx.drawImage(S.imgC,S.ox,S.oy,W,H);} }
 else{
  ctx.drawImage(S.imgC,S.ox,S.oy,W,H); ctx.globalAlpha=+$('op').value/100;
  if(S.show.gt&&S.gtC)ctx.drawImage(S.gtC,S.ox,S.oy,W,H); if(S.show.pred&&S.predC)ctx.drawImage(S.predC,S.ox,S.oy,W,H);
  if(S.show.mask&&S.maskC)ctx.drawImage(S.maskC,S.ox,S.oy,W,H); ctx.globalAlpha=1;
 }
 if(S.show.grid&&S.z>=8){ctx.strokeStyle='rgba(255,255,255,0.13)';ctx.lineWidth=1;ctx.beginPath();
  const x0=Math.max(0,Math.floor(-S.ox/S.z)),x1=Math.min(S.w,Math.ceil((cv.width-S.ox)/S.z)),y0=Math.max(0,Math.floor(-S.oy/S.z)),y1=Math.min(S.h,Math.ceil((cv.height-S.oy)/S.z));
  for(let x=x0;x<=x1;x++){const X=Math.round(S.ox+x*S.z)+0.5;ctx.moveTo(X,S.oy+y0*S.z);ctx.lineTo(X,S.oy+y1*S.z);}
  for(let y=y0;y<=y1;y++){const Y=Math.round(S.oy+y*S.z)+0.5;ctx.moveTo(S.ox+x0*S.z,Y);ctx.lineTo(S.ox+x1*S.z,Y);}ctx.stroke();}
 ctx.strokeStyle='#666';ctx.strokeRect(S.ox-0.5,S.oy-0.5,W+1,H+1);
 if(S.tool==='poly'&&!S.show.saved){S.ops.forEach(op=>{if(op.t==='poly')drawPoly(op.pts,true,'#00e5ff');});}
 if(S.poly.length)drawLivePoly();
 if((S.tool==='brush'||S.tool==='erase')&&S.mx>=0){const [px,py]=pix(S.mx,S.my),a=Math.floor((S.size-1)/2);
  ctx.strokeStyle=S.tool==='brush'?'#ffd54f':'#ff5252';ctx.lineWidth=1.5;ctx.strokeRect(S.ox+(px-a)*S.z,S.oy+(py-a)*S.z,S.size*S.z,S.size*S.z);}
 status();
}
function ring(x,y,r,col,fill){ctx.beginPath();ctx.arc(x,y,r,0,2*Math.PI);if(fill){ctx.fillStyle=fill;ctx.fill();}ctx.strokeStyle=col;ctx.lineWidth=2;ctx.stroke();}
function drawPoly(pts,closed,col){ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.beginPath();pts.forEach((p,k)=>{const [x,y]=toScr(...p);k?ctx.lineTo(x,y):ctx.moveTo(x,y);});
 if(closed)ctx.closePath();ctx.stroke();pts.forEach(p=>{const [x,y]=toScr(...p);ring(x,y,3,col,null);});}
const CLOSE_R=10;   // screen px: how close to the first point counts as "close the polygon"
function nearFirst(){if(S.poly.length<3||S.mx<0)return false;const [x,y]=toScr(...S.poly[0]);return Math.hypot(x-S.mx,y-S.my)<=CLOSE_R;}
function drawLivePoly(){
 const pts=S.poly.map(p=>toScr(...p)),cur=S.mx>=0?toScr(...pix(S.mx,S.my)):null,close=nearFirst();
 // shaded preview of the area, including the edge to the cursor
 ctx.beginPath();pts.forEach(([x,y],k)=>k?ctx.lineTo(x,y):ctx.moveTo(x,y));if(cur&&!close)ctx.lineTo(...cur);ctx.closePath();
 ctx.fillStyle='rgba(255,213,79,0.18)';ctx.fill();
 ctx.strokeStyle='#ffd54f';ctx.lineWidth=1.5;ctx.beginPath();pts.forEach(([x,y],k)=>k?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();
 if(cur){ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(...pts[pts.length-1]);ctx.lineTo(...(close?pts[0]:cur));ctx.stroke();ctx.setLineDash([]);}
 pts.slice(1).forEach(([x,y])=>ring(x,y,3,'#ffd54f',null));
 ring(pts[0][0],pts[0][1],7,'#00e676',close?'rgba(0,230,118,0.6)':null);           // first point: big green ring
 if(S.mx>=0){const msg=S.poly.length<3?(S.poly.length+' point(s) - keep clicking around the target (Esc = cancel)'):
  close?'click to CLOSE the polygon':(S.poly.length+' points - close: click the green ring, Enter or double-click | Backspace = remove last point');
  ctx.font='12px system-ui,Segoe UI,Arial';const w=ctx.measureText(msg).width+10,tx=Math.min(S.mx+16,cv.width-w-4),ty=Math.max(S.my-26,4);
  ctx.fillStyle='rgba(0,0,0,0.75)';ctx.fillRect(tx,ty,w,18);ctx.fillStyle=close?'#00e676':'#ffd54f';ctx.fillText(msg,tx+5,ty+13);}
}
function pix(sx,sy){const [x,y]=toImg(sx,sy);return [Math.min(S.w-1,Math.max(0,Math.floor(x))),Math.min(S.h-1,Math.max(0,Math.floor(y)))];}
function status(){
 if(S.idx<0){$('status').textContent='';return;} const it=S.items[S.idx];let v='';
 if(S.mx>=0&&S.raw){const [x,y]=toImg(S.mx,S.my);if(x>=0&&y>=0&&x<S.w&&y<S.h){const k=(Math.floor(y)*S.w+Math.floor(x))*S.rawC;
  v='x='+Math.floor(x)+' y='+Math.floor(y)+'  value='+(S.rawC===1?S.raw[k]:Array.from(S.raw.slice(k,k+S.rawC)).join(','))+(S.mask&&S.mask[Math.floor(y)*S.w+Math.floor(x)]?'  [in mask]':'');}}
 $('status').innerHTML='<span></span>  |  zoom <span>'+S.z.toFixed(1)+'x</span>  |  tool <span>'+S.tool+(S.tool==='poly'?'':' '+S.size+'px')+'</span>  |  mask <span>'+maskCount()+' px</span>  |  '+
  '<span>'+esc(it.id)+'</span> '+it.w+'x'+it.h+' '+it.mode+'  |  status <span>'+it.status.replace('_',' ')+'</span>'+(S.dirty?'  <span style="color:#e3a008">unsaved changes</span>':'');
 $('status').firstChild.textContent=v||'(move the mouse over the image)';
}

// ---------- mouse
function hitVertex(sx,sy){const R=7;for(let k=0;k<S.poly.length;k++){const [x,y]=toScr(...S.poly[k]);if(Math.abs(x-sx)<=R&&Math.abs(y-sy)<=R)return {list:S.poly,k,live:true};}
 if(S.tool==='poly')for(let o=S.ops.length-1;o>=0;o--){const op=S.ops[o];if(op.t!=='poly')continue;for(let k=0;k<op.pts.length;k++){const [x,y]=toScr(...op.pts[k]);if(Math.abs(x-sx)<=R&&Math.abs(y-sy)<=R)return {list:op.pts,k,live:false};}}return null;}
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('mousedown',e=>{
 if(S.idx<0||S.show.saved)return; const r=cv.getBoundingClientRect(),sx=e.clientX-r.left,sy=e.clientY-r.top;
 if(e.button===1||e.button===2||space){drag={pan:true,sx,sy,ox:S.ox,oy:S.oy};return;}
 if(e.button!==0)return; const p=pix(sx,sy);
 if(S.tool==='poly'){
  S.mx=sx;S.my=sy; if(nearFirst()){closePoly();return;}
  const h=hitVertex(sx,sy);
  if(h){if(!h.live)S.undo.push(JSON.stringify(S.ops));drag={vertex:h,moved:false};return;}
  const last=S.poly[S.poly.length-1]; if(last&&last[0]===p[0]&&last[1]===p[1])return;   // same pixel twice: ignore
  S.poly.push(p);draw();return;}
 drag={stroke:{t:S.tool,size:S.size,pts:[p]}};paintLocal(strokeCells([p]),S.size,S.tool==='brush');draw();
});
window.addEventListener('mousemove',e=>{
 const r=cv.getBoundingClientRect();S.mx=e.clientX-r.left;S.my=e.clientY-r.top;
 if(S.mx<0||S.my<0||S.mx>r.width||S.my>r.height){S.mx=S.my=-1;}
 if(drag&&drag.pan){S.ox=drag.ox+(e.clientX-r.left-drag.sx);S.oy=drag.oy+(e.clientY-r.top-drag.sy);}
 else if(drag&&drag.vertex){const p=pix(e.clientX-r.left,e.clientY-r.top),L=drag.vertex.list,k=drag.vertex.k;if(L[k][0]!==p[0]||L[k][1]!==p[1]){L[k]=p;drag.moved=true;}}
 else if(drag&&drag.stroke){const p=pix(e.clientX-r.left,e.clientY-r.top),pts=drag.stroke.pts,last=pts[pts.length-1];
  if(p[0]!==last[0]||p[1]!==last[1]){paintLocal(strokeCells([last,p]),drag.stroke.size,drag.stroke.t==='brush');pts.push(p);}}
 draw();
});
window.addEventListener('mouseup',()=>{
 if(!drag)return; const d=drag;drag=null;
 if(d.vertex){if(d.vertex.live){}
  else{if(d.moved){S.redo=[];S.dirty=true;serverRender();}else S.undo.pop();}draw();return;}
 if(d.stroke){S.undo.push(JSON.stringify(S.ops));S.redo=[];S.ops.push(d.stroke);S.dirty=true;serverRender();}
});
cv.addEventListener('dblclick',()=>{if(S.tool==='poly'&&S.poly.length>=3)closePoly();});
function closePoly(){const pts=S.poly;S.poly=[];pushOp({t:'poly',pts});}
function zoomAt(mx,my,f){const nz=Math.min(64,Math.max(Math.min(S.fitZ,1)*0.5,S.z*f)),[ix,iy]=toImg(mx,my);S.z=nz;S.ox=mx-ix*nz;S.oy=my-iy*nz;draw();}
let scrollPans=false;
cv.addEventListener('wheel',e=>{e.preventDefault();if(S.idx<0)return;const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
 const px=e.deltaMode===1?16:e.deltaMode===2?cv.height:1, dx=e.deltaX*px, dy=e.deltaY*px;
 if(e.ctrlKey){zoomAt(mx,my,Math.exp(Math.max(-0.5,Math.min(0.5,-dy*0.01))));return;}       // trackpad pinch
 if(scrollPans){S.ox-=dx;S.oy-=dy;draw();return;}                                          // trackpad 2-finger scroll
 zoomAt(mx,my,Math.exp(Math.max(-0.5,Math.min(0.5,-dy*0.0025))));                           // mouse wheel: ~1.28x per notch
},{passive:false});
function setScroll(p){scrollPans=p;$('scrollmode').textContent='T scroll: '+(p?'pan':'zoom');$('scrollmode').classList.toggle('on',p);
 try{localStorage.setItem('ann_scroll_pans',p?'1':'0');}catch(e){}}

// ---------- buttons / keys
function setTool(t){S.tool=t;['poly','brush','erase'].forEach(x=>$('t_'+x).classList.toggle('on',x===t));draw();}
function setSize(n){S.size=Math.max(1,Math.min(64,n|0));$('bsize').value=S.size;draw();}
function toggle(k){S.show[k]=!S.show[k];$('s_'+k).classList.toggle('on',S.show[k]);
 if(k==='gt'&&S.show.gt&&!S.gtC)toast('no GT mask for this image');if(k==='pred'&&S.show.pred&&!S.predC)toast('no prediction mask for this image');
 if(k==='saved'&&S.show.saved&&!S.savedC)toast('nothing saved yet for this image');draw();}
function startFrom(src){const it=S.items[S.idx];if(!(src==='gt'?it.has_gt:it.has_pred)){alert('No '+(src==='gt'?'GT':'prediction')+' mask exists for this image.');return;}
 if(maskCount()&&!confirm('Replace the current mask with the '+(src==='gt'?'GT':'prediction')+' mask? (Undo can bring it back.)'))return;
 S.poly=[];pushOp({t:'load',src});}
$('t_poly').onclick=()=>setTool('poly');$('t_brush').onclick=()=>setTool('brush');$('t_erase').onclick=()=>setTool('erase');
$('bsize').onchange=e=>setSize(+e.target.value);$('undo').onclick=undo;$('redo').onclick=redo;
$('fromgt').onclick=()=>startFrom('gt');$('frompred').onclick=()=>startFrom('pred');
$('clear').onclick=()=>{if(confirm('Clear the whole mask for this image? (Undo can bring it back.)')){S.poly=[];pushOp({t:'clear'});}};
$('notarget').onclick=()=>{if(confirm('Save an EMPTY mask for this image ("no target")?'))noTarget();};
['mask','gt','pred','saved','grid'].forEach(k=>$('s_'+k).onclick=()=>toggle(k));
['br','co','ga'].forEach(k=>$(k).oninput=applyDisplay);$('op').oninput=draw;
$('auto').onclick=autoStretch;$('reset').onclick=()=>{S.lo=0;S.hi=255;['br','co','ga'].forEach(k=>$(k).value=0);applyDisplay();};
function zoomCentre(){return S.mx>=0?[S.mx,S.my]:[cv.width/2,cv.height/2];}
$('scrollmode').onclick=()=>setScroll(!scrollPans);
try{setScroll(localStorage.getItem('ann_scroll_pans')==='1');}catch(e){setScroll(false);}
$('fit').onclick=fit;$('prev').onclick=()=>openImg(S.idx-1);$('next').onclick=()=>openImg(S.idx+1);$('save').onclick=saveDone;
$('helpb').onclick=()=>{const h=$('help');h.style.display=h.style.display==='block'?'none':'block';};
window.addEventListener('keydown',e=>{
 if(e.target.tagName==='INPUT'&&e.target.type==='number')return;
 const k=e.key, ctrl=e.ctrlKey||e.metaKey;
 if(e.target.tagName==='INPUT'&&k.startsWith('Arrow'))return;   // arrows move a focused slider, not the image
 if(k===' '){space=true;e.preventDefault();return;}
 if(ctrl&&(k==='s'||k==='S')){e.preventDefault();saveDone();return;}
 if(ctrl&&(k==='z'||k==='Z')){e.preventDefault();e.shiftKey?redo():undo();return;}
 if(ctrl&&(k==='y'||k==='Y')){e.preventDefault();redo();return;}
 if(ctrl)return;
 const map={'1':()=>setTool('poly'),'2':()=>setTool('brush'),'3':()=>setTool('erase'),'[':()=>setSize(S.size-1),']':()=>setSize(S.size+1),
  'n':()=>openImg(S.idx+1),'N':()=>openImg(S.idx+1),'p':()=>openImg(S.idx-1),'P':()=>openImg(S.idx-1),'m':()=>toggle('mask'),'M':()=>toggle('mask'),
  'g':()=>toggle('gt'),'G':()=>toggle('gt'),'r':()=>toggle('pred'),'R':()=>toggle('pred'),'v':()=>toggle('saved'),'V':()=>toggle('saved'),
  'x':()=>toggle('grid'),'X':()=>toggle('grid'),'a':autoStretch,'A':autoStretch,'0':fit,'h':()=>$('helpb').click(),'H':()=>$('helpb').click(),
  'Enter':()=>{if(S.poly.length>=3)closePoly();},'Backspace':()=>{if(S.poly.length){S.poly.pop();draw();}},'Escape':()=>{S.poly=[];draw();},
  't':()=>setScroll(!scrollPans),'T':()=>setScroll(!scrollPans),
  '+':()=>zoomAt(...zoomCentre(),1.25),'=':()=>zoomAt(...zoomCentre(),1.25),'-':()=>zoomAt(...zoomCentre(),1/1.25),
  'ArrowLeft':()=>{S.ox+=60;draw();},'ArrowRight':()=>{S.ox-=60;draw();},'ArrowUp':()=>{S.oy+=60;draw();},'ArrowDown':()=>{S.oy-=60;draw();}};
 if(map[k]){e.preventDefault();map[k]();}
});
window.addEventListener('keyup',e=>{if(e.key===' ')space=false;});
window.addEventListener('beforeunload',e=>{if(S.idx>=0&&(S.dirty||S.poly.length>=3)){finishPolyIfAny();
 navigator.sendBeacon('/api/save/'+S.idx,new Blob([JSON.stringify({ops:S.ops,kind:'auto'})],{type:'application/json'}));}});
new ResizeObserver(resize).observe(cv);
(async()=>{const j=await (await api('/api/list')).json();S.items=j.items;document.title='Mask annotation - '+j.images_dir;
 setTool('poly');resize();drawList();await openImg(j.last_index||0);})();
</script></body></html>"""


# ------------------------------------------------------------------------------------- server
def serve(cfg):
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    proj = Project(cfg)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, body, ctype, headers=None):
            body = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)

        def fail(self, code, e):
            self.reply(code, json.dumps({"error": "%s" % e}), "application/json")

        def index(self, part):
            idx = int(part)
            if not 0 <= idx < len(proj.ids):
                raise IndexError("image index out of range")
            return idx

        def do_GET(self):
            try:
                parts = self.path.split("?")[0].strip("/").split("/")
                if parts == [""]:
                    self.reply(200, PAGE, "text/html; charset=utf-8")
                elif parts == ["api", "list"]:
                    self.reply(200, json.dumps(proj.listing()), "application/json")
                elif parts[:2] == ["api", "image"]:
                    self.reply(200, proj.display_png(self.index(parts[2])), "image/png")
                elif parts[:2] == ["api", "raw"]:
                    data, meta = proj.raw_values(self.index(parts[2]))
                    self.reply(200, data, "application/octet-stream",
                               {"X-W": meta["w"], "X-H": meta["h"], "X-C": meta["c"], "X-Dtype": meta["dtype"]})
                elif parts[:2] == ["api", "maskbits"] and parts[3] in ("gt", "pred", "saved"):
                    b = proj.mask_bits(self.index(parts[2]), parts[3])
                    if b is None:
                        self.fail(404, "no %s mask" % parts[3])
                    else:
                        self.reply(200, b, "application/octet-stream")
                elif parts[:2] == ["api", "annotation"]:
                    self.reply(200, json.dumps(proj.annotation(self.index(parts[2]))), "application/json")
                else:
                    self.fail(404, "not found")
            except Exception as e:
                self.fail(500, "%s: %s" % (type(e).__name__, e))

        def do_POST(self):
            try:
                parts = self.path.split("?")[0].strip("/").split("/")
                n = int(self.headers.get("Content-Length") or 0)
                data = json.loads(self.rfile.read(n) or b"{}")
                if parts[:2] == ["api", "render"]:
                    m = proj.render(self.index(parts[2]), data.get("ops", []))
                    self.reply(200, m.astype(np.uint8).tobytes(), "application/octet-stream")
                elif parts[:2] == ["api", "save"]:
                    res = proj.save(self.index(parts[2]), data.get("ops", []), data.get("kind", "auto"))
                    self.reply(200, json.dumps(res), "application/json")
                elif parts == ["api", "progress"]:
                    proj.set_last(self.index(data.get("last_index", 0)))
                    self.reply(200, "{}", "application/json")
                else:
                    self.fail(404, "not found")
            except ValueError as e:
                self.fail(400, e)
            except Exception as e:
                self.fail(500, "%s: %s" % (type(e).__name__, e))

    server = None
    for port in range(cfg.port, cfg.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)   # this PC only
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("No free port between %d and %d" % (cfg.port, cfg.port + 19))
    url = "http://127.0.0.1:%d/" % server.server_address[1]
    st = Counter(proj.status(i) for i in proj.ids)
    print("Annotation tool: %s" % url)
    print("  images : %s  (%d images; GT masks for %d, prediction masks for %d)"
          % (cfg.images_dir, len(proj.ids), len(proj.gt), len(proj.pred)))
    print("  output : %s" % proj.out)
    print("  status : %s" % ", ".join("%s %d" % (s.replace("_", " "), st[s]) for s in STATUSES))
    print("Press H in the page for shortcuts. Press Ctrl+C here to stop (your work is saved).")
    if not cfg.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Progress is in %s" % proj.progress_path)


def main():
    cfg = finish_config(parse_args())
    if cfg.inspect:
        inspect(cfg)
    else:
        serve(cfg)


if __name__ == "__main__":
    main()
