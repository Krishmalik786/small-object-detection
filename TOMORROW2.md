# Annotation tool: `annotate_masks.py`

A small annotation tool for drawing **binary masks** (target / background) on infrared images, built for tiny and dim targets of 2–20 pixels. It runs as a page in your web browser, served by a Python script on your own PC. It needs no internet and sends nothing anywhere.

You use it for the images where neither the firm's GT nor DNANet's prediction was correct ("Both fail" in the review page), but it works for **any folder of images**.

---

## 1. What you need

- **The file** `annotate_masks.py`. Put it in the DNANet project folder, next to `dnanet_gt_check.py`. (It does not use DNANet's model; it only lives there so relative paths work the same way.)
- **Python 3.7 or newer**, with **numpy** and **Pillow**. Both are already needed by DNANet, so the same environment works. Nothing else: no Flask, no torch, no internet.
- **A web browser:** Chrome, Edge or Firefox.
- **A mouse** is easiest. A laptop trackpad also works (section 6).

---

## 2. Will it work with just a folder of images?

**Yes.** The images folder is the only thing it needs. Old masks are optional extras.

| You give it | What you get |
|---|---|
| Only an images folder | Full annotation: polygon, brush, eraser, undo, zoom, contrast, saving, progress. The overlays and "Start from GT / prediction" buttons just say "no mask exists". |
| Images + old GT and/or prediction masks | The same, plus: show the old masks as coloured overlays (G / R), and start from one and correct it instead of drawing from nothing. |

Images it reads: **.png, .bmp, .tif/.tiff, .jpg/.jpeg**, in grey or colour, **8-bit or 16-bit**, any size.

Limits to know:
- **One class only.** Each pixel is target or background, which is what DNANet uses. It can't label different object types.
- **Two images with the same name but different extensions** (e.g. `a.png` and `a.jpg` in the same folder): only one of them is used. Keep names unique.
- **An image called `progress`** stops the tool with a message, because that name is reserved for the progress file. Rename it in a copy of the folder.
- **16-bit colour TIFFs** can't be opened by Pillow. 16-bit grey images are fine.
- **Very large images** (e.g. 4000×3000) work, but each stroke takes a moment, because the whole mask is recomputed at full resolution each time. Normal infrared sizes (256 to 1280 wide) are instant.

---

## 3. Setting your folders

Open `annotate_masks.py`. The **CONFIG block** is at the very top, between the two lines of `=====`.

| Setting | What it means | Default |
|---|---|---|
| `images_dir` | **The folder with the images to annotate.** The only one you must get right. | `output_gt_check/review/both_failed/images` |
| `gt_masks_dir` | A folder with old GT masks, if you have one | `""` (not used) |
| `pred_masks_dir` | A folder with old predicted masks, if you have one | `""` (not used) |
| `mask_suffix` | Only if old masks have an extra ending, e.g. `frame_01_mask.png` → `_mask` | `""` |
| `source_output` | The GT-check output folder. Old GT and prediction masks are found there **automatically**, so for "Both fail" images you don't need to set the two mask folders. | `output_gt_check` |
| `output_dir` | Where your new masks are saved. `""` = automatic (section 4). | `""` |
| `port` | Only change it if the tool says the port is busy | `8770` |

The three usual cases:

1. **Your "Both fail" images, straight from the review page:** change nothing. The defaults point there.
2. **A folder you made yourself**, e.g. `manual_annotation/images` with `manual_annotation/gt_masks` and `manual_annotation/pred_masks`: set `images_dir`, `gt_masks_dir` and `pred_masks_dir` to those three folders. Set `source_output` to `""` if there's no `output_gt_check` folder.
3. **Just a folder of images, nothing else:** set `images_dir`, and set `source_output` to `""`.

Write paths with forward slashes, also on Windows: `D:/firm_data/images`. Relative paths (no drive letter) are taken from the folder the script is in.

Every setting can also be given on the command line instead of editing the file, e.g. `--images_dir manual_annotation/images`.

---

## 4. Where everything is stored, and how files are named

### The output folder

If `output_dir` is empty, the tool makes a folder called **`new_masks` next to your annotation folder**. The annotation folder is the parent of `images_dir` when `images_dir` is called `images`; otherwise it's `images_dir` itself.

| `images_dir` | New masks go to |
|---|---|
| `output_gt_check/review/both_failed/images` | `output_gt_check/review/new_masks/` |
| `manual_annotation/images` | `new_masks/` (next to `manual_annotation/`) |
| `my_frames` | `new_masks/` (next to `my_frames/`) |

The terminal prints the exact folder when the tool starts, and `--inspect` shows it too.

### What's inside, for an image called `frame_0042.bmp`

| File | What it is |
|---|---|
| `frame_0042.png` | **The binary mask.** Same width and height as the image. 0 = background (black), 255 = target (white). Grey PNG (mode `L`), no anti-aliasing, no other values. |
| `frame_0042.json` | Your editable drawing: the polygons and brush/eraser strokes, in order. Lets you reopen the image later and keep editing. You don't need it for training. |
| `progress.json` | One file for the whole folder: the status of every image, and which image you were on. |

### Is the mask name the same as the image name?

**Yes, the same name, always with `.png` at the end.**
- `frame_0042.png` → mask `frame_0042.png`
- `frame_0042.bmp` → mask `frame_0042.png`
- `Misc_12.jpg` → mask `Misc_12.png`

No suffix is added (no `_mask`), so image and mask pair up by name, which is what DNANet's loader and the GT-check script expect. The mask is always PNG even if the image is JPG, because JPG would add grey noise and the mask would no longer be strictly 0/255.

Keep images and masks in **separate folders** (`images/` and `masks/`): if the image is itself a `.png`, both files have exactly the same file name.

### What is never touched

Your images and old masks are **only read**. The tool refuses to start if the output folder is inside the images folder or one of the mask folders.

---

## 5. Starting the tool

1. Open a terminal in the DNANet folder and activate the DNANet Python environment.
2. **Check first (read-only):**
   `python annotate_masks.py --inspect`
   It shows the folder tree, how many images, their sizes and bit depth, how many old GT / prediction masks it found and their format, and where masks will be saved. Make sure the image count is what you expect and that it says "location OK". Nothing is written.
3. **Start:**
   `python annotate_masks.py`
   The browser opens the page. If it doesn't, copy the address the terminal prints (`http://127.0.0.1:8770/`) into the browser.
4. **Stop:** close the browser tab, then press **Ctrl+C** in the terminal. Your work is already saved.

---

## 6. The screen

- **Left:** the list of images with a coloured dot (grey = not started, orange = in progress, green = done, blue = no target), and "23 / 80 done" at the top. Click a name to open it.
- **Top:** tools, edit buttons, what to show, display sliders, save, next/previous. Press **H** for all shortcuts.
- **Middle:** the image.
- **Bottom:** the pixel under the mouse (x, y and its **real value**, 16-bit for 16-bit images), the zoom, the tool, how many pixels are in the mask, and whether there are unsaved changes.

**Moving around:**
- **Mouse:** the wheel zooms at the mouse position (up to 64×). Hold **Space** and drag, or drag with the right mouse button, to move the image. **0** fits the image to the screen.
- **Trackpad:** press **T** so the button shows "scroll: pan". Two-finger scroll then moves the image and **pinch zooms**. The setting is remembered.
- **Keyboard:** **+ / −** zoom, **arrow keys** move the image.
- The **pixel grid** appears from 8× zoom (toggle with **X**). Pixels are shown as sharp squares, never blurred.

**Seeing dim targets:** **A** = auto stretch, then the brightness, contrast and gamma sliders. These change only what you see, never the image or the saved mask. **reset** puts them back.

---

## 7. Drawing a mask

| Tool | Key | Best for |
|---|---|---|
| **Brush** | **2** | **Tiny targets (2–5 px).** Size 1 paints exactly one pixel per click; drag to paint a line. Change the size with **[ ]** or the size box. The square outline under the mouse shows exactly which pixels it will paint. |
| **Eraser** | **3** | Removing pixels; works like the brush |
| **Polygon** | **1** | Bigger targets. Click around the target; each point sits on a pixel centre. The **first point is a green ring**: when the mouse is close enough it fills in and says "click to CLOSE the polygon". You can also close with **Enter** or a double-click. **Backspace** removes the last point and **Esc** cancels. Every pixel whose centre is inside or on the outline becomes target. You can drag points afterwards to adjust them. |

**Step by step for a tiny target:**
1. Press **G** (old GT, green) or **R** (prediction, pink) to see where the target was supposed to be. Press again to hide.
2. Put the mouse on the target and zoom in to 16–32×.
3. Press **A**, then adjust contrast until you can see the target's edge. Check the pixel values in the status bar if unsure.
4. Press **2** (brush, size 1) and click each target pixel. Use **3** (eraser) for mistakes.
5. Several targets? Draw each one; they all go in the same mask.
6. **Ctrl+S** to save.

**Other buttons:**
- **Start from GT / Start from pred:** copies the old mask in as your starting point, then you correct it. Useful when it was almost right.
- **Undo / Redo:** Ctrl+Z / Ctrl+Y. Covers everything, including Start from, Clear and moving polygon points.
- **Clear:** empties the mask (asks first; Undo brings it back).
- **No target:** saves an empty mask for an image that really has no target, and marks it blue.

The **yellow overlay is exactly what will be saved.** It isn't drawn by the browser: after each stroke the Python side computes the mask at the image's real resolution, using the same code that writes the file, and sends it back.

---

## 8. Saving and checking

- **Ctrl+S** (or the green Save button) saves and marks the image **done**. A message shows the size, the pixel values (must be `[0, 255]`), and how many target pixels and separate targets were saved. If you try to save an empty mask, it asks whether you mean "No target".
- **Moving to another image** (N, P or clicking the list) saves automatically if you changed something. The image is marked **in progress** unless it was already done.
- **Every save is checked** by reading the file back from disk: same width and height as the image, only values 0 and 255, grey PNG. If anything is wrong, you get an error message and the image is not marked done.
- **V** shows the saved file as black and white, so you can see exactly what's on disk.

To check a mask outside the tool, open the PNG in any image viewer. It's black with white targets, but tiny targets are hard to see at normal size; zoom in.

---

## 9. Stopping and continuing

Everything is saved as you go: each mask when you save or move on, and the status and current image in `progress.json`. Close the browser and press Ctrl+C whenever you like.

Next time, run `python annotate_masks.py` again with the same settings. It opens the image you were on, shows every image's status, and your polygons and strokes are still editable. A "done" image can be reopened and changed at any time; save again and it stays done.

---

## 10. After annotating: building the full training set

The annotation tool only makes the masks for the images you annotate. To get one complete dataset:

- **Images:** `output_gt_check/final/images/` (from the review page) + the images you annotated (`output_gt_check/review/both_failed/images/`).
- **Masks:** `output_gt_check/final/masks/` + the masks from `new_masks/`, **except** images you marked "no target", if you don't want empty masks in training.

Because the names match, you can copy them together into one `images/` and one `masks/` folder. Check the counts match: one mask per image.

The `.json` files and `progress.json` are not needed for training. Leave them in `new_masks/` so you can still edit later.

---

## 11. Keyboard shortcuts (also under H in the page)

| Key | Action |
|---|---|
| 1 / 2 / 3 | polygon / brush / eraser |
| [ ] | brush size smaller / bigger |
| Enter, double-click, click the green ring | close polygon |
| Backspace / Esc | remove last polygon point / cancel polygon |
| Ctrl+Z / Ctrl+Y | undo / redo |
| Ctrl+S | save and mark done |
| N / P | next / previous image (auto-saves) |
| mouse wheel, + / − | zoom |
| Space+drag, right-drag, arrow keys | move the image |
| T | trackpad: two-finger scroll moves instead of zooms |
| 0 | fit to screen |
| A | auto stretch the display |
| M / G / R | show my mask / old GT / prediction |
| V | show the saved mask (black/white) |
| X | pixel grid |
| H | help |

---

## 12. If something goes wrong

| What you see | What it means / what to do |
|---|---|
| `images_dir does not exist` | The path in CONFIG is wrong. Check it with `--inspect`. |
| `No images found` | The folder exists, but has no .png/.bmp/.tif/.jpg files, or the images are in a subfolder. |
| `output_dir ... is inside the input folder` | Choose a different `output_dir`, outside your images and mask folders. |
| `No free port` | Another program is using ports 8770–8789. Start with `--port 9000`. |
| The browser doesn't open | Copy the printed address into the browser yourself. |
| "No GT mask for this image" when pressing G | No old mask was found for that image name. Check `gt_masks_dir` / `source_output` and `mask_suffix` with `--inspect`. |
| SAVE FAILED message | Nothing wrong was written as "done". Read the message (usually a full disk or a permission problem) and save again. |
| `ModuleNotFoundError: numpy` / `PIL` | You're in the wrong Python environment. Activate the DNANet one. |
