# -*- coding: utf-8 -*-
"""
Qualitative Segmentation Comparison Figure (Figure 4) — with AUTOMATIC ZOOM
===========================================================================
Produces two outputs:
  (a) figure4a_grid_<TILE>.png     -> Input | GT | Proposed | VGG16 | U-Net side by side
  (b) figure4b_boundary_<TILE>.png -> LEFT: full tile + zoom box, RIGHT: zoomed boundary

The zoom region is found automatically: the area where the models differ most from the GT.

Expected file name pattern (all in the same folder, 256x256, masks 0/255):
  input-<TILE>.tif      (RGB/multi-band)
  GT-<TILE>.tif
  v9-<TILE>.tif
  vgg16-<TILE>.tif
  unet-<TILE>_pred.tif

Requirements:  pip install rasterio matplotlib numpy scipy
Usage:         Set DIR and TILE, then:  python segmentation_comparison_figure_zoom.py
"""

import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")  # you can remove this line in Jupyter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.ndimage import uniform_filter

# ============================ SETTINGS ============================
DIR = r"path\to\input"                  # folder containing the .tif files
TILE = "tile_name"                      # tile name
RGB_BANDS = (1, 2, 3)           # input composite (1-indexed). If natural color does not appear, try (2,3,4).
OUT_DIR = DIR                   # outputs are saved to the input folder
ZOOM_HALF = 45                  # half side length of the zoom square (pixels)
DPI = 300
# ==================================================================

def load_mask(path):
    with rasterio.open(path) as ds:
        return ds.read(1)

def load_rgb(path, bands):
    with rasterio.open(path) as ds:
        a = ds.read()
    def n(x):
        x = x.astype(float)
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    return np.dstack([n(a[bands[0]-1]), n(a[bands[1]-1]), n(a[bands[2]-1])])

def make_figures(tile):
    p = lambda name: os.path.join(DIR, name)
    rgb  = load_rgb(p(f"input-{tile}.tif"), RGB_BANDS)
    gt   = load_mask(p(f"gt-{tile}.tif"))
    v9   = load_mask(p(f"v9-{tile}.tif"))
    vgg  = load_mask(p(f"vgg16-{tile}.tif"))
    unet = load_mask(p(f"unet-{tile}_pred.tif"))

    # ---------- (a) side-by-side grid ----------
    panels = [("Input", rgb, None), ("Ground Truth", gt, "gray"),
              ("Proposed Model", v9, "gray"), ("VGG16", vgg, "gray"), ("U-Net", unet, "gray")]
    fig, ax = plt.subplots(1, 5, figsize=(15, 3.2))
    for a, (title, im, cm) in zip(ax, panels):
        a.imshow(im) if cm is None else a.imshow(im, cmap=cm, vmin=0, vmax=255)
        a.set_title(title, fontsize=11); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    out_a = os.path.join(OUT_DIR, f"figure4a_grid_{tile}.png")
    fig.savefig(out_a, dpi=DPI, bbox_inches="tight"); plt.close(fig)

    # ---------- (b) boundary: full tile + AUTOMATIC ZOOM ----------
    specs = [("Ground Truth", gt, "yellow"), ("Proposed Model", v9, "red"),
             ("VGG16", vgg, "deepskyblue"), ("U-Net", unet, "lime")]

    # Automatic zoom region: the area where the models differ most from the GT
    dis = (((v9 > 127) != (gt > 127)) |
           ((vgg > 127) != (gt > 127)) |
           ((unet > 127) != (gt > 127)))
    dens = uniform_filter(dis.astype(float), size=40)
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    H, W = gt.shape
    y0, y1 = max(0, cy - ZOOM_HALF), min(H, cy + ZOOM_HALF)
    x0, x1 = max(0, cx - ZOOM_HALF), min(W, cx + ZOOM_HALF)

    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6))
    axL.imshow(rgb)
    for _, m, c in specs:
        axL.contour(m, levels=[127.5], colors=[c], linewidths=0.9)
    axL.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                            edgecolor="white", facecolor="none", lw=1.6))
    axL.set_xticks([]); axL.set_yticks([]); axL.set_title("Full tile", fontsize=11)

    axR.imshow(rgb[y0:y1, x0:x1])
    for _, m, c in specs:
        axR.contour(m[y0:y1, x0:x1], levels=[127.5], colors=[c], linewidths=2.2)
    axR.set_xticks([]); axR.set_yticks([]); axR.set_title("Zoomed boundary", fontsize=11)
    leg = [Line2D([0], [0], color=c, lw=3, label=n) for n, _, c in specs]
    axR.legend(handles=leg, loc="lower left", fontsize=9, framealpha=0.9)
    fig2.tight_layout()
    out_b = os.path.join(OUT_DIR, f"figure4b_boundary_{tile}.png")
    fig2.savefig(out_b, dpi=DPI, bbox_inches="tight"); plt.close(fig2)

    print("Saved:", out_a, "|", out_b)

if __name__ == "__main__":
    make_figures(TILE)

