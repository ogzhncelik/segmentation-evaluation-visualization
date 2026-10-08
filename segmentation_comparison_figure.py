# -*- coding: utf-8 -*-
"""
Qualitative Segmentation Comparison Figure (Figure 4)
=====================================================
Produces two outputs:
  (a) figure4a_grid_<TILE>.png     -> Input | GT | Proposed(v9) | VGG16 | U-Net side by side
  (b) figure4b_boundary_<TILE>.png -> shorelines of all models in a single panel (closeness to GT)

Expected file name pattern (all in the same folder, 256x256, masks 0/255):
  input-<TILE>.tif      (RGB/multi-band)
  GT-<TILE>.tif
  v9-<TILE>.tif
  vgg16-<TILE>.tif
  unet-<TILE>_pred.tif

Requirements:  pip install rasterio matplotlib numpy
Usage:         Set the LOG/path settings, then:  python segmentation_comparison_figure.py
For another tile: change the TILE variable (or use the loop at the bottom).
"""

import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")  # if there is no display; you can remove this line in Jupyter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================ SETTINGS ============================
DIR = r"path\to\input"                # folder containing the .tif files
TILE = "tile_file"                    # tile name
RGB_BANDS = (1, 2, 3)           # input composite (1-indexed). If natural color does not appear, try (2,3,4) etc.
OUT_DIR = DIR                  # folder where the outputs will be saved
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

    # ---------- (b) boundary (shoreline) overlay ----------
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.imshow(rgb)
    specs = [("Ground Truth", gt, "yellow", 2.4), ("Proposed Model", v9, "red", 1.6),
             ("VGG16", vgg, "deepskyblue", 1.6), ("U-Net", unet, "lime", 1.6)]
    for _, m, col, lw in specs:
        ax2.contour(m, levels=[127.5], colors=[col], linewidths=lw)
    ax2.set_xticks([]); ax2.set_yticks([])
    leg = [Line2D([0], [0], color=c, lw=2.5, label=n) for n, _, c, _ in specs]
    ax2.legend(handles=leg, loc="lower left", fontsize=9, framealpha=0.85)
    fig2.tight_layout()
    out_b = os.path.join(OUT_DIR, f"figure4b_boundary_{tile}.png")
    fig2.savefig(out_b, dpi=DPI, bbox_inches="tight"); plt.close(fig2)

    print("Saved:", out_a, "|", out_b)

if __name__ == "__main__":
    make_figures(TILE)

