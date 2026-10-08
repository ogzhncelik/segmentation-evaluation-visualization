# -*- coding: utf-8 -*-
"""
Training–Validation Curves (Figure 5) — for Kaggle tqdm logs
============================================================
Reads the following SUMMARY lines from the logs (NOT the tqdm batch lines):
    Train Loss: 0.1660 | Val Loss: 0.1453
    IoU: 0.9811 | Dice: 0.9903
Finds the stages from the "AşamaN Epoch .." and "AŞAMA N" lines
(the training logs contain Turkish stage markers: "Aşama" = "Stage").

USAGE:
  1) Set LOG_DIR to the folder containing the log files.
  2) python training_curves_from_log.py
Requirements: pip install matplotlib
"""
import os, glob, re
import matplotlib.pyplot as plt

# ============================ SETTINGS ============================
LOG_DIR = r"path\to\input"                          # folder containing the log files
FILE_PATTERN = "*.log"                              # .log extension
OUTPUT_PNG = r"path\to\output\training_curves.png"
TITLE = "Training–Validation Curves of the Proposed Model"
# ==================================================================

def _nat(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

LOG_FILES = sorted(glob.glob(os.path.join(LOG_DIR, FILE_PATTERN)),
                   key=lambda p: _nat(os.path.basename(p)))
print("Files to read (in order):", LOG_FILES)

def clean_line(ln):
    ln = ln.split("\r")[-1]                      # drop the tqdm CR fragments
    ln = re.sub(r"^\d+\.?\d*s\s+\d+\s+", "", ln)  # drop the "13724.9s 513 " Kaggle prefix
    return ln

def parse(files):
    rows = []; stage_boundaries = []; stage = None
    for fp in files:
        for raw in open(fp, encoding="utf-8", errors="ignore"):
            ln = clean_line(raw)
            m = re.search(r"A[şs]ama\s*([123]).*?Epoch\s*(\d+)/", ln, re.I)
            if m: stage = int(m.group(1))
            m = re.search(r"A[ŞS]AMA\s*([123])", ln)
            if m:
                ns = int(m.group(1))
                if ns != stage: stage_boundaries.append(len(rows))
                stage = ns
            m = re.search(r"Train Loss:\s*([\d.]+)\s*\|\s*Val Loss:\s*([\d.]+)", ln)
            if m:
                rows.append({"stage": stage, "train_loss": float(m.group(1)),
                             "val_loss": float(m.group(2)), "val_iou": None})
            m = re.search(r"^IoU:\s*([\d.]+)", ln)
            if m and rows and rows[-1]["val_iou"] is None:
                rows[-1]["val_iou"] = float(m.group(1))
    # also derive the stage boundaries from the stage changes in rows (fallback)
    for i in range(1, len(rows)):
        if rows[i]["stage"] != rows[i-1]["stage"]:
            stage_boundaries.append(i)
    return rows, sorted(set(b for b in stage_boundaries if 0 < b < len(rows)))

def main():
    rows, bounds = parse(LOG_FILES)
    if not rows:
        raise SystemExit("No summary line (Train Loss: .. | Val Loss: ..) found. "
                         "Check the LOG_DIR path.")
    n = len(rows); x = list(range(1, n+1))
    tr = [r["train_loss"] for r in rows]
    vl = [r["val_loss"] for r in rows]
    iou = [r["val_iou"] for r in rows]
    have_iou = any(v is not None for v in iou)
    print(f"Number of epochs read: {n} | stage boundaries (epoch): {bounds}")

    fig, axes = plt.subplots(1, 2 if have_iou else 1, figsize=(7*(2 if have_iou else 1), 4.5))
    if not have_iou: axes = [axes]

    ax = axes[0]
    ax.plot(x, tr, "-",  label="Training",   linewidth=1.8)
    ax.plot(x, vl, "--", label="Validation", linewidth=1.8)
    names = ["Stage 1","Stage 2","Stage 3","Stage 4"]
    for j,b in enumerate(bounds):
        ax.axvline(b+0.5, color="gray", ls=":", lw=1)
        ax.text(b+0.7, ax.get_ylim()[1], names[j+1] if j+1<len(names) else "", fontsize=8, color="gray", va="top")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("(a) Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    if have_iou:
        ax2 = axes[1]
        xv=[x[i] for i in range(n) if iou[i] is not None]
        yv=[iou[i] for i in range(n) if iou[i] is not None]
        ax2.plot(xv, yv, "--", color="tab:orange", label="Validation IoU", linewidth=1.8)
        for j,b in enumerate(bounds):
            ax2.axvline(b+0.5, color="gray", ls=":", lw=1)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("IoU"); ax2.set_title("(b) Validation IoU")
        ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle(TITLE, fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print("Saved:", OUTPUT_PNG)
    plt.show()

if __name__ == "__main__":
    main()