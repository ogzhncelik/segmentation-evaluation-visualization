# -*- coding: utf-8 -*-
"""
Calculates an error score for all test tiles (WITH PROGRESS INDICATOR).
Finds the best/median/worst tiles based on how much the proposed model (v9) deviates from the ground truth.
"""
import os, glob, csv, time
import numpy as np
import rasterio

# ==================== FOLDERS ====================
GT_DIR = r"path\to\input\ground_truth"
V9_DIR = r"path\to\input\v9_predictions"
OUTPUT_CSV = r"path\to\output\tile_error_scores.csv"
# =================================================

def load_mask(path):
    with rasterio.open(path) as ds:
        return ds.read(1)

results = []
gt_files = glob.glob(os.path.join(GT_DIR, "*.tif"))
total = len(gt_files)
print(f"{total} GT masks found.\n")
print("Processing...\n")

start_time = time.time()

for idx, gt_path in enumerate(gt_files, start=1):
    name = os.path.basename(gt_path)
    v9_path = os.path.join(V9_DIR, name)
    if not os.path.exists(v9_path):
        continue
    try:
        gt = load_mask(gt_path) > 127
        v9 = load_mask(v9_path) > 127
        error_percent = 100.0 * np.sum(gt != v9) / gt.size
        results.append((name, error_percent))
    except Exception as e:
        print(f"ERROR {name}: {e}")

    # --- PROGRESS INDICATOR (every 200 tiles) ---
    if idx % 200 == 0 or idx == total:
        elapsed = time.time() - start_time
        percent = 100.0 * idx / total
        speed = idx / elapsed if elapsed > 0 else 0
        remaining = (total - idx) / speed if speed > 0 else 0
        print(f"  {idx}/{total} ({percent:5.1f}%)  |  elapsed: {elapsed:4.0f}s  |  estimated remaining: {remaining:4.0f}s")

# Sort
results.sort(key=lambda x: x[1])

print(f"\nA total of {len(results)} tiles processed.\n")
print("=== BEST 5 (lowest error) ===")
for name, y in results[:5]:
    print(f"  {y:6.2f}%  |  {name}")

print("\n=== MEDIAN (around the median) ===")
m = len(results)//2
for name, y in results[m-2:m+3]:
    print(f"  {y:6.2f}%  |  {name}")

print("\n=== WORST 5 (highest error) ===")
for name, y in results[-5:]:
    print(f"  {y:6.2f}%  |  {name}")

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["File", "Error_Percent"])
    for name, y in results:
        w.writerow([name, f"{y:.2f}"])
print(f"\nAll scores saved: {OUTPUT_CSV}")