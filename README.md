# segmentation-evaluation-visualization

Scripts for evaluating and presenting the results of semantic segmentation models on satellite imagery.
They produce publication-ready comparison figures (input, ground truth and several model predictions side by side), rank test tiles by prediction error, and plot training–validation curves from training logs.
Each script is standalone: set the paths at the top and run it.

## Scripts

| Script | What it does |
|---|---|
| `segmentation_comparison_figure.py` | For one tile, creates (a) a grid of Input / Ground Truth / Proposed Model / VGG16 / U-Net and (b) an overlay of every model's predicted shoreline on the input image. |
| `segmentation_comparison_figure_zoom.py` | Same comparison, plus an automatic zoom panel on the region where the models disagree most with the ground truth. |
| `segmentation_comparison_figure_batch.py` | Batch version: reads inputs, ground truth and predictions from separate folders and generates the zoomed comparison figures for a whole list of tiles. |
| `segmentation_pixel_error_ranking.py` | Computes the percentage of mismatched pixels between predictions and ground truth for every test tile, prints the best / median / worst tiles and saves all scores to a CSV file. |
| `training_curves_from_log.py` | Parses training logs (train/validation loss and IoU per epoch), detects multi-stage training boundaries and plots loss and IoU curves. |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python segmentation_comparison_figure_batch.py
```

Edit the input/output paths at the top of each script before running it.
Masks are expected as 256×256 GeoTIFFs with values 0 / 255.

## License

MIT
