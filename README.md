# Lysosome Fluorescence Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

Automated image analysis tools for quantifying lysosomal distribution in confocal fluorescence microscopy images. Built during a research internship at **SUNY Upstate Medical University** (Summer 2026).

---

## Background

Lysosomes are the cell's recycling centers — membrane-bound compartments that break down waste material. Where they are positioned inside the cell is biologically significant: lysosomes that cluster near the nucleus behave differently from those spread toward the cell's outer edge, and this distribution shifts in diseases like cancer and neurodegeneration.

Measuring this distribution by hand across many cells from raw microscope images is time-consuming and inconsistent. These tools automate the entire pipeline — from raw `.lsm` confocal image to quantitative graphs and spreadsheet output — in a single click.

---

## Programs

| Program | Use case |
|---|---|
| `lysosome_fluorescence_v2.py` | Single cell — interactive GUI with radial profiling, zone analysis, heatmap, and pixel intensity |
| `lysosome_batch_v1.py` | Multi-cell field of view — automatically detects every nucleus, crops and analyses each cell independently, exports a summary CSV |

---

## How it works

1. **Segmentation** — The cell boundary is found by blurring the lysosome (Alexa 647) channel and thresholding. The nucleus is found separately on the DAPI channel using Otsu thresholding and connected component analysis. A user-drawn background ROI provides the median background level used to guide segmentation.

2. **Distance mapping** — Every pixel inside the cell is assigned a normalized distance value from 0 (nucleus boundary) to 1 (cell edge) using a distance transform. This turns the irregular cell shape into a consistent coordinate system.

3. **Radial profiling** — Pixels are grouped into 60 distance bins. For each bin, the program computes mean intensity and total integrated fluorescence density (IFD). This produces a radial profile showing how lysosome signal changes from the nucleus outward.

4. **Zone analysis** — The cytoplasm is divided into four 25%-width zones. The fraction of total fluorescence in each zone is reported as a bar chart, giving an at-a-glance summary of whether lysosomes are perinuclear or peripheral.

5. **Batch mode** — For multi-cell images, each nucleus is detected via connected components on the DAPI channel, cropped with padding, and run through the full pipeline independently.

---

## Sample Outputs

**Cell mask — automated segmentation of cell boundary (yellow) and nucleus (blue):**

![Mask Preview](samples/mask_preview.png)

**Fluorescence heatmap — lysosome signal intensity mapped across the cell:**

![Heatmap](samples/heatmap.png)

**Radial profile — mean fluorescence intensity from nucleus (0%) to cell edge (100%):**

![Radial Profile](samples/radial_profile.png)

**Zone analysis — percentage of total fluorescence per 25%-width zone:**

![Zones](samples/zones.png)

---

## Interpreting Results

- **Perinuclear pattern:** radial profile peaks near 0% distance; zone 1 (0–25%) dominates the bar chart. Lysosomes are clustered tightly around the nucleus.
- **Peripheral pattern:** radial profile peaks near 100% distance; zones 3–4 dominate. Lysosomes are spread toward the cell edge.
- **Uniform pattern:** flat radial profile; roughly equal zone bars. Lysosomes are evenly distributed throughout the cytoplasm.

---

## Installation

Requires Python 3.8+

```bash
pip install -r requirements.txt
```

Windows users can double-click `install_dependencies.bat` instead.

---

## Usage

**Single cell:**
```bash
python lysosome_fluorescence_v2.py
```
1. Click **Browse** and select your `.lsm` file
2. Draw rectangles over background areas (empty space with no cells)
3. Adjust settings if needed (default values work for most images)
4. Click **Run Analysis**
5. Results appear across tabs: Mask Preview, Heatmap, Radial Profile, Total Radial, Cumulative Radial, Zones

**Batch (multi-cell):**
```bash
python lysosome_batch_v1.py
```
1. Click **Browse** and select your multi-cell `.lsm` file
2. Adjust minimum nucleus size if small debris is being detected
3. Click **Run batch analysis**
4. Per-cell tabs populate as each cell finishes; `summary.csv` is saved to the output folder

A full illustrated step-by-step guide is included: `Lysosome_Software_Guide.pptx`

---

## Output Files

**`lysosome_fluorescence_v2.py`** saves next to the input LSM:
- `*_mask_preview_v2.png` — segmentation overlay
- `*_heatmap_v2.png` — fluorescence heatmap
- `*_radial_profile_v2.png` — mean intensity per distance bin
- `*_total_radial_v2.png` — total IFD per distance bin
- `*_cumulative_radial_v2.png` — cumulative % fluorescence by distance
- `*_zones_v2.png` — zone bar chart
- `*_zones_summary_v2.csv` — all measurements in spreadsheet form

**`lysosome_batch_v1.py`** saves to `<image_name>_batch/`:
- One set of PNGs per detected cell
- `summary.csv` — all cells with IFD, zone percentages, nucleus area, and cell area

---

## Tech Stack

Python · OpenCV · NumPy · Matplotlib · SciPy · Tkinter

---

## Author

**Elijah Poterbin** — [github.com/EliPots](https://github.com/EliPots)  
Computer Engineering, Binghamton University  
Research Intern, SUNY Upstate Medical University, Summer 2026
