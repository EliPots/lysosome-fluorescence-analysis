# Lysosome Fluorescence Analysis

Automated tools for quantifying lysosomal distribution in confocal fluorescence microscopy images. Built during a research internship at **SUNY Upstate Medical University** (Summer 2026).

---

## What it does

Lysosomes are the cell's recycling centers. Where they sit inside a cell — clustered near the nucleus or spread toward the edges — is relevant to disease research. These tools automatically measure that distribution from raw confocal images, replacing a process that would otherwise take weeks by hand.

**Two programs are included:**

| Program | Use case |
|---|---|
| `lysosome_fluorescence_v2.py` | Single cell — detailed radial profiling, zone analysis, heatmap |
| `lysosome_batch_v1.py` | Multi-cell field of view — auto-detects every nucleus, runs full analysis on each cell, exports summary CSV |

---

## Outputs

- Radial fluorescence profile (mean intensity from nucleus → cell edge)
- Total & cumulative radial fluorescence
- Zone bar chart (% of total signal per 25%-width zone)
- Pixel intensity heatmap
- Per-cell summary `.csv`

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
Load your `.lsm` file, draw background ROI boxes, click Run.

**Batch (multi-cell):**
```bash
python lysosome_batch_v1.py
```
Load your `.lsm` file, adjust nucleus size threshold if needed, click Run.

A full step-by-step guide is included: `Lysosome_Software_Guide.pptx`

---

## Tech stack

Python · OpenCV · NumPy · Matplotlib · SciPy · Tkinter

---

## Author

**Elijah Poterbin** — [github.com/EliPots](https://github.com/EliPots)  
Computer Engineering, Binghamton University
