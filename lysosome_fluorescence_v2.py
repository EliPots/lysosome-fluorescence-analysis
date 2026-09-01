#!/usr/bin/env python3
"""
Lysosome Fluorescence Analysis v2
----------------------------------
Pure pixel-density approach — no lysosome counting.

Key design decisions:
  - Only the LARGEST cell in the image is analysed; everything else is zeroed.
  - Three cell masking methods are shown side-by-side so the user can choose
    which gives the cleanest boundary.
  - Zones start at the NUCLEUS BOUNDARY (not nucleus centre), so the interior
    of the nucleus is never included in zone 0-25%.
  - Saturated pixels (=255) are flagged and optionally excluded from IFD sums.
  - Zone sensitivity plot shows results at multiple min-intensity thresholds.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from pathlib import Path
import struct, csv, io

# ── Constants ─────────────────────────────────────────────────────────────────
LYSO_CH = 2   # Red  (cv2 BGR index 2) = lysosome marker
NUKE_CH = 0   # Blue (cv2 BGR index 0) = DAPI nucleus
N_BINS  = 4
ZONE_LABELS = ['0–25%', '25–50%', '50–75%', '75–100%']
SENS_THRESHOLDS = [5, 10, 20, 30, 50]   # min-intensity values for sensitivity plot

# ── Cell-mask reliability ─────────────────────────────────────────────────────
# Lysosome-signal-based cell boundary detection fails for perinuclear
# distributions: outer cytoplasm has little lysosome signal, so the detected
# mask is too small, compressing all zone boundaries inward.
#
# Correction strategy (validated on Image 18 geometry synthetic cells):
#   When detected cell/nucleus area ratio < NC_FALLBACK_THRESHOLD, replace
#   the lysosome-detected mask with an ellipse scaled from the nucleus shape
#   at NUC_CELL_RATIO × nucleus area.  This uses only the always-reliable
#   DAPI channel to anchor the cell boundary.
#
#   NC_FALLBACK_THRESHOLD = 8.0  — trigger level; only fires when detection
#       is badly wrong (e.g. perinuclear NC ≈ 6.2).  Does NOT fire for
#       uniform (NC ≈ 8.5) or mild perinuclear (NC ≈ 8.2) where the lysosome
#       mask is already close to the true boundary.
#   NUC_CELL_RATIO = 9.0         — target NC for the replacement ellipse,
#       calibrated to Image 18 cell geometry (true NC ≈ 9.4).
#
# Validation results:
#   Perinuclear (exp-5)  → 2.3 pp   (vs 4.6 pp with target=8.0)
#   Mild perinuclear     → 3.4 pp   (unchanged — fallback does not fire)
#   Uniform              → 4.6 pp   (unchanged — fallback does not fire)
#   Maximum error        → 4.6 pp
#
# Adjust NUC_CELL_RATIO for your cell type if typical NC ratios differ.
NUC_CELL_RATIO        = 9.0   # target NC for nucleus-scaled ellipse
NC_FALLBACK_THRESHOLD = 8.0   # trigger replacement when detected NC < this
MIN_CELL_TO_NUC_RATIO = NC_FALLBACK_THRESHOLD   # backward compatibility alias

# ── LSM helpers ───────────────────────────────────────────────────────────────

def load_lsm(path):
    ok, frames = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)
    if not ok or not frames:
        raise IOError(f"Cannot read {path}")
    real = [f for f in frames if f.shape[0] > 128]
    arr  = np.stack([f if f.ndim==3 else np.stack([f,f,f],axis=-1) for f in real], axis=0)
    return arr.transpose(0,3,1,2)   # Z,C(BGR),Y,X

def get_voxel_um(path):
    with open(str(path),'rb') as f: raw = f.read()
    endian = '<' if raw[:2]==b'II' else '>'
    offset = struct.unpack_from(endian+'I',raw,4)[0]
    n      = struct.unpack_from(endian+'H',raw,offset)[0]
    pos, blk = offset+2, None
    for _ in range(n):
        tag = struct.unpack_from(endian+'H',raw,pos)[0]
        val = struct.unpack_from(endian+'I',raw,pos+8)[0]
        if tag==34412: blk=val; break
        pos+=12
    if blk is None: return 1.0,1.0
    vx = struct.unpack_from(endian+'d',raw,blk+40)[0]*1e6
    vy = struct.unpack_from(endian+'d',raw,blk+48)[0]*1e6
    return (vx,vy) if (vx>0 and vy>0) else (1.0,1.0)

def max_project(arr4d, ch):
    return arr4d[:,ch].max(axis=0)

# ── Segmentation helpers ──────────────────────────────────────────────────────

def keep_largest(mask):
    """Return a copy of mask with only the largest connected component kept."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def _nucleus_scaled_cell_mask(nuke_mask, target_nc_ratio):
    """
    Construct a cell mask by fitting an ellipse to the nucleus and scaling
    its semi-axes by sqrt(target_nc_ratio).  This produces a cell boundary
    with the correct area ratio relative to the nucleus while preserving
    nuclear shape and orientation — far more accurate than an isotropic
    expansion when the lysosome-based cell mask has failed.

    Returns None if fitEllipse cannot be applied (fewer than 5 contour points).
    """
    ct, _ = cv2.findContours(nuke_mask.astype(np.uint8),
                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ct:
        return None
    big = max(ct, key=cv2.contourArea)
    if len(big) < 5:
        return None

    ell = cv2.fitEllipse(big)
    cx, cy  = ell[0]
    fw, fh  = ell[1]   # full axis lengths (diameters)
    angle   = ell[2]
    scale   = float(np.sqrt(target_nc_ratio))

    H, W = nuke_mask.shape
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask,
                (int(round(cx)), int(round(cy))),
                (max(1, int(round(fw / 2 * scale))),
                 max(1, int(round(fh / 2 * scale)))),
                angle, 0, 360, 1, -1)
    # Ensure nucleus is always inside the cell mask
    mask = np.maximum(mask, nuke_mask.astype(np.uint8))
    return mask


def subtract_background(img, bg_val):
    return np.clip(img.astype(np.int32) - int(round(bg_val)), 0, 255).astype(np.uint8)


def get_nucleus_mask(arr4d):
    """
    Segment nucleus from DAPI channel.
    Heavy Gaussian blur smooths out internal heterochromatin spots into one
    blob, a floor of 50 eliminates diffuse dye background, Otsu then cleanly
    separates the nucleus, and a large closing fills any gaps for a smooth
    ellipse-shaped mask.
    Returns (nuke_mask uint8, centroid (cy, cx)) or (zeros, None).
    """
    nuke_mp = max_project(arr4d, NUKE_CH)
    blurred = cv2.GaussianBlur(nuke_mp, (61, 61), 15)
    clipped = np.clip(blurred.astype(np.int32) - 50, 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(clipped, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nuke_mask = keep_largest(thresh.astype(np.uint8))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    nuke_mask = cv2.morphologyEx(nuke_mask, cv2.MORPH_CLOSE, kern)
    if nuke_mask.sum() == 0:
        return nuke_mask, None
    M  = cv2.moments(nuke_mask)
    cy = M['m01'] / M['m00']
    cx = M['m10'] / M['m00']
    return nuke_mask, (float(cy), float(cx))


def save_nucleus_mask_preview(nuke_mp, nuke_mask, out_path):
    """Save a zoomed preview of the nucleus mask for inspection after each run."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ys, xs = np.where(nuke_mask)
    if len(ys) == 0:
        return
    pad = 80
    h, w = nuke_mask.shape
    y0, y1 = max(0, ys.min()-pad), min(h, ys.max()+pad)
    x0, x1 = max(0, xs.min()-pad), min(w, xs.max()+pad)

    dapi_crop = nuke_mp[y0:y1, x0:x1]
    mask_crop = nuke_mask[y0:y1, x0:x1]

    dapi_norm = cv2.normalize(dapi_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dapi_rgb  = cv2.cvtColor(dapi_norm, cv2.COLOR_GRAY2RGB)
    overlay   = dapi_rgb.copy(); overlay[mask_crop == 1] = [255, 60, 60]
    blended   = cv2.addWeighted(dapi_rgb, 0.4, overlay, 0.6, 0)
    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 0, 0), 3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('black')
    for ax in axes: ax.axis('off')
    axes[0].imshow(blended)
    axes[0].set_title('Nucleus mask on DAPI (zoomed)', color='white', fontsize=12)
    axes[1].imshow(mask_crop, cmap='gray')
    axes[1].set_title('Mask alone (zoomed)', color='white', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='black')
    plt.close(fig)


def make_cell_mask(img_sub, blur_sigma=8, threshold=30):
    """
    Trace the cell boundary from the background-subtracted lysosome image.

    After correct background subtraction pixels genuinely outside the cell
    are ≈ 0.  A moderate Gaussian blur (sigma=8 px) merges individual
    lysosome puncta into a continuous, cell-shaped signal while keeping
    enough amplitude at the cell periphery to be detected.  An absolute
    threshold (default 30 counts) is applied directly to the blurred image —
    this avoids the normalization + Otsu approach used previously, which
    rounded any pixel below ½ count to 0 after uint8 casting, making the
    dim outer cytoplasm indistinguishable from true extracellular background.
    A large morphological close (51 px kernel) bridges gaps between sparse
    peripheral spots and produces a solid cell outline.

    Parameters
    ----------
    img_sub   : background-subtracted lysosome max-projection (uint8)
    blur_sigma: Gaussian sigma in pixels (default 8)
    threshold : minimum blurred value to count as cell interior (default 30)
    """
    ksize   = int(blur_sigma * 6) | 1
    blurred = cv2.GaussianBlur(img_sub.astype(np.float32), (ksize, ksize), blur_sigma)
    ma      = (blurred > float(threshold)).astype(np.uint8) * 255
    kern    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    ma      = cv2.morphologyEx(ma, cv2.MORPH_CLOSE, kern)
    return keep_largest(ma)


def compute_zone_density(img_sub, cell_mask, nuke_mask,
                          min_intensity=0, n_bins=N_BINS, exclude_sat=False,
                          correct_nc=True):
    """
    IFD per zone, measured from the NUCLEUS BOUNDARY outward to the cell edge.
    The interior of the nucleus is excluded from all zones.

    Saturated pixels (=255):
      - Always reported separately.
      - If exclude_sat=True they are removed before summation; the IFD sum then
        represents only unsaturated signal. This is conservative — it avoids
        letting clipped pixels skew the result but underestimates true IFD.
      - If exclude_sat=False they are included at face value (255).  This
        overestimates slightly in saturated regions but keeps all signal.

    Returns
    -------
    ifd_z, area_z, total_ifd, norm_dist_map, n_sat_in_cell
    """
    # ── Cell-mask reliability check ───────────────────────────────────────────
    # A strongly perinuclear distribution produces a lysosome mask with
    # cell/nucleus area ratio well below the biological norm (e.g. NC ≈ 6.2
    # detected vs NC ≈ 9.4 true).  When the ratio falls below
    # NC_FALLBACK_THRESHOLD we REPLACE the mask with a nucleus-scaled ellipse
    # at NUC_CELL_RATIO.  Replacement (not union) is used because the union
    # of the small irregular lyso mask with the ellipse produces a composite
    # shape whose max_dist differs from either component alone, causing
    # larger zone errors than either mask independently.
    # correct_nc=False skips this step (used when the mask was already chosen
    # deliberately by the user, e.g. pure lyso or pure ellipse modes).
    nuc_area = float(nuke_mask.sum())
    if correct_nc and nuc_area > 0:
        detected_nc = float(cell_mask.sum()) / nuc_area
        if detected_nc < NC_FALLBACK_THRESHOLD:
            alt = _nucleus_scaled_cell_mask(nuke_mask, NUC_CELL_RATIO)
            if alt is not None:
                cell_mask = alt

    # Distance from nucleus boundary outward
    # distanceTransform on pixels NOT inside nucleus = how far each pixel is
    # from the nearest nucleus boundary pixel.
    nuke_inv = (nuke_mask == 0).astype(np.uint8)   # 0 inside nucleus (source), 1 outside
    dist_from_edge = cv2.distanceTransform(nuke_inv, cv2.DIST_L2, 5)

    # Normalise so 0 = nucleus boundary, 1 = furthest point inside cell
    cytoplasm = (cell_mask > 0) & (nuke_mask == 0)
    max_dist  = float(dist_from_edge[cytoplasm].max()) if cytoplasm.sum() > 0 else 1.0
    norm_dist = dist_from_edge / max_dist if max_dist > 0 else np.zeros_like(dist_from_edge)

    # Count saturated pixels inside cell (outside nucleus)
    sat_in_cell = (img_sub == 255) & cytoplasm
    n_sat = int(sat_in_cell.sum())

    # Working image
    work = img_sub.copy().astype(np.float32)
    if exclude_sat:
        work[img_sub == 255] = 0   # zero out saturated pixels

    valid = cytoplasm & (img_sub >= min_intensity) & (norm_dist <= 1.0)
    if exclude_sat:
        valid = valid & (img_sub < 255)

    ifd_z  = np.zeros(n_bins)
    area_z = np.zeros(n_bins)
    for z in range(n_bins):
        lo, hi    = z/n_bins, (z+1)/n_bins
        ring      = valid & (norm_dist >= lo) & (norm_dist < hi)
        ifd_z[z]  = float(work[ring].sum())
        area_z[z] = float(ring.sum())

    return ifd_z, area_z, float(ifd_z.sum()), norm_dist, n_sat


def _mask_by_any_signal(lyso_raw, nuke_mask):
    """
    Any-signal cell mask: everything connected to this cell's nucleus that
    gives lysosome signal is included — soma, axon, and all processes.

    1. Light Gaussian blur smooths shot noise on the raw channel.
    2. Threshold at raw value 1 — anything above zero is cell.
       (Equivalent to the viewer with max display = ~10: all signal visible.)
    3. Nucleus always included.
    4. Morphological closing fills internal holes to make a solid mask.
    """
    nuke_bin = (nuke_mask > 0).astype(np.uint8)

    # ── Raw signal threshold ─────────────────────────────────────────────────
    # Threshold at 1: anything above background is cell.
    blur = cv2.GaussianBlur(lyso_raw, (21, 21), 7)
    _, binary = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
    binary = np.maximum(binary, nuke_bin * 255)

    # ── Close small gaps so the full cell forms one connected region ─────────
    kern_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern_sm)

    # ── Keep only the component containing this cell's nucleus ───────────────
    # Drops disconnected blobs from other cells / debris elsewhere in the image.
    _, labels = cv2.connectedComponents(closed.astype(np.uint8))
    nuke_labels = labels[nuke_bin > 0]
    unique, counts = np.unique(nuke_labels[nuke_labels > 0], return_counts=True)
    if len(unique) == 0:
        cell_mask = closed
    else:
        cell_mask = (labels == unique[np.argmax(counts)]).astype(np.uint8) * 255

    kern_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
    cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_CLOSE, kern_lg)
    return cell_mask


def _mask_by_watershed(lyso_sub, nuke_mask):
    """
    Nucleus-seeded watershed.  The nucleus is labelled 'definite cell'; pixels
    far from the nucleus with near-zero intensity are labelled 'definite
    background'; watershed fills the ambiguous middle zone by expanding from
    both seeds until the regions collide at the lowest-intensity ridge.
    Unlike threshold methods this can capture sparse outer cytoplasm because it
    doesn't require signal to exceed a fixed floor — it just needs the cell
    interior to be *less dark* than true background.
    """
    H, W = lyso_sub.shape
    blur = cv2.GaussianBlur(lyso_sub.astype(np.float32), (31, 31), 10)

    # Distance of every pixel from the nearest nucleus pixel
    nuke_bin = (nuke_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(1 - nuke_bin, cv2.DIST_L2, 5)

    # Markers: 1 = cell seed (nucleus), 2 = background seed, 0 = unknown
    markers = np.zeros((H, W), dtype=np.int32)
    markers[nuke_bin > 0] = 1
    # Definite background: far from nucleus AND essentially black
    bg_cut = max(float(lyso_sub[nuke_bin == 0].min()) + 1.0, 3.0) \
             if (nuke_bin == 0).any() else 3.0
    markers[(dist > 250) & (lyso_sub < bg_cut)] = 2

    # Watershed needs a 3-channel uint8 image
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr  = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    cv2.watershed(bgr, markers)

    # Keep only the region the watershed grew from the nucleus seed
    cell = (markers == 1).astype(np.uint8)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kern)
    return keep_largest(cell * 255)


def _mask_by_spot_hull(lyso_sub, nuke_mask, spot_thr=15, pad_px=40):
    """
    Spot convex hull.  Detect every pixel above spot_thr (individual lysosome
    spots), take the convex hull of all detected spots plus the nucleus, then
    dilate by pad_px to fill the inter-spot gaps.  Unlike blur-based methods
    the boundary is set by *where spots actually are* rather than where blurred
    signal exceeds a floor — one peripheral spot pushes the boundary out to
    the cell edge even if the region between it and the nucleus is dim.
    """
    H, W = lyso_sub.shape
    fg  = ((lyso_sub > spot_thr) | (nuke_mask > 0)).astype(np.uint8)
    pts = np.column_stack(np.where(fg > 0))          # (N, 2) as (row, col)
    if len(pts) < 5:
        return None

    hull_mask = np.zeros((H, W), dtype=np.uint8)
    pts_xy    = pts[:, ::-1].reshape(-1, 1, 2).astype(np.int32)   # (x, y)
    hull      = cv2.convexHull(pts_xy)
    cv2.fillConvexPoly(hull_mask, hull, 1)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (2*pad_px+1, 2*pad_px+1))
    hull_mask = cv2.dilate(hull_mask, kern)
    return keep_largest(hull_mask * 255)


def _mask_by_nuc_expand(nuke_mask, target_nc=NUC_CELL_RATIO):
    """
    Nucleus distance-expansion.  Grow the nucleus outward pixel-by-pixel
    (using the exact Euclidean distance transform) until the enclosed area
    equals target_nc × nucleus_area.  Completely ignores lysosome signal —
    always produces the same boundary for a given nucleus regardless of
    lysosome distribution.  The shape follows the true nucleus contour (not a
    fitted ellipse), so irregular nuclei are handled correctly.
    """
    nuke_bin = (nuke_mask > 0).astype(np.uint8)
    nuc_area = float(nuke_bin.sum())
    if nuc_area == 0:
        return None
    target_area = nuc_area * target_nc

    dist = cv2.distanceTransform(1 - nuke_bin, cv2.DIST_L2, 5)

    # Binary-search for the radius d such that |pixels with dist ≤ d| ≈ target_area
    lo, hi = 0.0, float(dist.max())
    for _ in range(40):
        mid  = (lo + hi) / 2.0
        area = float(((dist <= mid) | nuke_bin.astype(bool)).sum())
        if area < target_area:
            lo = mid
        else:
            hi = mid

    cell = ((dist <= hi) | nuke_bin.astype(bool)).astype(np.uint8)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kern)
    return keep_largest(cell * 255)


def _all_cell_masks(lyso_raw, lyso_sub, nuke_mask):
    """
    Return four genuinely different cell-masking algorithms for visual
    comparison.  Each uses a fundamentally different strategy:
      any_signal  – CLAHE + Otsu on raw image (contrast-enhanced, like the viewer)
      watershed   – topology-based, nucleus-seeded gradient fill
      spot_hull   – point-cloud (convex hull of detected lysosome spots)
      nuc_expand  – morphological (DAPI only, no lyso signal used)
    """
    nuc_area = float((nuke_mask > 0).sum())
    results  = []

    # 1 – CLAHE + Otsu on raw image (contrast-enhanced, nucleus-seeded)
    m1  = _mask_by_any_signal(lyso_raw, nuke_mask)
    nc1 = m1.sum() / nuc_area if nuc_area else 0
    results.append(('CLAHE + Otsu\n(raw, high contrast)', 'any_signal', m1, nc1))

    # 2 – Watershed seeded from nucleus
    m2  = _mask_by_watershed(lyso_sub, nuke_mask)
    nc2 = m2.sum() / nuc_area if (nuc_area and m2 is not None) else 0
    results.append(('Watershed\n(nucleus-seeded)', 'watershed', m2 if m2 is not None else np.zeros_like(nuke_mask, dtype=np.uint8), nc2))

    # 3 – Convex hull of detected lysosome spots
    m3  = _mask_by_spot_hull(lyso_sub, nuke_mask)
    nc3 = m3.sum() / nuc_area if (nuc_area and m3 is not None) else 0
    results.append(('Spot Convex Hull\n(+ dilation)', 'spot_hull', m3 if m3 is not None else np.zeros_like(nuke_mask, dtype=np.uint8), nc3))

    # 4 – Nucleus distance-expansion (DAPI only)
    m4  = _mask_by_nuc_expand(nuke_mask, NUC_CELL_RATIO)
    nc4 = m4.sum() / nuc_area if (nuc_area and m4 is not None) else 0
    results.append((f'Nucleus Expand\n(DAPI only, NC={NUC_CELL_RATIO})', 'nuc_expand', m4 if m4 is not None else np.zeros_like(nuke_mask, dtype=np.uint8), nc4))

    return results


def fig_mask_comparison(lyso_raw, lyso_sub, nuke_mask, stem):
    """
    Side-by-side preview of all cell mask methods overlaid on the lysosome image.
    Green contour = cell boundary for that method.
    Cyan contour  = nucleus boundary (same in every panel).
    NC ratio and a warning if it is too low are printed in each subtitle.
    """
    masks = _all_cell_masks(lyso_raw, lyso_sub, nuke_mask)

    # Crop to the cell region so the image is not tiny
    all_m = [m for *_, m, _ in masks if m is not None]
    all_m.append(nuke_mask)
    union = np.zeros_like(nuke_mask, dtype=np.uint8)
    for m in all_m:
        union = np.maximum(union, m.astype(np.uint8))
    ys, xs = np.where(union)
    pad  = 60
    H, W = lyso_sub.shape
    y0 = max(0, ys.min() - pad);  y1 = min(H, ys.max() + pad)
    x0 = max(0, xs.min() - pad);  x1 = min(W, xs.max() + pad)

    fig, axes = plt.subplots(1, len(masks), figsize=(5*len(masks), 6))
    fig.patch.set_facecolor('#111')
    vmax = max(int(lyso_sub[y0:y1, x0:x1].max()), 10)

    for ax, (label, _key, mask, nc) in zip(axes, masks):
        ax.set_facecolor('#000')
        crop = lyso_sub[y0:y1, x0:x1].astype(np.float32)
        ax.imshow(crop, cmap='inferno', vmin=0, vmax=vmax, origin='upper')

        # Nucleus outline – cyan
        nk = nuke_mask[y0:y1, x0:x1]
        try: ax.contour(nk.astype(float), [0.5], colors=['cyan'], linewidths=1.2)
        except Exception: pass

        # Cell mask outline – green (or red if mask failed)
        if mask is not None and mask.sum() > 0:
            mc = mask[y0:y1, x0:x1]
            try: ax.contour(mc.astype(float), [0.5], colors=['#00ff44'], linewidths=2.0)
            except Exception: pass
            warn = '  ⚠ very small' if nc < NC_FALLBACK_THRESHOLD else ''
            subtitle = f'NC = {nc:.1f}{warn}'
        else:
            subtitle = '(mask failed)'

        ax.set_title(f'{label}\n{subtitle}', color='white', fontsize=9, pad=5)
        ax.axis('off')

    fig.suptitle(f'{stem}  —  Cell Mask Comparison\n'
                 f'Green = cell boundary  |  Cyan = nucleus  |  '
                 f'Select best method, then Run Analysis',
                 color='white', fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def compute_radial_profile(img_sub, cell_mask, nuke_mask, norm_dist,
                            min_intensity=0, n_pts=60, exclude_sat=False):
    cytoplasm = (cell_mask > 0) & (nuke_mask == 0)
    work = img_sub.copy().astype(np.float32)
    if exclude_sat:
        work[img_sub == 255] = 0
    valid  = cytoplasm & (img_sub >= min_intensity) & (norm_dist <= 1.0)
    if exclude_sat:
        valid = valid & (img_sub < 255)
    bins    = np.linspace(0, 1.0, n_pts+1)
    centers = (bins[:-1]+bins[1:])/2
    means   = np.zeros(n_pts)
    totals  = np.zeros(n_pts)
    for i in range(n_pts):
        ring = valid & (norm_dist >= bins[i]) & (norm_dist < bins[i+1])
        n = ring.sum()
        if n > 0:
            means[i]  = float(work[ring].mean())
            totals[i] = float(work[ring].sum())
    return centers, means, totals

# ── Figure generators ──────────────────────────────────────────────────────────

def _colorbar_white(cb):
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

def fig_heatmap(lyso_raw, lyso_sub, bg_val, stem, vx, vy,
                cell_mask=None, nuke_mask=None):
    slc  = cv2.GaussianBlur(lyso_sub.astype(np.float32),(5,5),1.5)
    bgm  = lyso_raw < max(int(bg_val)+3, 10)
    plot = np.ma.masked_where(bgm, slc)
    vmax = max(int(lyso_sub.max()), 10)

    fig, ax = plt.subplots(figsize=(8,8))
    fig.patch.set_facecolor('black'); ax.set_facecolor('black')
    cmap = plt.cm.jet.copy(); cmap.set_bad('black')
    im = ax.imshow(plot, cmap=cmap, vmin=0, vmax=vmax,
                   interpolation='bilinear', origin='upper')
    signal = np.where(bgm, np.nan, slc)
    try:
        levels = list(range(10, vmax, max(vmax//10, 5)))
        ax.contour(signal, levels=levels, colors='white', linewidths=0.5, alpha=0.5)
    except Exception: pass

    # Saturated = white
    sat = (lyso_raw==255)
    sat_rgba = np.zeros((*lyso_raw.shape,4), dtype=np.float32)
    sat_rgba[sat] = [1,1,1,1]
    ax.imshow(sat_rgba, origin='upper', interpolation='nearest')

    # Cell boundary = green
    if cell_mask is not None:
        try: ax.contour(cell_mask.astype(float),[0.5],colors=['#00ff88'],linewidths=1.5)
        except Exception: pass
    # Nucleus boundary = cyan
    if nuke_mask is not None:
        try: ax.contour(nuke_mask.astype(float),[0.5],colors=['cyan'],linewidths=1.2)
        except Exception: pass

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label('Intensity (BG subtracted)', color='white', fontsize=10)
    _colorbar_white(cb)
    h,w = lyso_raw.shape
    ax.set_title(
        f'{stem} — Lysosome heatmap\n'
        f'BG={bg_val:.0f}  |  White=255  |  Green=cell  |  Cyan=nucleus\n'
        f'FOV {w*vx:.1f}×{h*vy:.1f} µm  |  {int(sat.sum())} saturated px',
        color='white', fontsize=9, pad=8)
    ax.axis('off'); plt.tight_layout()
    return fig


def fig_compare_radial(c1, m1, stem1, c2, m2, stem2):
    """Overlay radial profiles for two cells on one chart."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(c1, m1, color='#4E79A7', lw=2.5, label=stem1)
    ax.fill_between(c1, m1, alpha=0.15, color='#4E79A7')
    ax.plot(c2, m2, color='#E15759', lw=2.5, label=stem2)
    ax.fill_between(c2, m2, alpha=0.15, color='#E15759')
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    ax.set_xlabel('Normalised distance  (0 = nucleus boundary,  1 = cell edge)', fontsize=11)
    ax.set_ylabel('Mean pixel intensity (BG subtracted)', fontsize=11)
    ax.set_title('Radial fluorescence density — Cell comparison', fontsize=12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['Nucleus\nboundary', '25%', '50%', '75%', 'Cell\nedge'])
    for x in [0.25, 0.5, 0.75]:
        ax.axvline(x, color='grey', lw=0.8, linestyle=':', alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout(); return fig


def fig_compare_zones(ifd_z1, total1, stem1, ifd_z2, total2, stem2):
    """Grouped bar chart comparing zone distributions for two cells."""
    pct1 = ifd_z1 / total1 * 100 if total1 > 0 else np.zeros(N_BINS)
    pct2 = ifd_z2 / total2 * 100 if total2 > 0 else np.zeros(N_BINS)
    x = np.arange(N_BINS); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, pct1, w, label=stem1, color='#4E79A7', edgecolor='black', lw=0.7)
    b2 = ax.bar(x + w/2, pct2, w, label=stem2, color='#E15759', edgecolor='black', lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(ZONE_LABELS, fontsize=12)
    ax.set_xlabel('Distance from nucleus boundary (% of cytoplasm width)', fontsize=11)
    ax.set_ylabel('% of total IFD in zone', fontsize=11)
    ax.set_title('Zone fluorescence density — Cell comparison', fontsize=12)
    ymax = max(int(max(pct1.max(), pct2.max()) * 1.3), 10)
    ax.set_ylim(0, ymax)
    for bar, pv in [(b, v) for bars, vals in [(b1, pct1), (b2, pct2)]
                   for b, v in zip(bars, vals)]:
        if pv > 0:
            ax.text(bar.get_x() + bar.get_width()/2, pv + ymax*0.01,
                    f'{pv:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout(); return fig


def fig_pixel_histogram(img_sub, cell_mask, nuke_mask, min_intensity, stem):
    cytoplasm = (cell_mask>0) & (nuke_mask==0)
    pixels = img_sub[cytoplasm & (img_sub>=min_intensity)].astype(np.float32)
    n_sat  = int((pixels==255).sum())

    fig, ax = plt.subplots(figsize=(8,5))
    if pixels.size == 0:
        ax.text(0.5,0.5,'No signal above threshold',ha='center',va='center',
                transform=ax.transAxes,fontsize=13)
        plt.tight_layout(); return fig

    ax.hist(pixels, bins=np.arange(0,257,5), weights=pixels,
            color='#E15759', edgecolor='none', alpha=0.85)
    ax.axvline(255, color='white', lw=1.5, linestyle='--',
               label=f'Saturated (255): {n_sat} px')
    ax.set_xlim(0,255)
    ax.set_xlabel('Pixel intensity (0–255)', fontsize=12)
    ax.set_ylabel('Total IFD contributed by bin', fontsize=12)
    ax.set_title(
        f'{stem} — Pixel intensity (cytoplasm only, min={min_intensity})\n'
        f'Total IFD={pixels.sum():,.0f}  |  '
        f'Signal px={pixels.size:,}  |  '
        f'Saturated={n_sat} ({n_sat/pixels.size*100:.2f}%)',
        fontsize=10)
    med = float(np.median(pixels))
    ax.axvline(med, color='navy', lw=2, linestyle='-',
               label=f'Median: {med:.1f}')
    ax.legend(fontsize=9)

    # Annotation about saturation behaviour
    if n_sat > 0:
        ax.text(0.98, 0.97,
                '⚠ Saturated pixels capped at 255.\n'
                'True IFD may be higher in these regions.\n'
                'Use "Exclude 255 px" to remove them.',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='#f39c12',
                bbox=dict(boxstyle='round,pad=0.3', fc='#2c2c2c', ec='#f39c12'))
    plt.tight_layout(); return fig


def fig_radial_profile(centers, means, stem):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(centers, means, color='#E15759', lw=2.5)
    ax.fill_between(centers, means, alpha=0.2, color='#E15759')
    ax.set_xlim(0,1); ax.set_ylim(bottom=0)
    ax.set_xlabel('Normalised distance  (0 = nucleus boundary,  1 = cell edge)', fontsize=11)
    ax.set_ylabel('Mean pixel intensity (background subtracted)', fontsize=11)
    ax.set_title(f'{stem} — Radial fluorescence density profile\n'
                 f'Nucleus interior excluded', fontsize=11)
    ax.set_xticks([0,0.25,0.5,0.75,1.0])
    ax.set_xticklabels(['Nucleus\nboundary','25%','50%','75%','Cell\nedge'])
    for x in [0.25,0.5,0.75]:
        ax.axvline(x, color='grey', lw=0.8, linestyle=':', alpha=0.5)
    plt.tight_layout(); return fig


def fig_zones(ifd_z, total_ifd, stem, n_sat=0, exclude_sat=False):
    pcts = ifd_z/total_ifd*100 if total_ifd>0 else np.zeros(N_BINS)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.bar(range(N_BINS), pcts, color='#4E79A7', edgecolor='black', linewidth=0.7, width=0.6)
    ax.set_xticks(range(N_BINS)); ax.set_xticklabels(ZONE_LABELS, fontsize=12)
    ax.set_xlabel('Distance from nucleus boundary (% of cytoplasm width)', fontsize=11)
    ax.set_ylabel('% of total IFD in zone', fontsize=11)
    sat_note = f'  |  {n_sat} sat px excluded' if exclude_sat else \
               (f'  |  {n_sat} sat px included (may overestimate)' if n_sat>0 else '')
    ax.set_title(
        f'{stem} — Fluorescence density by zone\n'
        f'Total IFD = {total_ifd:,.0f}{sat_note}',
        fontsize=10)
    ymax = max(int(pcts.max()*1.3) if pcts.max()>0 else 10, 10)
    ax.set_ylim(0, ymax)
    for xi,pv in enumerate(pcts):
        if pv>0:
            ax.text(xi, pv+ymax*0.015, f'{pv:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout(); return fig


def fig_total_radial(centers, totals, stem):
    """Total IFD per fine-grained radial bin (same 60-bin resolution as Radial Profile)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, totals, color='#59A14F', lw=2.5)
    ax.fill_between(centers, totals, alpha=0.2, color='#59A14F')
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    ax.set_xlabel('Normalised distance  (0 = nucleus boundary,  1 = cell edge)', fontsize=11)
    ax.set_ylabel('Total IFD per bin (sum of raw pixel values)', fontsize=11)
    ax.set_title(f'{stem} — Total radial fluorescence profile\n'
                 f'Nucleus interior excluded', fontsize=11)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['Nucleus\nboundary', '25%', '50%', '75%', 'Cell\nedge'])
    for x in [0.25, 0.5, 0.75]:
        ax.axvline(x, color='grey', lw=0.8, linestyle=':', alpha=0.5)
    plt.tight_layout(); return fig


def fig_cumulative_radial(centers, totals, stem):
    """Cumulative % of total IFD from nucleus boundary outward (60-bin resolution)."""
    total = totals.sum()
    cumulative = np.cumsum(totals) / total * 100 if total > 0 else np.zeros(len(totals))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(centers, cumulative, color='#F28E2B', lw=2.5)
    ax.fill_between(centers, cumulative, alpha=0.2, color='#F28E2B')
    ax.set_xlim(0, 1); ax.set_ylim(0, 105)
    ax.set_xlabel('Normalised distance  (0 = nucleus boundary,  1 = cell edge)', fontsize=11)
    ax.set_ylabel('Cumulative % of total fluorescence', fontsize=11)
    ax.set_title(f'{stem} — Cumulative radial fluorescence profile\n'
                 f'Nucleus interior excluded', fontsize=11)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['Nucleus\nboundary', '25%', '50%', '75%', 'Cell\nedge'])
    for x in [0.25, 0.5, 0.75]:
        ax.axvline(x, color='grey', lw=0.8, linestyle=':', alpha=0.5)
    ax.axhline(50, color='grey', lw=0.8, linestyle='--', alpha=0.5)
    ax.text(0.01, 51, '50%', fontsize=8, color='grey', va='bottom')
    plt.tight_layout(); return fig


def fig_zone_sensitivity(img_sub, cell_mask, nuke_mask, stem,
                          thresholds=SENS_THRESHOLDS, exclude_sat=False,
                          correct_nc=True):
    """
    Zone bar charts at multiple min-intensity thresholds side by side.
    Lets the user see how sensitive the zone distribution is to that parameter.
    """
    colors = ['#E15759','#4E79A7','#59A14F','#F28E2B','#B07AA1']
    fig, axes = plt.subplots(1, len(thresholds),
                              figsize=(3.5*len(thresholds), 5), sharey=False)
    fig.suptitle(
        f'{stem} — Zone density sensitivity to min pixel intensity\n'
        f'(zones start at nucleus boundary, nucleus interior excluded)',
        fontsize=11, fontweight='bold')

    for ax, thresh, color in zip(axes, thresholds, colors):
        ifd_z, area_z, total_ifd, norm_dist, n_sat = compute_zone_density(
            img_sub, cell_mask, nuke_mask,
            min_intensity=thresh, exclude_sat=exclude_sat, correct_nc=correct_nc)
        pcts = ifd_z/total_ifd*100 if total_ifd>0 else np.zeros(N_BINS)
        ax.bar(range(N_BINS), pcts, color=color,
               edgecolor='black', linewidth=0.7, width=0.65)
        ax.set_xticks(range(N_BINS))
        ax.set_xticklabels(['0–25%','25–50%','50–75%','75–100%'],
                           fontsize=8, rotation=30, ha='right')
        ax.set_title(f'Min intensity = {thresh}\nTotal IFD = {total_ifd:,.0f}',
                     fontsize=9)
        ymax = max(int(pcts.max()*1.3) if pcts.max()>0 else 10, 10)
        ax.set_ylim(0, ymax)
        for xi,pv in enumerate(pcts):
            if pv>0:
                ax.text(xi, pv+ymax*0.02, f'{pv:.0f}%',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    axes[0].set_ylabel('% of total IFD', fontsize=10)
    plt.tight_layout(); return fig

# ── GUI widgets ────────────────────────────────────────────────────────────────

BOX_COLORS = ['yellow', 'orange', 'lime', 'cyan']

class MultiBoxCanvas(tk.Canvas):
    """Canvas that lets the user draw up to 4 background-selection boxes."""
    MAX = 4

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._start = None; self._drag_rect = None
        self._boxes = []          # list of (ix0,iy0,ix1,iy1) in image coords
        self._img_offset = (0,0); self._scale = 1.0
        self.bind('<ButtonPress-1>',   self._on_press)
        self.bind('<B1-Motion>',       self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)

    def _to_img(self, cx, cy):
        ox, oy = self._img_offset
        return int((cx-ox)/self._scale), int((cy-oy)/self._scale)

    def _to_canvas(self, ix, iy):
        ox, oy = self._img_offset
        return int(ix*self._scale + ox), int(iy*self._scale + oy)

    def _on_press(self, e):
        self._start = (e.x, e.y)
        if self._drag_rect: self.delete(self._drag_rect); self._drag_rect = None

    def _on_drag(self, e):
        if self._drag_rect: self.delete(self._drag_rect)
        x0, y0 = self._start
        color = BOX_COLORS[len(self._boxes) % len(BOX_COLORS)]
        self._drag_rect = self.create_rectangle(x0, y0, e.x, e.y,
                                                 outline=color, width=2, dash=(4,2))

    def _on_release(self, e):
        if self._start is None: return
        if self._drag_rect: self.delete(self._drag_rect); self._drag_rect = None
        if len(self._boxes) >= self.MAX: return
        x0c, y0c = self._start
        ix0, iy0 = self._to_img(min(x0c,e.x), min(y0c,e.y))
        ix1, iy1 = self._to_img(max(x0c,e.x), max(y0c,e.y))
        if ix1-ix0 < 2 or iy1-iy0 < 2: return
        self._boxes.append((ix0, iy0, ix1, iy1))
        self._draw_boxes()

    def _draw_boxes(self):
        # Redraw all permanent box rectangles + labels
        self.delete('box')
        for i, (ix0,iy0,ix1,iy1) in enumerate(self._boxes):
            color = BOX_COLORS[i % len(BOX_COLORS)]
            cx0,cy0 = self._to_canvas(ix0, iy0)
            cx1,cy1 = self._to_canvas(ix1, iy1)
            self.create_rectangle(cx0,cy0,cx1,cy1, outline=color, width=2, tags='box')
            self.create_text((cx0+cx1)//2, (cy0+cy1)//2, text=str(i+1),
                             fill=color, font=('Helvetica',10,'bold'), tags='box')

    def clear_boxes(self):
        self._boxes = []
        self.delete('box')

    def show_np(self, rgb):
        self.update_idletasks()
        cw = self.winfo_width() or 500; ch = self.winfo_height() or 500
        ih, iw = rgb.shape[:2]; scale = min(cw/iw, ch/ih)
        self._scale = scale; nw, nh = int(iw*scale), int(ih*scale)
        self._img_offset = ((cw-nw)//2, (ch-nh)//2)
        resized = cv2.resize(rgb, (nw,nh), interpolation=cv2.INTER_LINEAR)
        pil = Image.fromarray(resized); self._tk_img = ImageTk.PhotoImage(pil)
        self.delete('all')
        ox, oy = self._img_offset
        self.create_image(ox, oy, anchor='nw', image=self._tk_img)
        self._draw_boxes()


class FigCanvas(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg='#1a1a2e', **kw); self._tk_img=None

    def show_fig(self, fig):
        buf=io.BytesIO()
        fig.savefig(buf,format='png',dpi=100,bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0); pil=Image.open(buf)
        self.update_idletasks()
        cw=self.winfo_width() or 700; ch=self.winfo_height() or 500
        pil.thumbnail((cw,ch),Image.LANCZOS)
        self._tk_img=ImageTk.PhotoImage(pil)
        self.delete('all'); iw,ih=pil.size
        self.create_image(cw//2,ch//2,anchor='center',image=self._tk_img)
        buf.close()

# ── Main application ───────────────────────────────────────────────────────────

DARK=  '#2c3e50'; DARKER='#1a1a2e'; ACCENT='#3498db'
GREEN= '#27ae60'; GREY=  '#7f8c8d'; TEXT=  '#bdc3c7'
PURPLE='#8e44ad'
BTN=dict(relief='flat',cursor='hand2',font=('Helvetica',10,'bold'))

TABS = ['Heatmap','Pixel Intensity','Radial Profile','Zones','Zone Sensitivity','Comparison']


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Lysosome Fluorescence Analysis v2')
        self.configure(bg=DARK); self.geometry('1400x900'); self.minsize(1000,700)

        # Per-cell state: keyed by 1 and 2
        self._path     = {1:None, 2:None}
        self._arr      = {1:None, 2:None}
        self._vx       = {1:1.0,  2:1.0}
        self._vy       = {1:1.0,  2:1.0}
        self._lyso_raw = {1:None, 2:None}
        self._lyso_sub = {1:None, 2:None}
        self._bg_val   = {1:0.0,  2:0.0}

        self._build_ui()

    # ── Build UI ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self, bg=DARK); top.pack(side='top', fill='x', padx=8, pady=6)
        tk.Button(top, text='Open Cell 1…', command=lambda: self._open_lsm(1),
                  bg=ACCENT, fg='white', padx=12, pady=4, **BTN).pack(side='left', padx=4)
        tk.Button(top, text='Open Cell 2…', command=lambda: self._open_lsm(2),
                  bg=PURPLE, fg='white', padx=12, pady=4, **BTN).pack(side='left', padx=4)
        tk.Button(top, text='🔍  Preview Masks', command=self._preview_masks,
                  bg='#e67e22', fg='white', padx=12, pady=4, **BTN).pack(side='left', padx=4)
        tk.Button(top, text='▶  Run Analysis', command=self._run,
                  bg=GREEN, fg='white', padx=14, pady=4, **BTN).pack(side='left', padx=8)
        self._path_lbl = {1:None, 2:None}
        for i, color in [(1, ACCENT), (2, PURPLE)]:
            lbl = tk.Label(top, text=f'Cell {i}: —', bg=DARK, fg=TEXT,
                           font=('Helvetica', 9))
            lbl.pack(side='left', padx=6)
            self._path_lbl[i] = lbl

        paned = tk.PanedWindow(self, orient='horizontal', bg=DARK, sashwidth=5, sashrelief='flat')
        paned.pack(fill='both', expand=True, padx=6, pady=4)

        left = tk.Frame(paned, bg=DARK, width=370); paned.add(left, minsize=310)

        # Cell tabs (canvas + background boxes per cell)
        style = ttk.Style(); style.theme_use('clam')
        style.configure('TNotebook',       background=DARKER, borderwidth=0)
        style.configure('TNotebook.Tab',   background=DARK, foreground='white',
                        font=('Helvetica',9,'bold'), padding=[10,4])
        style.map('TNotebook.Tab', background=[('selected', ACCENT)])
        style.configure('TFrame', background=DARKER)

        cell_nb = ttk.Notebook(left); cell_nb.pack(fill='both', expand=True, padx=4, pady=4)
        self._img_cv  = {}   # cell_idx → MultiBoxCanvas
        self._bg_lbl_w = {}  # cell_idx → Label showing median value

        for i, color in [(1, ACCENT), (2, PURPLE)]:
            frame = ttk.Frame(cell_nb); cell_nb.add(frame, text=f'  Cell {i}  ')

            img_lf = tk.LabelFrame(frame,
                text=f' Cell {i} — draw up to 4 background boxes ',
                bg=DARK, fg='white', font=('Helvetica',9), bd=1, relief='groove')
            img_lf.pack(fill='both', expand=True, padx=4, pady=4)

            cv = MultiBoxCanvas(img_lf, bg='#111111', cursor='crosshair')
            cv.pack(fill='both', expand=True)
            self._img_cv[i] = cv

            bg_lf = tk.LabelFrame(frame, text=' Background ', bg=DARK, fg='white',
                                   font=('Helvetica',9), bd=1, relief='groove')
            bg_lf.pack(fill='x', padx=4, pady=(0,4))

            btn_row = tk.Frame(bg_lf, bg=DARK); btn_row.pack(fill='x', padx=6, pady=4)
            tk.Button(btn_row, text='Set Background',
                      command=lambda i=i: self._set_bg(i),
                      bg=GREY, fg='white', padx=8, pady=3,
                      relief='flat', cursor='hand2',
                      font=('Helvetica',9)).pack(side='left', padx=(0,4))
            tk.Button(btn_row, text='Clear Boxes',
                      command=lambda i=i: self._clear_boxes(i),
                      bg='#c0392b', fg='white', padx=8, pady=3,
                      relief='flat', cursor='hand2',
                      font=('Helvetica',9)).pack(side='left')

            bg_row = tk.Frame(bg_lf, bg=DARK); bg_row.pack(fill='x', padx=6, pady=(0,4))
            tk.Label(bg_row, text='Median background:', bg=DARK, fg=TEXT,
                     font=('Helvetica',9)).pack(side='left')
            lbl = tk.Label(bg_row, text='—', bg=DARK, fg='#f39c12',
                           font=('Helvetica',9,'bold'))
            lbl.pack(side='left', padx=4)
            self._bg_lbl_w[i] = lbl

        # Shared analysis settings
        ctrl = tk.Frame(left, bg=DARK); ctrl.pack(fill='x', padx=4, pady=2)
        det_lf = tk.LabelFrame(ctrl, text=' Analysis settings ', bg=DARK, fg='white',
                                font=('Helvetica',9), bd=1, relief='groove')
        det_lf.pack(fill='x', pady=3)

        def _row(parent, label, var, lo, hi, w=5):
            r = tk.Frame(parent, bg=DARK); r.pack(fill='x', padx=6, pady=2)
            tk.Label(r, text=label, bg=DARK, fg=TEXT, font=('Helvetica',9),
                     width=22, anchor='w').pack(side='left')
            tk.Spinbox(r, from_=lo, to=hi, textvariable=var,
                       width=w, font=('Helvetica',9)).pack(side='left', padx=4)

        self._min_int   = tk.IntVar(value=20)
        self._blur_sigma= tk.IntVar(value=8)
        _row(det_lf, 'Min pixel intensity:',   self._min_int,    1, 254)
        _row(det_lf, 'Lyso mask blur sigma:',  self._blur_sigma,  1, 200)

        # ── Mask method selector ──────────────────────────────────────────────
        mask_lf = tk.LabelFrame(det_lf, text=' Cell mask method ',
                                bg=DARK, fg='white',
                                font=('Helvetica', 9), bd=1, relief='groove')
        mask_lf.pack(fill='x', padx=6, pady=(2, 4))
        self._mask_method = tk.StringVar(value='any_signal')
        for val, lbl in [
            ('any_signal', 'CLAHE + Otsu  (raw, high contrast, recommended)'),
            ('watershed',  'Watershed  (nucleus-seeded)'),
            ('spot_hull',  'Spot Convex Hull  (+ dilation)'),
            ('nuc_expand', f'Nucleus Expand  (DAPI only, NC={NUC_CELL_RATIO})'),
        ]:
            tk.Radiobutton(mask_lf, text=lbl,
                           variable=self._mask_method, value=val,
                           bg=DARK, fg=TEXT, selectcolor='#1a3a5c',
                           activebackground=DARK,
                           font=('Helvetica', 9)).pack(anchor='w', padx=8, pady=1)

        self._excl_sat = tk.BooleanVar(value=False)
        tk.Checkbutton(det_lf, text='Exclude saturated pixels (=255) from IFD',
                       variable=self._excl_sat, bg=DARK, fg=TEXT,
                       selectcolor='#1a3a5c', activebackground=DARK,
                       font=('Helvetica',9)).pack(anchor='w', padx=6, pady=(2,4))

        self._status = tk.StringVar(value='Load LSM file(s) to begin.')
        tk.Label(ctrl, textvariable=self._status, bg=DARK, fg=TEXT,
                 font=('Helvetica',9), wraplength=300,
                 justify='left').pack(anchor='w', padx=4)

        # Right notebook — three top-level tabs: Cell 1 | Cell 2 | Comparison
        right = tk.Frame(paned, bg=DARKER); paned.add(right, minsize=580)
        self._nb = ttk.Notebook(right)
        self._nb.pack(fill='both', expand=True, padx=4, pady=4)
        self._tab_cv = {}   # keyed by (cell_idx, tab_name) or 'Comparison'

        CELL_TABS = ['Mask Preview', 'Heatmap', 'Pixel Intensity',
                     'Radial Profile', 'Total Radial', 'Cumulative Radial',
                     'Zones', 'Zone Sensitivity']

        for i, label in [(1, 'Cell 1'), (2, 'Cell 2')]:
            outer = ttk.Frame(self._nb)
            self._nb.add(outer, text=f'  {label}  ')
            inner_nb = ttk.Notebook(outer)
            inner_nb.pack(fill='both', expand=True)
            for name in CELL_TABS:
                f = ttk.Frame(inner_nb); inner_nb.add(f, text=name)
                cv = FigCanvas(f); cv.pack(fill='both', expand=True)
                self._tab_cv[(i, name)] = cv

        comp_frame = ttk.Frame(self._nb)
        self._nb.add(comp_frame, text='  Comparison  ')
        cv = FigCanvas(comp_frame); cv.pack(fill='both', expand=True)
        self._tab_cv['Comparison'] = cv

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _open_lsm(self, cell_idx):
        path = filedialog.askopenfilename(title=f'Open Cell {cell_idx} LSM',
                filetypes=[('LSM files','*.lsm'),('All files','*.*')])
        if not path: return
        self._status.set(f'Loading Cell {cell_idx}…'); self.update()
        try:
            p = Path(path)
            self._path[cell_idx]     = p
            self._arr[cell_idx]      = load_lsm(path)
            self._vx[cell_idx], self._vy[cell_idx] = get_voxel_um(path)
            self._lyso_raw[cell_idx] = max_project(self._arr[cell_idx], LYSO_CH)
            self._lyso_sub[cell_idx] = self._lyso_raw[cell_idx].copy()
            self._bg_val[cell_idx]   = 0.0
            self._bg_lbl_w[cell_idx].config(text='—')
            self._path_lbl[cell_idx].config(text=f'Cell {cell_idx}: {p.name}')
            self._refresh_preview(cell_idx)
            z,c,h,w = self._arr[cell_idx].shape
            self._status.set(
                f'Cell {cell_idx} loaded: {w}×{h} px  Z={z}\n'
                f'Pixel: {self._vx[cell_idx]:.4f} µm\n'
                f'Draw background boxes, then run.')
        except Exception as ex:
            messagebox.showerror('Load error', str(ex))

    def _refresh_preview(self, cell_idx):
        raw = self._lyso_raw[cell_idx]
        if raw is None: return
        colored = cv2.applyColorMap(raw, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        rgb[raw==255] = [255,255,255]
        self._img_cv[cell_idx].show_np(rgb)

    def _set_bg(self, cell_idx):
        raw = self._lyso_raw[cell_idx]
        if raw is None:
            messagebox.showwarning('No image', f'Load Cell {cell_idx} first.'); return
        boxes = self._img_cv[cell_idx]._boxes
        if not boxes:
            messagebox.showwarning('No boxes', 'Draw at least one background box.'); return
        h, w = raw.shape
        all_px = []
        for (x0,y0,x1,y1) in boxes:
            x0,x1 = sorted([max(0,min(x0,w-1)), max(0,min(x1,w-1))])
            y0,y1 = sorted([max(0,min(y0,h-1)), max(0,min(y1,h-1))])
            if x1-x0 >= 2 and y1-y0 >= 2:
                all_px.extend(raw[y0:y1+1, x0:x1+1].ravel().tolist())
        if not all_px:
            messagebox.showwarning('Too small', 'Boxes are too small.'); return
        bg = float(np.median(all_px))
        self._bg_val[cell_idx]   = bg
        self._lyso_sub[cell_idx] = subtract_background(raw, bg)
        self._bg_lbl_w[cell_idx].config(text=f'{bg:.1f}')
        self._status.set(
            f'Cell {cell_idx} background = {bg:.1f}\n'
            f'({len(boxes)} box{"es" if len(boxes)>1 else ""}, '
            f'{len(all_px):,} pixels)\nReady to run.')

    def _clear_boxes(self, cell_idx):
        self._img_cv[cell_idx].clear_boxes()
        self._bg_val[cell_idx]   = 0.0
        self._lyso_sub[cell_idx] = (self._lyso_raw[cell_idx].copy()
                                    if self._lyso_raw[cell_idx] is not None else None)
        self._bg_lbl_w[cell_idx].config(text='—')

    def _analyze_cell(self, cell_idx, min_int, excl, blur_sigma, mask_method='auto'):
        """Run full analysis for one cell. Returns result dict."""
        arr      = self._arr[cell_idx]
        lyso_sub = self._lyso_sub[cell_idx]
        lyso_raw = self._lyso_raw[cell_idx]
        bg_val   = self._bg_val[cell_idx]
        stem     = self._path[cell_idx].stem
        out_dir  = self._path[cell_idx].parent

        nuke_mask, _ = get_nucleus_mask(arr)
        nuke_mp      = max_project(arr, NUKE_CH)
        save_nucleus_mask_preview(nuke_mp, nuke_mask,
                                  out_dir / f'{stem}_nucleus_mask_v2.png')

        # ── Build cell mask according to selected method ──────────────────────
        # any_signal: thr=0 → any non-zero pixel is cell; large closing fills gaps
        # watershed:  nucleus-seeded watershed; no lyso threshold needed
        # spot_hull:  convex hull of detected lyso spots; point-cloud approach
        # nuc_expand: DAPI-only distance expansion; fully distribution-independent
        if mask_method == 'watershed':
            cell_mask  = _mask_by_watershed(lyso_sub, nuke_mask)
            if cell_mask is None:
                cell_mask = make_cell_mask(lyso_sub, blur_sigma=8)
            correct_nc = False
        elif mask_method == 'spot_hull':
            cell_mask  = _mask_by_spot_hull(lyso_sub, nuke_mask)
            if cell_mask is None:
                cell_mask = make_cell_mask(lyso_sub, blur_sigma=8)
            correct_nc = False
        elif mask_method == 'nuc_expand':
            cell_mask  = _mask_by_nuc_expand(nuke_mask, NUC_CELL_RATIO)
            if cell_mask is None:
                cell_mask = make_cell_mask(lyso_sub, blur_sigma=8)
            correct_nc = False
        else:  # 'any_signal' (default) — CLAHE + Otsu on raw image, nucleus-seeded
            cell_mask  = _mask_by_any_signal(lyso_raw, nuke_mask)
            correct_nc = False

        # Mask is built from lyso_sub (bright/background-subtracted) above.
        # Density measurement uses lyso_raw (original unchanged LSM).
        # cell_mask has values 0 or 255 — use it as a boolean, NOT a multiplier.
        # Multiplying by 255 would saturate every non-zero pixel to 255.
        lyso_masked = lyso_raw.copy()
        lyso_masked[cell_mask == 0] = 0

        ifd_z, area_z, total_ifd, norm_dist, n_sat = compute_zone_density(
            lyso_masked, cell_mask, nuke_mask,
            min_intensity=min_int, exclude_sat=excl, correct_nc=correct_nc)

        rad_c, rad_m, rad_t = compute_radial_profile(
            lyso_masked, cell_mask, nuke_mask,
            norm_dist, min_intensity=min_int, exclude_sat=excl)

        return dict(stem=stem, out_dir=out_dir,
                    lyso_raw=lyso_raw, lyso_masked=lyso_masked,
                    bg_val=bg_val, vx=self._vx[cell_idx], vy=self._vy[cell_idx],
                    cell_mask=cell_mask, nuke_mask=nuke_mask,
                    ifd_z=ifd_z, area_z=area_z, total_ifd=total_ifd,
                    norm_dist=norm_dist, n_sat=n_sat,
                    rad_c=rad_c, rad_m=rad_m, rad_t=rad_t,
                    min_int=min_int, excl=excl, correct_nc=correct_nc)

    def _preview_masks(self):
        """Generate side-by-side mask comparison for the loaded cell(s)."""
        for cell_idx in [1, 2]:
            if self._arr[cell_idx] is None:
                continue
            self._status.set(f'Previewing masks for Cell {cell_idx}…'); self.update()
            try:
                arr      = self._arr[cell_idx]
                lyso_raw = self._lyso_raw[cell_idx]
                lyso_sub = self._lyso_sub[cell_idx]
                nuke, _  = get_nucleus_mask(arr)
                stem     = self._path[cell_idx].stem
                fig = fig_mask_comparison(lyso_raw, lyso_sub, nuke, stem)
                self._tab_cv[(cell_idx, 'Mask Preview')].show_fig(fig)
                # Also save a PNG alongside the LSM
                out = self._path[cell_idx].parent / f'{stem}_mask_comparison_v2.png'
                fig.savefig(out, dpi=130, bbox_inches='tight',
                            facecolor=fig.get_facecolor())
                plt.close(fig)
            except Exception:
                import traceback
                from tkinter import messagebox as mb
                mb.showerror('Mask preview error', traceback.format_exc())
        self._status.set('Mask preview done — pick a method, then Run Analysis.')

    def _run(self):
        if self._arr[1] is None:
            messagebox.showwarning('No data','Load at least Cell 1 first.'); return
        self._status.set('Running…'); self.update()
        try:
            min_int = self._min_int.get()
            excl    = self._excl_sat.get()
            blur    = self._blur_sigma.get()
            method  = self._mask_method.get()

            r1 = self._analyze_cell(1, min_int, excl, blur, method)
            r2 = self._analyze_cell(2, min_int, excl, blur, method) \
                 if self._arr[2] is not None else None

            def _cell_figs(r):
                return {
                    'Mask Preview': fig_mask_comparison(
                        r['lyso_raw'], r['lyso_masked'], r['nuke_mask'], r['stem']),
                    'Heatmap': fig_heatmap(
                        r['lyso_raw'], r['lyso_masked'], r['bg_val'],
                        r['stem'], r['vx'], r['vy'],
                        cell_mask=r['cell_mask'], nuke_mask=r['nuke_mask']),
                    'Pixel Intensity': fig_pixel_histogram(
                        r['lyso_masked'], r['cell_mask'], r['nuke_mask'],
                        r['min_int'], r['stem']),
                    'Radial Profile': fig_radial_profile(
                        r['rad_c'], r['rad_m'], r['stem']),
                    'Total Radial': fig_total_radial(
                        r['rad_c'], r['rad_t'], r['stem']),
                    'Cumulative Radial': fig_cumulative_radial(
                        r['rad_c'], r['rad_t'], r['stem']),
                    'Zones': fig_zones(
                        r['ifd_z'], r['total_ifd'], r['stem'],
                        n_sat=r['n_sat'], exclude_sat=r['excl']),
                    'Zone Sensitivity': fig_zone_sensitivity(
                        r['lyso_masked'], r['cell_mask'], r['nuke_mask'], r['stem'],
                        thresholds=SENS_THRESHOLDS, exclude_sat=r['excl'],
                        correct_nc=r['correct_nc']),
                }

            # Cell 1 — always
            for name, fig in _cell_figs(r1).items():
                self._tab_cv[(1, name)].show_fig(fig)
                safe = name.lower().replace(' ','_')
                fig.savefig(r1['out_dir'] / f"{r1['stem']}_{safe}_v2.png",
                            dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
                plt.close(fig)

            # Cell 2 — if loaded
            if r2:
                for name, fig in _cell_figs(r2).items():
                    self._tab_cv[(2, name)].show_fig(fig)
                    safe = name.lower().replace(' ','_')
                    fig.savefig(r2['out_dir'] / f"{r2['stem']}_{safe}_v2.png",
                                dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
                    plt.close(fig)

            # Comparison — overlaid radial profile + grouped zones
            if r2:
                fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                fig.suptitle('Cell Comparison Summary', fontsize=13, fontweight='bold')
                ax = axes[0]
                ax.plot(r1['rad_c'], r1['rad_m'], color='#4E79A7', lw=2.5, label=r1['stem'])
                ax.fill_between(r1['rad_c'], r1['rad_m'], alpha=0.15, color='#4E79A7')
                ax.plot(r2['rad_c'], r2['rad_m'], color='#E15759', lw=2.5, label=r2['stem'])
                ax.fill_between(r2['rad_c'], r2['rad_m'], alpha=0.15, color='#E15759')
                ax.set_xlim(0,1); ax.set_ylim(bottom=0)
                ax.set_xlabel('Distance (nucleus→cell edge)', fontsize=10)
                ax.set_ylabel('Mean intensity', fontsize=10)
                ax.set_title('Radial Profile', fontsize=11)
                ax.set_xticks([0,.25,.5,.75,1])
                ax.set_xticklabels(['Nuc','25%','50%','75%','Edge'], fontsize=9)
                ax.legend(fontsize=9)
                ax = axes[1]
                pct1 = r1['ifd_z']/r1['total_ifd']*100 if r1['total_ifd']>0 else np.zeros(N_BINS)
                pct2 = r2['ifd_z']/r2['total_ifd']*100 if r2['total_ifd']>0 else np.zeros(N_BINS)
                xs = np.arange(N_BINS); bw = 0.35
                ax.bar(xs-bw/2, pct1, bw, label=r1['stem'], color='#4E79A7', edgecolor='black', lw=0.7)
                ax.bar(xs+bw/2, pct2, bw, label=r2['stem'], color='#E15759', edgecolor='black', lw=0.7)
                ax.set_xticks(xs); ax.set_xticklabels(ZONE_LABELS, fontsize=10)
                ax.set_ylabel('% of total IFD', fontsize=10)
                ax.set_title('Zone Distribution', fontsize=11)
                ax.legend(fontsize=9)
                plt.tight_layout()
            else:
                fig, ax = plt.subplots(figsize=(8,4))
                ax.text(0.5, 0.5, 'Load Cell 2 to see comparison',
                        ha='center', va='center', fontsize=14, color='grey')
                ax.axis('off')
            self._tab_cv['Comparison'].show_fig(fig)
            fig.savefig(r1['out_dir'] / f"{r1['stem']}_comparison_v2.png",
                        dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)

            # CSV for Cell 1
            zone_csv = r1['out_dir'] / f"{r1['stem']}_zones_summary_v2.csv"
            with open(zone_csv,'w',newline='',encoding='utf-8') as f:
                f.write('zone,pct,count\n')
                for i,lbl in enumerate(['0-25%','25-50%','50-75%','75-100%']):
                    pct = r1['ifd_z'][i]/r1['total_ifd']*100 if r1['total_ifd']>0 else 0.0
                    f.write(f'{lbl},{pct:.4f},{int(r1["area_z"][i])}\n')

            msg = f"Done!  Cell 1 IFD: {r1['total_ifd']:,.0f}"
            if r2: msg += f"\nCell 2 IFD: {r2['total_ifd']:,.0f}"
            msg += f"\nSaved to {r1['out_dir'].name}/"
            self._status.set(msg)

        except Exception:
            import traceback
            messagebox.showerror('Error', traceback.format_exc())


if __name__ == '__main__':
    App().mainloop()
