#!/usr/bin/env python3
"""
Lysosome Fluorescence Batch Analyser  v1
-----------------------------------------
Analyses every cell in a multi-cell LSM field of view.
Results are displayed in-app AND saved to <lsm_stem>_batch/.
"""

import sys, struct, csv, traceback, threading, io
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# ── Constants ─────────────────────────────────────────────────────────────────
LYSO_CH        = 2
NUKE_CH        = 0
N_BINS         = 4
ZONE_LABELS    = ['0–25%', '25–50%', '50–75%', '75–100%']
MIN_NUCLEUS_PX = 5_000
CROP_PAD_PX    = 150

DARK   = '#2c3e50'
DARKER = '#1a1a2e'
TEXT   = '#ecf0f1'

CELL_TABS = ['Mask Preview', 'Radial Profile', 'Total Radial',
             'Cumulative Radial', 'Zones']

# ── FigCanvas — loads a saved PNG from disk ───────────────────────────────────

class FigCanvas(tk.Canvas):
    """Displays a PNG file; re-scales whenever the widget is resized."""
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=DARKER, **kw)
        self._pil    = None
        self._tk_img = None
        self.bind('<Configure>', lambda e: self._redraw())

    def load_png(self, path):
        self._pil = Image.open(str(path)).convert('RGB')
        self._redraw()

    def _redraw(self):
        if self._pil is None:
            return
        self.update_idletasks()
        cw = max(self.winfo_width(),  800)
        ch = max(self.winfo_height(), 550)
        pil = self._pil.copy()
        pil.thumbnail((cw, ch), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.delete('all')
        self.create_image(cw // 2, ch // 2, anchor='center', image=self._tk_img)

# ── LSM loading ───────────────────────────────────────────────────────────────

def load_lsm(path):
    ok, frames = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)
    if not ok or not frames:
        raise IOError(f"Cannot read {path}")
    real = [f for f in frames if f.shape[0] > 128]
    arr  = np.stack([f if f.ndim == 3 else np.stack([f, f, f], axis=-1)
                     for f in real], axis=0)
    return arr.transpose(0, 3, 1, 2)   # (Z, C_BGR, Y, X)

def max_project(arr, ch):
    return arr[:, ch].max(axis=0)

# ── Nucleus detection ─────────────────────────────────────────────────────────

def detect_all_nuclei(nuke_mp, min_px):
    blurred = cv2.GaussianBlur(nuke_mp, (61, 61), 15)
    clipped = np.clip(blurred.astype(np.int32) - 50, 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(clipped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kern)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8)
    nuclei = []
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_px:
            continue
        mask = (labels == lbl).astype(np.uint8) * 255
        cy = float(centroids[lbl][1]); cx = float(centroids[lbl][0])
        y0 = int(stats[lbl, cv2.CC_STAT_TOP]);  h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        x0 = int(stats[lbl, cv2.CC_STAT_LEFT]); w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        nuclei.append({'label': lbl, 'mask': mask,
                       'bbox': (y0, x0, y0+h, x0+w),
                       'centroid': (cy, cx), 'area': area})
    nuclei.sort(key=lambda d: (d['centroid'][0], d['centroid'][1]))
    return nuclei

# ── Segmentation helpers ──────────────────────────────────────────────────────

def keep_largest(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    best = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    return (labels == best).astype(np.uint8) * 255

def mask_by_any_signal(lyso_raw, nuke_mask):
    nuke_bin = (nuke_mask > 0).astype(np.uint8)
    blur = cv2.GaussianBlur(lyso_raw.astype(np.float32), (21, 21), 7)
    _, binary = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    binary = np.maximum(binary, nuke_bin * 255)
    _, labels_raw = cv2.connectedComponents(binary)
    nuke_labels = labels_raw[nuke_bin > 0]
    unique = np.unique(nuke_labels[nuke_labels > 0])
    if len(unique) == 0:
        cell_binary = binary
    else:
        cell_binary = np.zeros_like(binary)
        for lbl in unique:
            cell_binary = np.maximum(cell_binary,
                                     (labels_raw == lbl).astype(np.uint8) * 255)
    kern_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    cell_mask = cv2.morphologyEx(cell_binary, cv2.MORPH_CLOSE, kern_sm)
    kern_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
    cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_CLOSE, kern_lg)
    return cell_mask

# ── Analysis ──────────────────────────────────────────────────────────────────

def compute_zone_density(img, cell_mask, nuke_mask, min_intensity=0, n_bins=N_BINS):
    nuke_inv  = (nuke_mask == 0).astype(np.uint8)
    dist      = cv2.distanceTransform(nuke_inv, cv2.DIST_L2, 5)
    cytoplasm = (cell_mask > 0) & (nuke_mask == 0)
    max_dist  = float(dist[cytoplasm].max()) if cytoplasm.sum() > 0 else 1.0
    norm_dist = dist / max_dist if max_dist > 0 else np.zeros_like(dist)
    work      = img.copy().astype(np.float32)
    n_sat     = int(((img == 255) & cytoplasm).sum())
    valid     = cytoplasm & (img >= min_intensity) & (norm_dist <= 1.0)
    ifd_z = np.zeros(n_bins); area_z = np.zeros(n_bins)
    for z in range(n_bins):
        lo, hi    = z / n_bins, (z+1) / n_bins
        ring      = valid & (norm_dist >= lo) & (norm_dist < hi)
        ifd_z[z]  = float(work[ring].sum())
        area_z[z] = float(ring.sum())
    return ifd_z, area_z, float(ifd_z.sum()), norm_dist, n_sat

def compute_radial_profile(img, cell_mask, nuke_mask, norm_dist,
                            min_intensity=0, n_pts=60):
    cytoplasm = (cell_mask > 0) & (nuke_mask == 0)
    work  = img.copy().astype(np.float32)
    valid = cytoplasm & (img >= min_intensity) & (norm_dist <= 1.0)
    bins  = np.linspace(0, 1.0, n_pts+1)
    centers = (bins[:-1] + bins[1:]) / 2
    means = np.zeros(n_pts); totals = np.zeros(n_pts)
    for i in range(n_pts):
        ring = valid & (norm_dist >= bins[i]) & (norm_dist < bins[i+1])
        n = ring.sum()
        if n > 0:
            means[i]  = float(work[ring].mean())
            totals[i] = float(work[ring].sum())
    return centers, means, totals

# ── Figure generators (dark background, color) ───────────────────────────────

BG   = '#1a1a2e'
GRID = '#2c2c4a'

def _dark_ax(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#445566')

def fig_mask_preview(lyso_raw, lyso_masked, nuke_mask, cell_mask, stem):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor('black')

    # Left: raw channel in hot colormap
    axes[0].imshow(lyso_raw, cmap='hot', vmin=0, vmax=255)
    axes[0].set_title('Raw lysosome channel', color=TEXT)
    # Right: masked channel
    axes[1].imshow(lyso_masked, cmap='hot', vmin=0, vmax=255)
    axes[1].set_title('Masked (cell only)', color=TEXT)

    nuke_cnts, _ = cv2.findContours(nuke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for ax in axes:
        for c in cell_cnts:
            pts = c.squeeze()
            if pts.ndim == 2:
                ax.plot(pts[:, 0], pts[:, 1], color='#f1c40f', lw=1.5, alpha=0.9)
        for c in nuke_cnts:
            pts = c.squeeze()
            if pts.ndim == 2:
                ax.plot(pts[:, 0], pts[:, 1], color='#3498db', lw=1.5, alpha=0.9)
        ax.axis('off')
    fig.suptitle(f'{stem} — Mask preview  (blue=nucleus  yellow=cell)', color=TEXT, fontsize=11)
    plt.tight_layout(); return fig

def fig_radial_profile(centers, means, stem):
    fig, ax = plt.subplots(figsize=(8, 5))
    _dark_ax(fig, ax)
    ax.plot(centers, means, color='#E15759', lw=2.5)
    ax.fill_between(centers, means, alpha=0.25, color='#E15759')
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    ax.set_xlabel('Normalised distance  (0 = nucleus boundary,  1 = cell edge)', fontsize=11, color=TEXT)
    ax.set_ylabel('Mean pixel intensity (raw LSM)', fontsize=11, color=TEXT)
    ax.set_title(f'{stem} — Radial fluorescence density profile', fontsize=11, color=TEXT)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['Nucleus\nboundary', '25%', '50%', '75%', 'Cell\nedge'], color=TEXT)
    for x in [0.25, 0.5, 0.75]:
        ax.axvline(x, color=GRID, lw=0.8, linestyle=':')
    plt.tight_layout(); return fig

def fig_total_radial(centers, totals, stem):
    fig, ax = plt.subplots(figsize=(8, 5))
    _dark_ax(fig, ax)
    ax.plot(centers, totals, color='#59A14F', lw=2.5)
    ax.fill_between(centers, totals, alpha=0.25, color='#59A14F')
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    ax.set_xlabel('Normalised distance  (0 = nucleus boundary,  1 = cell edge)', fontsize=11, color=TEXT)
    ax.set_ylabel('Total IFD per bin (sum of raw pixel values)', fontsize=11, color=TEXT)
    ax.set_title(f'{stem} — Total radial fluorescence profile', fontsize=11, color=TEXT)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['Nucleus\nboundary', '25%', '50%', '75%', 'Cell\nedge'], color=TEXT)
    for x in [0.25, 0.5, 0.75]:
        ax.axvline(x, color=GRID, lw=0.8, linestyle=':')
    plt.tight_layout(); return fig

def fig_cumulative_radial(centers, totals, stem):
    total = totals.sum()
    cumulative = np.cumsum(totals) / total * 100 if total > 0 else np.zeros(len(totals))
    fig, ax = plt.subplots(figsize=(8, 5))
    _dark_ax(fig, ax)
    ax.plot(centers, cumulative, color='#F28E2B', lw=2.5)
    ax.fill_between(centers, cumulative, alpha=0.25, color='#F28E2B')
    ax.set_xlim(0, 1); ax.set_ylim(0, 105)
    ax.set_xlabel('Normalised distance  (0 = nucleus boundary,  1 = cell edge)', fontsize=11, color=TEXT)
    ax.set_ylabel('Cumulative % of total fluorescence', fontsize=11, color=TEXT)
    ax.set_title(f'{stem} — Cumulative radial fluorescence profile', fontsize=11, color=TEXT)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['Nucleus\nboundary', '25%', '50%', '75%', 'Cell\nedge'], color=TEXT)
    for x in [0.25, 0.5, 0.75]:
        ax.axvline(x, color=GRID, lw=0.8, linestyle=':')
    ax.axhline(50, color='#445566', lw=0.8, linestyle='--')
    ax.text(0.01, 51, '50%', fontsize=8, color='#95a5a6', va='bottom')
    plt.tight_layout(); return fig

def fig_zones(ifd_z, total_ifd, stem, n_sat=0):
    pcts = ifd_z / total_ifd * 100 if total_ifd > 0 else np.zeros(N_BINS)
    fig, ax = plt.subplots(figsize=(7, 5))
    _dark_ax(fig, ax)
    ax.bar(range(N_BINS), pcts, color='#4E79A7', edgecolor='#2c3e50', linewidth=0.7, width=0.6)
    ax.set_xticks(range(N_BINS)); ax.set_xticklabels(ZONE_LABELS, fontsize=12, color=TEXT)
    ax.set_xlabel('Distance from nucleus boundary (% of cytoplasm width)', fontsize=11, color=TEXT)
    ax.set_ylabel('% of total IFD in zone', fontsize=11, color=TEXT)
    sat_note = f'  |  {n_sat} sat px' if n_sat > 0 else ''
    ax.set_title(f'{stem} — Fluorescence density by zone\n'
                 f'Total IFD = {total_ifd:,.0f}{sat_note}', fontsize=10, color=TEXT)
    ymax = max(int(pcts.max() * 1.3) if pcts.max() > 0 else 10, 10)
    ax.set_ylim(0, ymax)
    for xi, pv in enumerate(pcts):
        if pv > 0:
            ax.text(xi, pv + ymax * 0.015, f'{pv:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color=TEXT)
    plt.tight_layout(); return fig

# ── Per-cell analysis ─────────────────────────────────────────────────────────

def analyse_cell(lyso_crop, nuke_crop, stem, out_dir, min_intensity=0):
    lyso_raw  = lyso_crop.max(axis=0)
    nuke_full = nuke_crop.max(axis=0)
    bg_val = float(np.percentile(lyso_raw.flatten(), 20))

    blurred = cv2.GaussianBlur(nuke_full, (61, 61), 15)
    clipped = np.clip(blurred.astype(np.int32) - 50, 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(clipped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    nuke_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kern)
    nuke_mask = keep_largest(nuke_mask)

    cell_mask   = mask_by_any_signal(lyso_raw, nuke_mask)
    lyso_masked = lyso_raw.copy()
    lyso_masked[cell_mask == 0] = 0

    ifd_z, area_z, total_ifd, norm_dist, n_sat = compute_zone_density(
        lyso_masked, cell_mask, nuke_mask, min_intensity=min_intensity)
    rad_c, rad_m, rad_t = compute_radial_profile(
        lyso_masked, cell_mask, nuke_mask, norm_dist, min_intensity=min_intensity)

    png_paths = {}
    fig_data = {
        'Mask Preview':      fig_mask_preview(lyso_raw, lyso_masked, nuke_mask, cell_mask, stem),
        'Radial Profile':    fig_radial_profile(rad_c, rad_m, stem),
        'Total Radial':      fig_total_radial(rad_c, rad_t, stem),
        'Cumulative Radial': fig_cumulative_radial(rad_c, rad_t, stem),
        'Zones':             fig_zones(ifd_z, total_ifd, stem, n_sat),
    }
    for name, fig in fig_data.items():
        safe = name.lower().replace(' ', '_')
        p = out_dir / f'{stem}_{safe}.png'
        fig.savefig(p, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        png_paths[name] = p

    return dict(stem=stem, png_paths=png_paths,
                ifd_z=ifd_z, area_z=area_z, total_ifd=total_ifd,
                n_sat=n_sat, bg_val=bg_val,
                nuke_area=int((nuke_mask > 0).sum()),
                cell_area=int((cell_mask > 0).sum()))

# ── Main application ──────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Lysosome Batch Analyser v1')
        self.geometry('1150x800')
        self.configure(bg=DARK)
        self._tab_cv    = {}
        self._path_full = None
        self._build_ui()

    def _build_ui(self):
        # ── Left panel ────────────────────────────────────────────────────────
        left = tk.Frame(self, bg=DARK, width=270)
        left.pack(side='left', fill='y', padx=6, pady=6)
        left.pack_propagate(False)

        tk.Label(left, text='Lysosome Batch Analyser', bg=DARK, fg=TEXT,
                 font=('Helvetica', 12, 'bold')).pack(pady=(8, 4))

        # File
        file_lf = tk.LabelFrame(left, text=' LSM file ', bg=DARK, fg='white',
                                 font=('Helvetica', 9), bd=1, relief='groove')
        file_lf.pack(fill='x', padx=4, pady=4)
        self._path_var = tk.StringVar(value='(no file selected)')
        tk.Label(file_lf, textvariable=self._path_var, bg=DARK, fg='#3498db',
                 font=('Helvetica', 8), wraplength=230, anchor='w',
                 justify='left').pack(fill='x', padx=4, pady=2)
        tk.Button(file_lf, text='Browse…', command=self._browse,
                  bg='#3498db', fg='white', font=('Helvetica', 9),
                  relief='flat', cursor='hand2').pack(pady=4)

        # Settings
        cfg_lf = tk.LabelFrame(left, text=' Settings ', bg=DARK, fg='white',
                                font=('Helvetica', 9), bd=1, relief='groove')
        cfg_lf.pack(fill='x', padx=4, pady=4)

        def _row(label, var, lo, hi):
            r = tk.Frame(cfg_lf, bg=DARK); r.pack(fill='x', padx=6, pady=2)
            tk.Label(r, text=label, bg=DARK, fg=TEXT,
                     font=('Helvetica', 8), anchor='w').pack(fill='x')
            inner = tk.Frame(r, bg=DARK); inner.pack(fill='x')
            tk.Scale(inner, from_=lo, to=hi, variable=var, orient='horizontal',
                     bg=DARK, fg=TEXT, troughcolor='#445566',
                     highlightthickness=0, length=170).pack(side='left')
            tk.Label(inner, textvariable=var, bg=DARK, fg='#f1c40f',
                     font=('Courier', 9), width=6).pack(side='left')

        self._min_int  = tk.IntVar(value=0)
        self._min_nuc  = tk.IntVar(value=5000)
        self._crop_pad = tk.IntVar(value=150)
        _row('Min pixel intensity:',    self._min_int,   0,     50)
        _row('Min nucleus size (px):',  self._min_nuc,   1000,  30000)
        _row('Crop padding (px):',      self._crop_pad,  50,    400)

        tk.Button(left, text='▶  Run batch analysis',
                  command=self._run,
                  bg='#27ae60', fg='white', font=('Helvetica', 11, 'bold'),
                  relief='flat', cursor='hand2', pady=6).pack(fill='x', padx=4, pady=8)

        self._status = tk.StringVar(value='Select an LSM file and click Run.')
        tk.Label(left, textvariable=self._status, bg=DARK, fg='#95a5a6',
                 font=('Helvetica', 8), wraplength=230,
                 justify='left').pack(padx=4)

        # Log
        log_lf = tk.LabelFrame(left, text=' Log ', bg=DARK, fg='white',
                                font=('Helvetica', 9), bd=1, relief='groove')
        log_lf.pack(fill='both', expand=True, padx=4, pady=4)
        self._log = tk.Text(log_lf, height=12, state='disabled',
                            font=('Courier', 8), bg='#111111', fg='#aaffaa',
                            wrap='word', relief='flat')
        sb = tk.Scrollbar(log_lf, command=self._log.yview, bg=DARK)
        self._log.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self._log.pack(fill='both', expand=True)

        # ── Right panel: results notebook ─────────────────────────────────────
        right = tk.Frame(self, bg=DARKER)
        right.pack(side='left', fill='both', expand=True, padx=4, pady=6)

        self._nb = ttk.Notebook(right)
        self._nb.pack(fill='both', expand=True)

        ph = ttk.Frame(self._nb)
        self._nb.add(ph, text='  Results  ')
        tk.Label(ph, text='Results will appear here after running.',
                 font=('Helvetica', 13), fg='grey').pack(expand=True)
        self._placeholder = ph

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _browse(self):
        p = filedialog.askopenfilename(
            title='Select LSM file',
            filetypes=[('LSM files', '*.lsm'), ('All files', '*.*')])
        if p:
            self._path_full = p
            name = Path(p).name
            self._path_var.set(name if len(name) < 32 else '…' + name[-29:])

    def _log_msg(self, msg):
        self._log.configure(state='normal')
        self._log.insert('end', msg + '\n')
        self._log.see('end')
        self._log.configure(state='disabled')
        self.update_idletasks()

    def _add_cell_tab(self, cell_num, r):
        """Called on the main thread once a cell has been analysed."""
        outer = ttk.Frame(self._nb)
        self._nb.add(outer, text=f'  Cell {cell_num}  ')

        # Summary bar
        info = tk.Frame(outer, bg=DARK)
        info.pack(fill='x', padx=4, pady=2)
        summary = (f"Total IFD: {r['total_ifd']:,.0f}   |   "
                   f"Sat px: {r['n_sat']}   |   "
                   f"Nucleus: {r['nuke_area']:,} px   |   "
                   f"Cell: {r['cell_area']:,} px")
        tk.Label(info, text=summary, bg=DARK, fg='#f1c40f',
                 font=('Helvetica', 9)).pack(side='left', padx=8)

        inner_nb = ttk.Notebook(outer)
        inner_nb.pack(fill='both', expand=True)

        for name in CELL_TABS:
            f  = ttk.Frame(inner_nb)
            inner_nb.add(f, text=name)
            cv = FigCanvas(f)
            cv.pack(fill='both', expand=True)
            self._tab_cv[(cell_num, name)] = cv

        # Load PNGs into canvases — use after() so widgets have time to be shown
        def _load_pngs(cn=cell_num, paths=r['png_paths']):
            for tab_name, png_path in paths.items():
                canvas = self._tab_cv.get((cn, tab_name))
                if canvas and png_path.exists():
                    canvas.load_png(png_path)

        self.after(100, _load_pngs)

        # Switch to the first cell tab automatically
        if self._nb.index('end') == 2:   # placeholder + first cell
            self._nb.select(1)

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        if not self._path_full:
            messagebox.showwarning('No file', 'Please select an LSM file first.')
            return

        # Clear old results
        tabs = self._nb.tabs()
        for tab_id in tabs:
            if self.nametowidget(tab_id) is not self._placeholder:
                self._nb.forget(tab_id)
        self._tab_cv.clear()
        self._nb.select(0)

        self._log.configure(state='normal'); self._log.delete('1.0', 'end')
        self._log.configure(state='disabled')
        self._status.set('Running…')

        def worker():
            try:
                lsm_path = Path(self._path_full)
                self._log_msg(f'Loading {lsm_path.name} …')
                arr = load_lsm(lsm_path)
                H, W = arr.shape[2], arr.shape[3]
                self._log_msg(f'Shape: Z={arr.shape[0]}  {W}×{H} px')

                nuke_mp = max_project(arr, NUKE_CH)
                nuclei  = detect_all_nuclei(nuke_mp, self._min_nuc.get())
                self._log_msg(f'Found {len(nuclei)} nucleus/nuclei')

                if not nuclei:
                    self._log_msg('No nuclei found — try lowering Min nucleus size.')
                    self.after(0, lambda: self._status.set('No nuclei found.'))
                    return

                out_dir = lsm_path.parent / f'{lsm_path.stem}_batch'
                out_dir.mkdir(exist_ok=True)
                self._log_msg(f'Output → {out_dir.name}/')

                results = []
                for i, nuc in enumerate(nuclei):
                    cell_num = i + 1
                    stem = f'cell_{cell_num:02d}'
                    self._log_msg(f'  {stem}: analysing …')
                    y0, x0, y1, x1 = nuc['bbox']
                    y0c = max(0, y0 - self._crop_pad.get())
                    y1c = min(H, y1 + self._crop_pad.get())
                    x0c = max(0, x0 - self._crop_pad.get())
                    x1c = min(W, x1 + self._crop_pad.get())
                    lyso_crop = arr[:, LYSO_CH, y0c:y1c, x0c:x1c]
                    nuke_crop = arr[:, NUKE_CH, y0c:y1c, x0c:x1c]
                    try:
                        r = analyse_cell(lyso_crop, nuke_crop, stem, out_dir,
                                         min_intensity=self._min_int.get())
                        results.append((cell_num, r))
                        self._log_msg(
                            f'  {stem}: IFD={r["total_ifd"]:,.0f}  sat={r["n_sat"]}  ✓')
                        self.after(0, lambda cn=cell_num, rv=r: self._add_cell_tab(cn, rv))
                    except Exception:
                        self._log_msg(f'  {stem}: ERROR\n{traceback.format_exc()}')

                # Write summary CSV
                csv_path = out_dir / 'summary.csv'
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    wr = csv.writer(f)
                    wr.writerow(['cell', 'total_ifd', 'n_sat', 'bg_val',
                                 'nuke_area_px', 'cell_area_px',
                                 'pct_0_25', 'pct_25_50', 'pct_50_75', 'pct_75_100'])
                    for cell_num, r in results:
                        t = r['total_ifd']
                        pcts = [r['ifd_z'][z] / t * 100 if t > 0 else 0.0
                                for z in range(N_BINS)]
                        wr.writerow([r['stem'], f"{t:.1f}", r['n_sat'],
                                     f"{r['bg_val']:.1f}",
                                     r['nuke_area'], r['cell_area'],
                                     f"{pcts[0]:.2f}", f"{pcts[1]:.2f}",
                                     f"{pcts[2]:.2f}", f"{pcts[3]:.2f}"])

                n_ok = len(results)
                msg  = f'Done — {n_ok}/{len(nuclei)} cells analysed.'
                self._log_msg(f'\n{msg}')
                self._log_msg(f'summary.csv saved.')
                self.after(0, lambda: self._status.set(msg))

            except Exception:
                err = traceback.format_exc()
                self._log_msg(f'FATAL ERROR:\n{err}')
                self.after(0, lambda: self._status.set('Error — see log.'))

        threading.Thread(target=worker, daemon=True).start()


if __name__ == '__main__':
    App().mainloop()
