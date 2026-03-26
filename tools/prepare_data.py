"""GUI tool to preview and export 3D volumes for the web viewer.

Load raw 3D data (.npy / .tif), adjust percentile clip + gamma,
preview MIP in real-time, then save as uint8 to data/<name>/.

Usage:
    python tools/prepare_data.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import json
import os
import glob

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(REPO_DIR, "raw_data")
DATA_DIR = os.path.join(REPO_DIR, "data")
DATASETS_JSON = os.path.join(DATA_DIR, "datasets.json")


def load_volume(path: str) -> np.ndarray:
    """Load a 3D volume from .npy or .tif, return as float32 (H, W, D)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        vol = np.load(path).astype(np.float32)
    elif ext in (".tif", ".tiff"):
        import tifffile
        img = tifffile.imread(path).astype(np.float32)
        # Assume (Z, H, W) -> (H, W, D)
        if img.ndim == 3:
            vol = img.transpose(1, 2, 0)
        else:
            raise ValueError(f"Expected 3D TIFF, got shape {img.shape}")
    else:
        raise ValueError(f"Unsupported format: {ext}")
    return vol


def process_volume(vol: np.ndarray, pct_lo: float, pct_hi: float,
                   gamma: float) -> np.ndarray:
    """Percentile clip -> gamma -> uint8."""
    vmin = np.percentile(vol, pct_lo)
    vmax = np.percentile(vol, pct_hi)
    v = np.clip((vol - vmin) / (vmax - vmin + 1e-10), 0, 1)
    v = np.power(v, gamma)
    return (v * 255).astype(np.uint8)


class PrepareApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Volume Prepare Tool")
        self.root.geometry("900x700")
        self.root.minsize(700, 550)

        self.vol_raw = None       # original float32 (H, W, D)
        self.vol_processed = None  # uint8 after processing
        self.source_path = None

        self._build_ui()
        self._scan_raw_files()

    def _build_ui(self):
        # ---- Top: file selection ----
        top = ttk.LabelFrame(self.root, text="  Source  ", padding=6)
        top.pack(fill="x", padx=8, pady=(8, 2))

        ttk.Label(top, text="Raw file:").pack(side="left")
        self.file_combo = ttk.Combobox(top, state="readonly", width=50)
        self.file_combo.pack(side="left", padx=4)
        self.file_combo.bind("<<ComboboxSelected>>", self._on_file_select)

        ttk.Button(top, text="Browse...", command=self._browse).pack(side="left", padx=4)

        self.info_var = tk.StringVar(value="No file loaded")
        ttk.Label(top, textvariable=self.info_var, foreground="gray").pack(
            side="left", padx=8)

        # ---- Middle: controls + preview ----
        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=8, pady=4)

        # Controls (left)
        ctrl = ttk.LabelFrame(mid, text="  Processing  ", padding=6)
        ctrl.pack(side="left", fill="y", padx=(0, 4))

        self.pct_lo_var = tk.DoubleVar(value=0.5)
        self.pct_hi_var = tk.DoubleVar(value=99.8)
        self.gamma_var = tk.DoubleVar(value=1.0)

        # Clip Lo: slider 0-100 + spinbox for fine tuning
        f_lo = ttk.Frame(ctrl)
        f_lo.pack(fill="x", pady=2)
        ttk.Label(f_lo, text="Clip Lo %", width=10).pack(side="left")
        ttk.Scale(f_lo, from_=0, to=100, variable=self.pct_lo_var,
                  orient="horizontal",
                  command=self._on_param_change).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Spinbox(f_lo, from_=0, to=100, increment=0.001, width=8,
                    textvariable=self.pct_lo_var, format="%.3f",
                    command=self._on_param_change).pack(side="left", padx=2)

        # Clip Hi: slider 0-100 + spinbox for fine tuning
        f_hi = ttk.Frame(ctrl)
        f_hi.pack(fill="x", pady=2)
        ttk.Label(f_hi, text="Clip Hi %", width=10).pack(side="left")
        ttk.Scale(f_hi, from_=0, to=100, variable=self.pct_hi_var,
                  orient="horizontal",
                  command=self._on_param_change).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Spinbox(f_hi, from_=0, to=100, increment=0.001, width=8,
                    textvariable=self.pct_hi_var, format="%.3f",
                    command=self._on_param_change).pack(side="left", padx=2)

        # Gamma: slider + spinbox
        f_g = ttk.Frame(ctrl)
        f_g.pack(fill="x", pady=2)
        ttk.Label(f_g, text="Gamma", width=10).pack(side="left")
        ttk.Scale(f_g, from_=0.1, to=3.0, variable=self.gamma_var,
                  orient="horizontal",
                  command=self._on_param_change).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Spinbox(f_g, from_=0.01, to=5.0, increment=0.001, width=8,
                    textvariable=self.gamma_var, format="%.3f",
                    command=self._on_param_change).pack(side="left", padx=2)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=6)

        # Histogram (small, in left panel)
        self.hist_fig = Figure(figsize=(3, 1.8), dpi=80, facecolor="#2a2a2a")
        self.hist_fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.18)
        self.ax_hist = self.hist_fig.add_subplot(111)
        self.ax_hist.set_facecolor("#1a1a1a")
        self.ax_hist.set_title("Histogram (raw)", color="#aaa", fontsize=8)
        self.ax_hist.tick_params(colors="#666", labelsize=7)
        for spine in self.ax_hist.spines.values():
            spine.set_color("#444")
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=ctrl)
        self.hist_canvas.get_tk_widget().pack(fill="x", pady=2)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=6)

        # Stats
        self.stats_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self.stats_var, justify="left",
                  font=("Consolas", 9)).pack(fill="x", pady=4)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=8)

        # Export controls
        ttk.Label(ctrl, text="Dataset name:").pack(anchor="w")
        self.name_var = tk.StringVar(value="")
        ttk.Entry(ctrl, textvariable=self.name_var, width=25).pack(
            fill="x", pady=2)

        ttk.Label(ctrl, text="Display name:").pack(anchor="w")
        self.display_name_var = tk.StringVar(value="")
        ttk.Entry(ctrl, textvariable=self.display_name_var, width=25).pack(
            fill="x", pady=2)

        ttk.Button(ctrl, text="Save to data/",
                   command=self._on_save).pack(fill="x", pady=(8, 2))
        self.save_status = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self.save_status, foreground="green").pack(
            fill="x")

        # Preview (right) - 2 panels: XY MIP and XZ MIP
        preview = ttk.Frame(mid)
        preview.pack(side="left", fill="both", expand=True)

        self.fig = Figure(figsize=(6, 5), dpi=96, facecolor="#1a1a1a")
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                                 wspace=0.05, hspace=0.15)
        self.ax_xy = self.fig.add_subplot(211)
        self.ax_xz = self.fig.add_subplot(212)
        for ax in (self.ax_xy, self.ax_xz):
            ax.set_facecolor("black")
            ax.axis("off")
        self.ax_xy.set_title("XY MIP", color="#aaa", fontsize=10)
        self.ax_xz.set_title("XZ MIP", color="#aaa", fontsize=10)
        self.im_xy = None
        self.im_xz = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=preview)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _scan_raw_files(self):
        patterns = ["*.npy", "*.tif", "*.tiff"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(RAW_DIR, p)))
        names = [os.path.basename(f) for f in files]
        self.file_combo["values"] = names
        self._raw_files = {os.path.basename(f): f for f in files}

    def _browse(self):
        path = filedialog.askopenfilename(
            initialdir=RAW_DIR,
            filetypes=[("Volume files", "*.npy *.tif *.tiff"), ("All", "*.*")])
        if path:
            self._load(path)

    def _on_file_select(self, _event=None):
        name = self.file_combo.get()
        if name in self._raw_files:
            self._load(self._raw_files[name])

    def _load(self, path):
        try:
            self.vol_raw = load_volume(path)
            self.source_path = path
            H, W, D = self.vol_raw.shape
            self.info_var.set(f"{H}x{W}x{D}  range: {self.vol_raw.min():.1f}-{self.vol_raw.max():.1f}")

            # Auto-fill name
            base = os.path.splitext(os.path.basename(path))[0]
            self.name_var.set(base.replace(" ", "_").replace("-", "_"))
            self.display_name_var.set(base.replace("_", " ").title())

            # Pre-compute histogram of raw data (log scale, fixed)
            self._draw_raw_histogram()
            self._update_preview()
        except Exception as e:
            self.info_var.set(f"Error: {e}")

    def _draw_raw_histogram(self):
        """Draw histogram of raw data with log y-axis."""
        ax = self.ax_hist
        ax.clear()
        ax.set_facecolor("#1a1a1a")
        ax.set_title("Histogram (raw)", color="#aaa", fontsize=8)
        ax.tick_params(colors="#666", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#444")

        flat = self.vol_raw.ravel()
        # Use 200 bins, skip zeros for cleaner view
        nonzero = flat[flat > 0]
        if len(nonzero) > 0:
            ax.hist(nonzero, bins=200, color="#6a6", alpha=0.8, log=True)
        ax.set_xlim(flat.min(), np.percentile(flat, 99.99))

        # Store clip line references
        self._clip_lo_line = ax.axvline(0, color="#f44", lw=1, ls="--")
        self._clip_hi_line = ax.axvline(0, color="#f44", lw=1, ls="--")
        self.hist_canvas.draw_idle()

    def _on_param_change(self, *_):
        self._update_preview()

    def _update_preview(self):
        if self.vol_raw is None:
            return
        pct_lo = self.pct_lo_var.get()
        pct_hi = self.pct_hi_var.get()
        gamma = self.gamma_var.get()

        self.vol_processed = process_volume(self.vol_raw, pct_lo, pct_hi, gamma)

        # Update clip lines on histogram
        if hasattr(self, "_clip_lo_line"):
            vmin = np.percentile(self.vol_raw, pct_lo)
            vmax = np.percentile(self.vol_raw, pct_hi)
            self._clip_lo_line.set_xdata([vmin, vmin])
            self._clip_hi_line.set_xdata([vmax, vmax])
            self.hist_canvas.draw_idle()

        # Stats
        v = self.vol_processed
        mip = np.max(v, axis=2)
        lines = [
            f"uint8 volume:",
            f"  mean={v.mean():.1f}  p50={np.median(v):.0f}",
            f"  p95={np.percentile(v,95):.0f}  p99={np.percentile(v,99):.0f}",
            f"  ==255: {np.mean(v==255)*100:.2f}%",
            f"",
            f"MIP (XY):",
            f"  mean={mip.mean():.1f}  p50={np.median(mip):.0f}",
            f"  p95={np.percentile(mip,95):.0f}  p99={np.percentile(mip,99):.0f}",
            f"  ==255: {np.mean(mip==255)*100:.1f}%",
            f"  ==0:   {np.mean(mip==0)*100:.1f}%",
        ]
        self.stats_var.set("\n".join(lines))

        # XY MIP
        if self.im_xy is None:
            self.im_xy = self.ax_xy.imshow(mip, cmap="gray", vmin=0, vmax=255)
        else:
            self.im_xy.set_data(mip)

        # XZ MIP
        xz = np.max(v, axis=0)  # (W, D)
        if self.im_xz is None:
            self.im_xz = self.ax_xz.imshow(xz.T, cmap="gray", vmin=0, vmax=255,
                                            aspect="auto")
        else:
            self.im_xz.set_data(xz.T)

        self.canvas.draw_idle()

    def _on_save(self):
        if self.vol_processed is None:
            messagebox.showwarning("No data", "Load a file first.")
            return
        ds_id = self.name_var.get().strip()
        ds_name = self.display_name_var.get().strip()
        if not ds_id:
            messagebox.showwarning("No name", "Enter a dataset name.")
            return
        if not ds_name:
            ds_name = ds_id

        # Export
        out_dir = os.path.join(DATA_DIR, ds_id)
        os.makedirs(out_dir, exist_ok=True)

        data = self.vol_processed
        H, W, D = data.shape
        data_gl = np.ascontiguousarray(data.transpose(2, 0, 1))
        data_gl.tofile(os.path.join(out_dir, "volume.raw"))

        meta = {"width": W, "height": H, "depth": D}
        with open(os.path.join(out_dir, "volume_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Update datasets.json
        if os.path.exists(DATASETS_JSON):
            with open(DATASETS_JSON) as f:
                datasets = json.load(f)
        else:
            datasets = []

        # Update or add entry
        found = False
        for ds in datasets:
            if ds["id"] == ds_id:
                ds["name"] = ds_name
                found = True
                break
        if not found:
            datasets.append({"id": ds_id, "name": ds_name})

        with open(DATASETS_JSON, "w") as f:
            json.dump(datasets, f, indent=2)

        size_mb = os.path.getsize(os.path.join(out_dir, "volume.raw")) / 1024 / 1024
        self.save_status.set(f"Saved {ds_id} ({size_mb:.1f} MB)")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    PrepareApp().run()
