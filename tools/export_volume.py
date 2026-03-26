"""Web viewer launcher: export .npy volume to uint8 .raw and serve.

Usage:
    # Export a .npy volume and start the web viewer:
    uv run python viewer/export_volume.py --npy path/to/volume.npy

    # Export with custom percentile clipping:
    uv run python viewer/export_volume.py --npy vol.npy --pct-lo 1.0 --pct-hi 99.5

    # Just serve (volume.raw already exists):
    uv run python viewer/export_volume.py --serve

    # Export and immediately serve:
    uv run python viewer/export_volume.py --npy vol.npy --serve
"""

import argparse
import http.server
import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")


def export_volume(volume: np.ndarray, output_dir: str = DATA_DIR,
                  pct_lo: float = 0.5, pct_hi: float = 99.8):
    """Normalize volume to uint8 and save as raw binary + metadata JSON."""
    os.makedirs(output_dir, exist_ok=True)

    v = volume.astype(np.float32)
    vmin = np.percentile(v, pct_lo)
    vmax = np.percentile(v, pct_hi)
    v = np.clip((v - vmin) / (vmax - vmin + 1e-10), 0, 1)
    data = (v * 255).astype(np.uint8)

    # WebGL Data3DTexture(data, width, height, depth):
    #   index = x + y*width + z*width*height
    # volume shape is (H, W, D) = (y, x, z)
    # Transpose to (D, H, W) for C-order -> x varies fastest
    data_gl = np.ascontiguousarray(data.transpose(2, 0, 1))

    raw_path = os.path.join(output_dir, "volume.raw")
    data_gl.tofile(raw_path)

    meta = {
        "width":  int(volume.shape[1]),   # W = x
        "height": int(volume.shape[0]),   # H = y
        "depth":  int(volume.shape[2]),   # D = z
    }
    meta_path = os.path.join(output_dir, "volume_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(raw_path) / 1024 / 1024
    print(f"Exported {volume.shape} -> {raw_path} ({size_mb:.1f} MB)")
    print(f"Metadata: {meta_path}")


def serve(port: int = 8080):
    """Serve the viewer directory on localhost."""
    os.chdir(os.path.dirname(SCRIPT_DIR))
    handler = http.server.SimpleHTTPRequestHandler
    with http.server.HTTPServer(("localhost", port), handler) as srv:
        print(f"\n  Open in browser: http://localhost:{port}\n")
        srv.serve_forever()


def main():
    ap = argparse.ArgumentParser(description="Web viewer: export .npy to .raw and serve")
    ap.add_argument("--npy", type=str, help="Volume .npy file to export")
    ap.add_argument("--serve", action="store_true", help="Start HTTP server after export")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--pct-lo", type=float, default=0.5,
                    help="Low percentile for normalization (default: 0.5)")
    ap.add_argument("--pct-hi", type=float, default=99.8,
                    help="High percentile for normalization (default: 99.8)")
    args = ap.parse_args()

    if not args.npy and not args.serve:
        ap.print_help()
        print("\nError: provide --npy to export, --serve to start server, or both.")
        sys.exit(1)

    if args.npy:
        volume = np.load(args.npy)
        print(f"Loaded {args.npy}: {volume.shape}")
        export_volume(volume, pct_lo=args.pct_lo, pct_hi=args.pct_hi)

    if args.serve:
        serve(args.port)


if __name__ == "__main__":
    main()
