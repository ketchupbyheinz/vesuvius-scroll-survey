"""Render a flattened surface volume (layer stack) from a tifxyz mesh and a CT zarr.

Mirrors vc_render_tifxyz in its default "linear" mode:
  * pixel positions are bilinear interpolations of the mesh nodes,
  * normals are per-node central-difference normals (xv x yv, normalised),
    taken nearest-node for each pixel,
  * layer k samples the CT trilinearly at (k - (n-1)/2) * step voxels along the normal.

Output: zarr array [layers, rows, cols] uint8, plus a PNG of the middle layer.

Usage:
  python render_patch.py MESH.tifxyz CT_URL OUT.zarr --rows R0 R1 --cols C0 C1
         [--layers 28] [--step 1] [--tile 512]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy.ndimage import map_coordinates


def load_mesh(path: Path):
    x = tifffile.imread(path / "x.tif").astype(np.float32)
    y = tifffile.imread(path / "y.tif").astype(np.float32)
    z = tifffile.imread(path / "z.tif").astype(np.float32)
    meta = json.loads((path / "meta.json").read_text())
    scale = float(meta["scale"][0])
    pts = np.stack([x, y, z], axis=-1)  # rows, cols, (x, y, z)
    valid = ~((x == -1) | (y == -1) | (z == -1))
    return pts, valid, scale


def node_normals(pts: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = np.full(pts.shape, np.nan, np.float32)
    xv = pts[1:-1, 2:] - pts[1:-1, :-2]
    yv = pts[2:, 1:-1] - pts[:-2, 1:-1]
    ok = valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
    c = np.cross(xv, yv)
    norm = np.linalg.norm(c, axis=-1, keepdims=True)
    c = c / np.where(norm > 0, norm, np.nan)
    c[~ok] = np.nan
    n[1:-1, 1:-1] = c
    return n


def pixel_geometry(pts, valid, nrm, scale, r0, r1, c0, c1):
    rows = (np.arange(r0, r1, dtype=np.float64) + 0.5) * scale  # pixel centres, as vc_render_tifxyz
    cols = (np.arange(c0, c1, dtype=np.float64) + 0.5) * scale
    gr, gc = np.meshgrid(rows, cols, indexing="ij")
    H, W = pts.shape[:2]
    inside = (gr >= 0) & (gr <= H - 1) & (gc >= 0) & (gc <= W - 1)
    ir = np.clip(np.floor(gr).astype(int), 0, H - 2)
    ic = np.clip(np.floor(gc).astype(int), 0, W - 2)
    fr = (gr - ir)[..., None]
    fc = (gc - ic)[..., None]
    p00, p01 = pts[ir, ic], pts[ir, ic + 1]
    p10, p11 = pts[ir + 1, ic], pts[ir + 1, ic + 1]
    pos = (1 - fr) * ((1 - fc) * p00 + fc * p01) + fr * ((1 - fc) * p10 + fc * p11)
    cell_ok = valid[ir, ic] & valid[ir, ic + 1] & valid[ir + 1, ic] & valid[ir + 1, ic + 1] & inside
    nr = np.clip(np.rint(gr).astype(int), 0, H - 1)
    nc = np.clip(np.rint(gc).astype(int), 0, W - 1)
    nn = nrm[nr, nc]
    ok = cell_ok & np.isfinite(nn).all(-1)
    return pos.astype(np.float32), nn.astype(np.float32), ok


def render_tile(ct, pos, nn, ok, offsets):
    L = len(offsets)
    h, w = ok.shape
    out = np.zeros((L, h, w), np.uint8)
    if not ok.any():
        return out
    p = pos[ok]
    n = nn[ok]
    allp = p[None, :, :] + offsets[:, None, None] * n[None, :, :]  # L, N, (x,y,z)
    lo = np.floor(allp.reshape(-1, 3).min(0)).astype(int) - 1
    hi = np.ceil(allp.reshape(-1, 3).max(0)).astype(int) + 2
    shape_zyx = ct.shape
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [shape_zyx[2], shape_zyx[1], shape_zyx[0]])
    block = None
    for attempt in range(8):
        try:
            block = np.asarray(ct[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]).astype(np.float32)
            break
        except Exception as e:  # network hiccups while streaming the CT
            wait = min(60, 2 ** attempt)
            print(f"  read failed ({type(e).__name__}: {e}); retry in {wait}s", flush=True)
            time.sleep(wait)
    if block is None:
        raise RuntimeError("CT read failed after 8 attempts")
    coords = np.stack([
        allp[..., 2] - lo[2],
        allp[..., 1] - lo[1],
        allp[..., 0] - lo[0],
    ]).reshape(3, -1)
    vals = map_coordinates(block, coords, order=1, mode="constant", cval=0.0)
    vals = np.clip(np.floor(vals), 0, 255).astype(np.uint8).reshape(L, -1)
    for k in range(L):
        out[k][ok] = vals[k]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", type=Path)
    ap.add_argument("ct")
    ap.add_argument("out", type=Path)
    ap.add_argument("--rows", nargs=2, type=int, required=True)
    ap.add_argument("--cols", nargs=2, type=int, required=True)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="keep tiles finished by an earlier run")
    a = ap.parse_args(argv)

    pts, valid, scale = load_mesh(a.mesh)
    nrm = node_normals(pts, valid)
    root = zarr.open_group(a.ct, mode="r")
    ct = root["0"]
    print(f"CT shape {ct.shape} chunks {ct.chunks} dtype {ct.dtype}", flush=True)

    r0, r1 = a.rows
    c0, c1 = a.cols
    H, W = r1 - r0, c1 - c0
    center = 0.5 * (a.layers - 1)
    offsets = ((np.arange(a.layers) - center) * a.step).astype(np.float32)

    out = None
    if a.resume and a.out.exists():
        try:
            out = zarr.open_array(str(a.out), mode="r+")
            if out.shape != (a.layers, H, W):
                out = None
        except Exception:
            out = None
    if out is None:
        out = zarr.open_array(str(a.out), mode="w", shape=(a.layers, H, W),
                              chunks=(a.layers, 256, 256), dtype="u1", zarr_format=2)
    done_dir = a.out.parent / (a.out.name + ".tiles_done")
    done_dir.mkdir(exist_ok=True)
    tiles = [(tr, tc) for tr in range(0, H, a.tile) for tc in range(0, W, a.tile)]
    t0 = time.time()

    def job(t):
        tr, tc = t
        marker = done_dir / f"{tr}_{tc}_{a.tile}"
        if a.resume and marker.exists():
            return t
        rr1, cc1 = min(tr + a.tile, H), min(tc + a.tile, W)
        pos, nn, ok = pixel_geometry(pts, valid, nrm, scale, r0 + tr, r0 + rr1, c0 + tc, c0 + cc1)
        img = render_tile(ct, pos, nn, ok, offsets)
        out[:, tr:rr1, tc:cc1] = img
        marker.touch()
        return t

    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for _ in ex.map(job, tiles):
            done += 1
            print(f"tile {done}/{len(tiles)}  {time.time() - t0:.0f}s", flush=True)

    out.attrs.update({
        "source_mesh": str(a.mesh), "ct": a.ct, "rows": [r0, r1], "cols": [c0, c1],
        "num_slices": a.layers, "slice_step": a.step,
        "renderer": "render_patch.py (vc_render_tifxyz linear-mode equivalent)",
    })
    from PIL import Image
    mid = np.asarray(out[a.layers // 2])
    Image.fromarray(mid).save(a.out.with_suffix(".mid.png"))
    print(f"done in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
