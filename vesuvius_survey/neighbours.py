"""neighbours.py - seeds on the windings either side of a lead, over its most face-consistent tile.

Writing continues onto neighbouring windings; texture artefacts usually do not. This puts seeds +-15/30/45 voxels
along the patch normal (snapped to the published surface prediction) so the same report card can be made for them.

  python -m vesuvius_survey.neighbours WORK_DIR PATCH_DIR --scroll PHerc0846B [--offsets -45 -30 -15 15 30 45]

WORK_DIR holds the patch's ink maps and ct_mid (as written by survey.py); PATCH_DIR is its tifxyz mesh.
Prints one JSON seed list ({"x","y","z","offset"}) to feed to trace.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr

from .report_card import TILE
from .trace import scroll_info


def best_tile(work: Path, ct: np.ndarray) -> tuple[int, int]:
    from .report_card import analyse
    res, ct, cross, valid, good, _ = analyse(work, ct, 9.0)
    # recompute per-tile strength on the consistent face
    from .inkcheck import _norm
    face = res['consistent_face'] if res['consistent_face'] in ('forward', 'reverse') else res['face']   # tie: use the best face
    suf = {'forward': ('', '_reverse'), 'reverse': ('_reverse', '')}[face]
    m = good if good.mean() > 0.05 else valid
    def top(s):
        a = _norm(tifffile.imread(work / f'seed42_s6{s}.tif').astype(np.float32)[:m.shape[0], :m.shape[1]], m)
        b = _norm(tifffile.imread(work / f'dnative_s6{s}.tif').astype(np.float32)[:m.shape[0], :m.shape[1]], m)
        ta, tb = a >= np.quantile(a[m], .97), b >= np.quantile(b[m], .97)
        return ta & tb & m, ta | tb
    (bi, ui), (bo, uo) = top(suf[0]), top(suf[1])
    best, where = -1.0, (0, 0)
    h, w = m.shape
    for y in range(0, h - TILE + 1, TILE // 2):
        for x in range(0, w - TILE + 1, TILE // 2):
            sl = (slice(y, y + TILE), slice(x, x + TILE))
            if m[sl].mean() < 0.5:
                continue
            i = bi[sl].sum() / max((ui[sl] & m[sl]).sum(), 1)
            o = bo[sl].sum() / max((uo[sl] & m[sl]).sum(), 1)
            if i - o > best:
                best, where = i - o, (y, x)
    return where


def seeds(work: Path, patch: Path, scroll: str, offsets=(-45, -30, -15, 15, 30, 45), snap: int = 5) -> list[dict]:
    ct = tifffile.imread(work / 'ct_mid.tif').astype(np.float32)
    y, x = best_tile(work, ct)
    X, Y, Z = (tifffile.imread(patch / f'{c}.tif').astype(np.float64) for c in 'xyz')
    iy, ix = (y + TILE // 2) // 20, (x + TILE // 2) // 20
    P = lambda a, b: np.array([X[a, b], Y[a, b], Z[a, b]])
    n = np.cross(P(iy, ix + 1) - P(iy, ix - 1), P(iy + 1, ix) - P(iy - 1, ix))
    n /= np.linalg.norm(n)
    sp = zarr.open_group(scroll_info(scroll)['surface_url'], mode='r')['0']
    out = []
    for d in offsets:
        cx, cy, cz = (int(round(v)) for v in P(iy, ix) + d * n)
        blk = np.asarray(sp[cz - snap:cz + snap + 1, cy - snap:cy + snap + 1, cx - snap:cx + snap + 1])
        idx = np.argwhere(blk > 0)
        if not len(idx):
            continue
        b = idx[np.linalg.norm(idx - snap, axis=1).argmin()]
        out.append({'x': int(cx - snap + b[2]), 'y': int(cy - snap + b[1]), 'z': int(cz - snap + b[0]), 'offset': d})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('work', type=Path)
    ap.add_argument('patch', type=Path)
    ap.add_argument('--scroll', required=True)
    ap.add_argument('--offsets', type=int, nargs='+', default=[-45, -30, -15, 15, 30, 45])
    a = ap.parse_args()
    print(json.dumps(seeds(a.work, a.patch, a.scroll, a.offsets), indent=1))


if __name__ == '__main__':
    main()
