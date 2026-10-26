"""seeds.py - pick seed points for surface tracing automatically.

For a given height z, the scroll's cross-section is segmented from a coarse CT level, and seeds are
placed at a chosen fraction of the way from the scroll's centre of mass to its edge, in several
directions. Each seed is snapped to the nearest voxel of the published surface prediction, so the
tracer starts exactly on a predicted sheet.

  python -m vesuvius_survey.seeds --scroll PHerc0211 --z 0.5 --radius 0.7 --n 2 > seeds.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import zarr
from scipy import ndimage

from .trace import scroll_info


def pick(scroll: str, z_frac: float = 0.5, r_frac: float = 0.7, n: int = 2,
         angles=(np.pi / 2, np.pi, 0.0, -np.pi / 2), level: int = 3) -> list[dict]:
    info = scroll_info(scroll)
    ct = zarr.open_group(info['ct_url'], mode='r')
    sp = zarr.open_group(info['surface_url'], mode='r')['0']
    shape = ct['0'].shape
    z = int(shape[0] * z_frac)
    f = 2 ** level
    a = np.asarray(ct[str(level)][z // f])
    m = ndimage.binary_fill_holes(ndimage.binary_opening(a > 40, iterations=2))
    lab, k = ndimage.label(m)
    if k == 0:
        return []
    m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    cy, cx = ndimage.center_of_mass(m)
    out = []
    for ang in angles:
        if len(out) >= n:
            break
        dy, dx = np.sin(ang), np.cos(ang)
        r = 0
        while True:
            y, x = int(cy + dy * r), int(cx + dx * r)
            if not (0 <= y < m.shape[0] and 0 <= x < m.shape[1]) or not m[y, x]:
                break
            r += 1
        X, Y = int((cx + dx * r_frac * r) * f), int((cy + dy * r_frac * r) * f)
        rr = 16
        blk = np.asarray(sp[max(z - rr, 0):z + rr, max(Y - rr, 0):Y + rr, max(X - rr, 0):X + rr])
        idx = np.argwhere(blk > 0)
        if len(idx) == 0:
            continue
        q = idx[np.linalg.norm(idx - rr, axis=1).argmin()]
        out.append({'scroll': scroll, 'x': int(X - rr + q[2]), 'y': int(Y - rr + q[1]), 'z': int(z - rr + q[0])})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scroll', required=True)
    ap.add_argument('--z', type=float, default=0.5, help='height as a fraction of the scan (0-1)')
    ap.add_argument('--radius', type=float, default=0.7, help='fraction of the way from centre to edge')
    ap.add_argument('--n', type=int, default=2)
    a = ap.parse_args()
    print(json.dumps(pick(a.scroll, a.z, a.radius, a.n), indent=1))


if __name__ == '__main__':
    main()
