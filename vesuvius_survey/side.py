"""side.py - which ink face of a traced patch faces the centre of the scroll?

Herculaneum rolls carry their text on the inside face. A face-consistent ink signal on the inner face is a better
lead than one on the outer face. Both known-text patches of PHerc. 343 put their signal on the inner face.

'forward' is the +normal direction of render_patch (layer index increasing), as used by the ink models'
`--direction forward`. The scroll centre at the patch height comes from a coarse level of the masked CT.

  python -m vesuvius_survey.side PATCH_DIR --scroll PHerc0211
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage

from .render_patch import load_mesh, node_normals
from .trace import scroll_info


@functools.lru_cache(None)
def scroll_centre(scroll: str, z: int, level: int = 3) -> tuple[float, float]:
    ct = zarr.open_group(scroll_info(scroll)['ct_url'], mode='r')
    f = 2 ** level
    a = np.asarray(ct[str(level)][z // f])
    m = ndimage.binary_fill_holes(ndimage.binary_opening(a > 40, iterations=2))
    lab, k = ndimage.label(m)
    m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    cy, cx = ndimage.center_of_mass(m)
    return cx * f, cy * f


def forward_inward(patch: Path, scroll: str) -> float:
    """Mean sign of (normal . direction to the scroll axis): +1 = forward faces the centre, -1 = reverse does."""
    p, v, _ = load_mesh(Path(patch))
    n = node_normals(p, v)
    ok = np.isfinite(n).all(-1)
    P, N = p[ok], n[ok]
    cx, cy = scroll_centre(scroll, int(np.median(P[:, 2])) // 64 * 64)
    inward = np.stack([cx - P[:, 0], cy - P[:, 1]], 1)
    inward /= np.linalg.norm(inward, axis=1, keepdims=True)
    return float(np.mean(np.sign((N[:, :2] * inward).sum(1))))


def curvature_sign(patch: Path, pairs: int = 20000, seed: int = 0) -> float:
    """Geometry-only cross-check. Normals of a rolled sheet converge towards the roll axis:
    median over node pairs of (n_i - n_j).(p_i - p_j) / |p_i - p_j|^2 (x1000). < 0: forward faces the centre of
    curvature. Near 0 for flattened or crushed regions, where only the scroll-centre test is informative."""
    p, v, _ = load_mesh(Path(patch))
    n = node_normals(p, v)
    ok = np.isfinite(n).all(-1)
    P, N = p[ok].astype(float), n[ok].astype(float)
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(P), pairs), rng.integers(0, len(P), pairs)
    d = P[i] - P[j]
    L = (d ** 2).sum(1)
    m = L > 200 ** 2
    return float(np.median(((N[i] - N[j]) * d).sum(1)[m] / L[m]) * 1000)


def side_of(face: str, score: float) -> str:
    """'inner' / 'outer' for face 'forward' or 'reverse'; 'unclear' for a tie or a weak scroll-centre test."""
    if face not in ('forward', 'reverse') or abs(score) <= 0.5:
        return 'unclear'
    return 'inner' if (face == 'forward') == (score > 0) else 'outer'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('patch', type=Path)
    ap.add_argument('--scroll', required=True)
    a = ap.parse_args()
    s = forward_inward(a.patch, a.scroll)
    c = curvature_sign(a.patch)
    print(json.dumps({'forward_inward_score': round(s, 3), 'forward_face_is': side_of('forward', s),
                      'curvature': round(c, 3), 'curvature_agrees': (abs(c) > 0.05) and ((s > 0) == (c < 0))}))


if __name__ == '__main__':
    main()
