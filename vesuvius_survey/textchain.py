"""Letter-chain score: do letter-sized ink blobs line up along a fibre direction?"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
try:
    from .inkcheck import _norm, crossness_map
except ImportError:
    from inkcheck import _norm, crossness_map

def blobs(ink, valid, px_um, q=92, size_mm=(0.6, 6.0)):
    e = ndimage.gaussian_filter(_norm(ink, valid), 2)
    m = (e > np.percentile(e[valid], q)) & valid
    lab, n = ndimage.label(m)
    if n == 0: return np.zeros((0, 2)), []
    idx = np.arange(1, n + 1)
    area = ndimage.sum(np.ones_like(e), lab, idx)
    com = np.array(ndimage.center_of_mass(m, lab, idx))
    d = 2 * np.sqrt(area / np.pi) * px_um / 1000  # equivalent diameter mm
    keep = (d >= size_mm[0]) & (d <= size_mm[1])
    return com[keep], d[keep]

def chain_score(ink, ct, px_um, fibre_deg=None, pitch_mm=(1.5, 6.0), tol=15):
    valid = ct > 0
    if fibre_deg is None:
        c, ang, v = crossness_map(ct, px_scale=1.0)
        a = np.radians(ang[v] * 2); fibre_deg = (np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2) % 180
    pts, d = blobs(ink, valid, px_um)
    if len(pts) < 8: return {'n_blobs': int(len(pts)), 'chain': 0.0, 'axis': None}
    t = cKDTree(pts)
    lo, hi = pitch_mm[0] * 1000 / px_um, pitch_mm[1] * 1000 / px_um
    pairs = np.array(list(t.query_pairs(hi)))
    if len(pairs) == 0: return {'n_blobs': int(len(pts)), 'chain': 0.0, 'axis': None}
    v = pts[pairs[:, 1]] - pts[pairs[:, 0]]
    L = np.hypot(v[:, 0], v[:, 1]); v, L = v[L >= lo], L[L >= lo]
    angs = (np.degrees(np.arctan2(v[:, 0], v[:, 1])) + 180) % 180  # angle in image (row, col) frame
    best, best_axis = 0.0, None
    for ax in (fibre_deg, (fibre_deg + 90) % 180):
        dd = np.abs(((angs - ax) + 90) % 180 - 90)
        frac = (dd <= tol).mean()
        if frac > best: best, best_axis = frac, ax
    expected = 2 * tol / 180
    return {'n_blobs': int(len(pts)), 'pairs': int(len(L)), 'chain': float(best / expected), 'axis': best_axis}
