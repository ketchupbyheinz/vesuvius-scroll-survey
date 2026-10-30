"""inkvote.py - keep only ink that repeats where independently traced patches overlap.

Many patches on one winding cover the same papyrus. Real ink sits at the same 3D spot in every patch that covers it;
crack hits and model noise move around from patch to patch. For every patch this module

  1. masks cracks: off-sheet pixels (report card sheet check) and the darkest DARK_Q of on-sheet CT, dilated,
  2. turns seed42 and dense_native into within-patch percentile ranks and keeps the lower of the two (both must agree),
  3. maps every pixel to 3D with the mesh, assigns it to a winding, and lays it onto a flat grid (the best-fit plane of
     the main winding; PHerc. 1545 is crushed flat, so a plane suits it better than unrolling around a centre),

and averages per grid cell over the patches covering it. Windings are numbered from tile to tile: every patch is cut
into ~1 mm tiles, overlapping tiles are compared along the sheet normal (0 = same sheet, one spacing = next winding).
The two faces of the sheet are kept apart: patch normals are made to agree by passing the direction from patch to
overlapping patch, starting at the first patch; 'front' is the side that patch's normal points to.

  python -m vesuvius_survey.inkvote vote OUT_DIR PATCH_DIR [PATCH_DIR ...] [--grid 3] [--stride 2]
      PATCH_DIR holds mesh/{x,y,z.tif,meta.json}, ct_mid.tif, seed42_s6[_reverse].tif, dnative_s6[_reverse].tif
  python -m vesuvius_survey.inkvote queue STITCH_DIR SURVEY_DIR CT_URL [--tag V] [--cover 3]
      picks patches so most of the winding is covered >= 3 times and writes survey dirs for the laptop worker
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

try:
    from .render_patch import load_mesh, node_normals
    from .report_card import sheet_masks
except ImportError:  # plain script
    sys.path.insert(0, str(Path(__file__).parent))
    from render_patch import load_mesh, node_normals  # noqa
    from report_card import sheet_masks  # noqa

DARK_Q = 0.15      # darkest share of on-sheet CT treated as crack / gap
DARK_SIGMA = 4     # px blur before picking dark pixels
DARK_DILATE = 5    # px
TOP = 0.97         # "both models in their top 3 %"
MODELS = ('seed42', 'dnative')


# ----------------------------------------------------------------------------------------------- geometry
def mesh_nodes(mesh_dir: Path):
    pts, valid, scale = load_mesh(mesh_dir)
    nrm = node_normals(pts, valid)
    ok = valid & np.isfinite(nrm).all(-1)
    return pts, valid, scale, nrm, ok


def pixel_xyz(pts, valid, scale, rows, cols):
    """3D position of render pixels (rows x cols, pixel-centre convention of render_patch) + validity."""
    H, W = pts.shape[:2]
    gr = (rows.astype(np.float64) + 0.5) * scale
    gc = (cols.astype(np.float64) + 0.5) * scale
    gr, gc = np.meshgrid(gr, gc, indexing='ij')
    inside = (gr <= H - 1) & (gc <= W - 1)
    ir = np.clip(np.floor(gr).astype(int), 0, H - 2); ic = np.clip(np.floor(gc).astype(int), 0, W - 2)
    fr = (gr - ir)[..., None]; fc = (gc - ic)[..., None]
    pos = (1 - fr) * ((1 - fc) * pts[ir, ic] + fc * pts[ir, ic + 1]) + fr * ((1 - fc) * pts[ir + 1, ic] + fc * pts[ir + 1, ic + 1])
    ok = valid[ir, ic] & valid[ir, ic + 1] & valid[ir + 1, ic] & valid[ir + 1, ic + 1] & inside
    return pos.astype(np.float32), ok


# ----------------------------------------------------------------------------------------------- per patch
def rank_map(a: np.ndarray, use: np.ndarray) -> np.ndarray:
    """Percentile rank (0..1, mid-rank for ties) of each pixel's value among the ``use`` pixels."""
    a = a.astype(np.int64) if a.dtype.kind in 'ui' and a.max() < 65536 else a
    if a.dtype.kind in 'i':
        h = np.bincount(a[use], minlength=int(a.max()) + 1).astype(np.float64)
        cdf = (np.cumsum(h) - h / 2) / max(h.sum(), 1)
        return cdf[a].astype(np.float32)
    v = np.sort(a[use])
    lo = np.searchsorted(v, a, 'left'); hi = np.searchsorted(v, a, 'right')
    return ((lo + hi) / 2 / max(len(v), 1)).astype(np.float32)


def crack_mask(ct, m):
    """Darkest DARK_Q of on-sheet CT after a light blur (sigma DARK_SIGMA px, so single dark fibre pixels do not
    count), grown by DARK_DILATE px."""
    import cv2
    b = cv2.GaussianBlur(ct.astype(np.float32), (0, 0), DARK_SIGMA)
    dark = (b < np.quantile(b[m], DARK_Q)) & m
    k = np.ones((DARK_DILATE, DARK_DILATE), np.uint8)
    return cv2.dilate(dark.astype(np.uint8), k).astype(bool)


def patch_scores(d: Path):
    """use mask and agreed rank per face ('forward', 'reverse'), all at render resolution."""
    ct = tifffile.imread(d / 'ct_mid.tif')
    maps = {(m, f): tifffile.imread(d / f'{m}_s6{suf}.tif') for m in MODELS for f, suf in (('forward', ''), ('reverse', '_reverse'))}
    h = min([ct.shape[0]] + [a.shape[0] for a in maps.values()]); w = min([ct.shape[1]] + [a.shape[1] for a in maps.values()])
    ct = ct[:h, :w]
    _, valid, good, m = sheet_masks(ct)
    use = m & ~crack_mask(ct, m)
    out = {}
    for f in ('forward', 'reverse'):
        r = [rank_map(maps[(mm, f)][:h, :w], use) for mm in MODELS]
        out[f] = np.minimum(r[0], r[1])
    return ct, use, out


# ----------------------------------------------------------------------------------------------- vote
LAYERS = range(-3, 4)   # windings kept: 0 = the winding with most tiles, +-1 its neighbours (+ = front side), ...
TILE_NODES = 5          # tile = 5 x 5 mesh nodes (100 voxels, ~0.9 mm)
MAX_CELLS = 12_000_000  # grid cells per (layer, face); the cell size grows when the area needs it


def make_tiles(nodes, tile=TILE_NODES):
    """Cut every patch into tiles: centroid, unit normal as the mesh gives it, patch id; plus per-patch tile-id grids."""
    cent, nrm, pid, grids = [], [], [], []
    for k, (pts, valid, scale, nn, ok) in enumerate(nodes):
        H, W = ok.shape; th, tw = -(-H // tile), -(-W // tile)
        g = np.full((th, tw), -1, np.int64)
        for i in range(th):
            for j in range(tw):
                m = ok[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]
                if m.sum() < tile * tile // 2:
                    continue
                P = pts[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile][m].astype(np.float64)
                N = nn[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile][m].astype(np.float64)
                n = N.mean(0); n /= np.linalg.norm(n) + 1e-12
                g[i, j] = len(cent); cent.append(P.mean(0)); nrm.append(n); pid.append(k)
        grids.append(g)
    return {'c': np.array(cent).reshape(-1, 3), 'n': np.array(nrm).reshape(-1, 3), 'p': np.array(pid, int)}, grids


def tile_pairs(T, radius=None, lateral=None):
    """tiles of different patches lying on top of each other (centroids < 0.6 tile apart sideways), with roughly
    parallel normals (either sign). Returns pairs [n, 2] and the dot of their raw normals. Neighbouring tiles of the
    same patch are not paired: they sit a whole tile apart sideways, so a small tilt of their normals would read as
    several voxels of offset (and a patch that jumps sheets would be glued across its jump)."""
    from scipy.spatial import cKDTree
    w = TILE_NODES * 20.0
    lateral = 0.6 * w if lateral is None else lateral
    radius = lateral + 30.0 if radius is None else radius
    pairs = cKDTree(T['c']).query_pairs(radius, output_type='ndarray') if len(T['c']) > 1 else np.zeros((0, 2), int)
    if len(pairs) == 0:
        return pairs, np.zeros(0)
    pairs = pairs[T['p'][pairs[:, 0]] != T['p'][pairs[:, 1]]]
    a, b = pairs[:, 0], pairs[:, 1]
    dot = (T['n'][a] * T['n'][b]).sum(1)
    n = T['n'][a] + np.sign(dot)[:, None] * T['n'][b]; n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    d = T['c'][b] - T['c'][a]
    o = (d * n).sum(1)
    lat = np.linalg.norm(d - o[:, None] * n, axis=1)
    k = (lat < lateral) & (np.abs(dot) > 0.7)
    return pairs[k], dot[k]


def intra_pairs(nodes, grids, T, tile=TILE_NODES):
    """neighbouring tiles of the same patch and their offset (b - a) along the tiles' mean normal (raw orientation),
    measured between the node rows/columns on either side of the shared tile border (20 voxels apart): a patch that
    jumps sheets at a border shows the full sheet spacing there."""
    pa, pb, off = [], [], []
    for k, (pts, valid, scale, nn, ok) in enumerate(nodes):
        g = grids[k]
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                a = g[i, j]
                if a < 0:
                    continue
                for di, dj in ((0, 1), (1, 0)):
                    if i + di >= g.shape[0] or j + dj >= g.shape[1] or g[i + di, j + dj] < 0:
                        continue
                    b = g[i + di, j + dj]
                    if dj:
                        r = slice(i * tile, (i + 1) * tile); ca, cb = (j + 1) * tile - 1, (j + 1) * tile
                        A, B, m = pts[r, ca], pts[r, cb], ok[r, ca] & ok[r, cb]
                    else:
                        c = slice(j * tile, (j + 1) * tile); ra, rb = (i + 1) * tile - 1, (i + 1) * tile
                        A, B, m = pts[ra, c], pts[rb, c], ok[ra, c] & ok[rb, c]
                    if m.sum() < 2:
                        continue
                    n = T['n'][a] + T['n'][b]; n = n / (np.linalg.norm(n) + 1e-12)
                    pa.append(a); pb.append(b); off.append(float(np.median((B[m] - A[m]).astype(np.float64) @ n)))
    return np.array([pa, pb], int).T.reshape(-1, 2), np.array(off)


def orient_patches(T, pairs, dot, n_patches):
    """+-1 per patch so that overlapping patches' normals agree: breadth-first over the patch overlap graph from patch 0
    (each edge's sign = majority sign of its tile pairs). Patches not connected to patch 0 keep +1."""
    a, b = T['p'][pairs[:, 0]], T['p'][pairs[:, 1]]
    k = a != b
    votes = {}
    for x, y, s in zip(a[k].tolist(), b[k].tolist(), np.sign(dot[k]).tolist()):
        key = (min(x, y), max(x, y)); votes[key] = votes.get(key, 0) + s
    nbr = [[] for _ in range(n_patches)]
    for (x, y), v in votes.items():
        if v != 0:
            nbr[x].append((y, 1 if v > 0 else -1, abs(v))); nbr[y].append((x, 1 if v > 0 else -1, abs(v)))
    o = np.zeros(n_patches, int)
    from collections import deque
    for s0 in range(n_patches):
        if o[s0]:
            continue
        o[s0] = 1; q = deque([s0])
        while q:
            x = q.popleft()
            for y, sg, _ in sorted(nbr[x], key=lambda e: -e[2]):     # strongest overlaps first
                if not o[y]:
                    o[y] = o[x] * sg; q.append(y)
    return o


def tile_offsets(T, pairs):
    """signed offset of tile b from tile a along their shared (already oriented) normal, voxels."""
    a, b = pairs[:, 0], pairs[:, 1]
    n = T['n'][a] + T['n'][b]; n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return ((T['c'][b] - T['c'][a]) * n).sum(1)


def sheet_spacing(o, lo=9.0, hi=24.0, default=14.0):
    """distance between neighbouring windings: peak of |offset| between lo and hi voxels (PHerc. 1545: ~13-15)."""
    x = np.abs(o); x = x[(x > lo) & (x < hi)]
    if len(x) < 200:
        return default
    h, e = np.histogram(x, bins=int((hi - lo) / 1.5), range=(lo, hi))
    h = np.convolve(h, [1, 2, 3, 2, 1], 'same')
    return float((e[np.argmax(h)] + e[np.argmax(h) + 1]) / 2)


def tile_layers(T, pairs, o, spacing, tol=None, iters=4, keep=None):
    """Number the windings: an edge says 'b is round(o / spacing) windings to the front of a' when o is within tol of
    a multiple of the spacing (else no edge). Breadth-first labelling from the best-connected tile, then a few rounds of
    majority vote over each tile's neighbours. Tiles outside the kept connected part get -999: the part holding most of
    the ``keep`` tiles (indices) when given, else the biggest part."""
    tol = spacing / 3 if tol is None else tol
    dl = np.rint(o / spacing).astype(int)
    ok = np.abs(o - dl * spacing) < tol
    a, b, dl = pairs[ok, 0], pairs[ok, 1], dl[ok]
    n = len(T['c'])
    nbr = [[] for _ in range(n)]
    for x, y, d in zip(a.tolist(), b.tolist(), dl.tolist()):
        nbr[x].append((y, d)); nbr[y].append((x, -d))
    lay = np.full(n, -999, int)
    from collections import deque
    comp = np.full(n, -1, int); sizes = []
    for s in sorted(range(n), key=lambda t: -sum(1 for _, d in nbr[t] if d == 0)):
        if lay[s] != -999:
            continue
        lay[s] = 0; comp[s] = len(sizes); q = deque([s]); size = 1
        while q:
            t = q.popleft()
            for u, d in sorted(nbr[t], key=lambda e: abs(e[1])):
                if lay[u] == -999:
                    lay[u] = lay[t] + d; comp[u] = comp[s]; q.append(u); size += 1
        sizes.append(size)
    for _ in range(iters):
        new = lay.copy()
        for t in range(n):
            v = [lay[u] - d for u, d in nbr[t]]
            if v:
                vals, cnt = np.unique(v, return_counts=True)
                if cnt.max() > (np.array(v) == lay[t]).sum():
                    new[t] = vals[np.argmax(cnt)]
        lay = new
    if sizes:
        kc = comp[keep] if keep is not None and len(keep) else np.zeros(0, int)
        kc = kc[kc >= 0]
        best = int(np.bincount(kc).argmax()) if len(kc) else int(np.argmax(sizes))
        lay[comp != best] = -999
    return lay


class Grid:
    """Flat grid on a plane (origin p0, in-plane axes e1, e2): cell = ``grid`` voxels. One set of sums per (layer, face)."""

    def __init__(self, p0, e1, e2, umin, umax, vmin, vmax, grid):
        self.p0, self.e1, self.e2, self.g = p0, e1, e2, grid
        self.umin, self.vmin = umin, vmin
        self.nu = int(math.ceil((umax - umin) / grid)) + 1
        self.nv = int(math.ceil((vmax - vmin) / grid)) + 1
        self.acc = {}

    def index(self, xyz):
        d = xyz - self.p0
        u = np.floor(((d @ self.e1) - self.umin) / self.g).astype(np.int64)
        v = np.floor(((d @ self.e2) - self.vmin) / self.g).astype(np.int64)
        ok = (u >= 0) & (u < self.nu) & (v >= 0) & (v < self.nv)
        return v * self.nu + u, ok

    def _get(self, layer, face):
        k = (layer, face)
        if k not in self.acc:
            n = self.nu * self.nv
            self.acc[k] = {'cov': np.zeros(n, np.int16), 'sum': np.zeros(n, np.float32), 'top': np.zeros(n, np.float32)}
        return self.acc[k]

    def add(self, layer, face, idx, val):
        """one patch: per-cell mean of val -> +1 coverage, + mean, + share of the cell's samples in the top band."""
        if len(idx) == 0:
            return
        A = self._get(layer, face)
        uq, inv = np.unique(idx, return_inverse=True)
        cnt = np.bincount(inv).astype(np.float64)
        A['cov'][uq] += 1
        A['sum'][uq] += (np.bincount(inv, weights=val) / cnt).astype(np.float32)
        A['top'][uq] += (np.bincount(inv, weights=(val >= TOP).astype(np.float64)) / cnt).astype(np.float32)

    def image(self, a):
        return a.reshape(self.nv, self.nu)


def winding_numbers(nodes, anchor=None):
    """Tiles, patch orientation (+-1), winding number per tile (-999 = not connected) and sheet spacing for a set of
    patch meshes (``mesh_nodes`` tuples). Winding 0 is the main winding of patch ``anchor`` (index), or the winding with
    the most tiles when anchor is None. Returns (T, tile grids, o_p, lay, spacing, {winding: tiles})."""
    T, tgrids = make_tiles(nodes)
    pairs, dot = tile_pairs(T)
    ipairs, ioff = intra_pairs(nodes, tgrids, T)
    o_p = orient_patches(T, pairs, dot, len(nodes))
    T['n'] = T['n'] * o_p[T['p']][:, None]                       # all tile normals now point to the same side
    pairs = pairs[(T['n'][pairs[:, 0]] * T['n'][pairs[:, 1]]).sum(1) > 0.7]
    off = tile_offsets(T, pairs)
    spacing = sheet_spacing(off)
    if len(ipairs):
        pairs = np.concatenate([pairs, ipairs]); off = np.concatenate([off, ioff * o_p[T['p'][ipairs[:, 0]]]])
    lay = tile_layers(T, pairs, off, spacing, keep=np.flatnonzero(T['p'] == anchor) if anchor is not None else None)
    ref = lay[(T['p'] == anchor) & (lay != -999)] if anchor is not None else np.zeros(0, int)
    if len(ref) == 0:
        ref = lay[lay != -999]
    vals, cnt = np.unique(ref, return_counts=True)
    zero = int(vals[np.argmax(cnt)]) if len(vals) else 0
    lay = np.where(lay == -999, -999, lay - zero)
    vals, cnt = np.unique(lay[lay != -999], return_counts=True)
    return T, tgrids, o_p, lay, spacing, {int(v): int(c) for v, c in zip(vals, cnt)}


def plan(patches, grid):
    """tiles, patch orientation, winding numbers and the flat grid for a set of patch dirs."""
    nodes = [mesh_nodes(Path(p) / 'mesh') for p in patches]
    T, tgrids, o_p, lay, spacing, counts = winding_numbers(nodes)
    # plane of the main winding
    m = lay == 0 if (lay == 0).any() else np.ones(len(lay), bool)
    p0 = T['c'][m].mean(0)
    e3 = T['n'][m].mean(0); e3 /= np.linalg.norm(e3)
    X = T['c'][m] - p0; X = X - (X @ e3)[:, None] * e3
    e1 = np.linalg.svd(X, full_matrices=False)[2][0]; e1 -= (e1 @ e3) * e3; e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    k = lay != -999
    U = (T['c'][k] - p0) @ e1; V = (T['c'][k] - p0) @ e2
    pad = TILE_NODES * 20
    umin, umax, vmin, vmax = U.min() - pad, U.max() + pad, V.min() - pad, V.max() + pad
    while (umax - umin) * (vmax - vmin) / grid ** 2 > MAX_CELLS:
        grid += 1
    G = Grid(p0, e1, e2, umin, umax, vmin, vmax, grid)
    G.e3 = e3
    return nodes, T, tgrids, o_p, lay, spacing, G, counts


def vote(out: Path, patches, grid=3.0, stride=2, log=print, min_facing=0.3):
    out.mkdir(parents=True, exist_ok=True)
    nodes, T, tgrids, o_p, lay, spacing, G, layer_counts = plan(patches, grid)
    log(f'tiles {len(lay)}, winding spacing {spacing:.1f} vox, grid {G.g} vox ({G.nu} x {G.nv}), layers {layer_counts}')
    info = []
    for k, (p, (pts, valid, scale, nrm, ok)) in enumerate(zip(patches, nodes)):
        p = Path(p)
        try:
            ct, use, ranks = patch_scores(p)
        except Exception as e:  # noqa
            log('skip', p.name, e); continue
        face_of = {'forward': 'front' if o_p[k] > 0 else 'back', 'reverse': 'back' if o_p[k] > 0 else 'front'}
        g = tgrids[k]
        tl = np.where(g >= 0, lay[np.maximum(g, 0)], -999)                     # layer per tile of this patch
        facing = np.where(g >= 0, T['n'][np.maximum(g, 0)] @ G.e3, 0)          # tiles tilted away from the plane
        tl = np.where(facing >= min_facing, tl, -999)
        H, W = ct.shape
        rows = np.arange(0, H, stride); cols = np.arange(0, W, stride)
        band = max(1, 400 // stride)
        counts = {}
        for r0 in range(0, len(rows), band):
            rr = rows[r0:r0 + band]
            xyz, ok_px = pixel_xyz(pts, valid, scale, rr, cols)
            ti = np.minimum(((rr[:, None] + 0.5) * scale).astype(int) // TILE_NODES, tl.shape[0] - 1)
            tj = np.minimum(((cols[None, :] + 0.5) * scale).astype(int) // TILE_NODES, tl.shape[1] - 1)
            lp = tl[ti, tj]
            u = use[np.ix_(rr, cols)] & ok_px & (lp != -999)
            if not u.any():
                continue
            idx, inside = G.index(xyz[u].astype(np.float64))
            lpu = lp[u][inside]; idx = idx[inside]
            vals_ = {f: a[np.ix_(rr, cols)][u][inside].astype(np.float64) for f, a in ranks.items()}
            for l in np.unique(lpu):
                s_ = lpu == l
                counts[int(l)] = counts.get(int(l), 0) + int(s_.sum())
                if int(l) not in LAYERS:
                    continue
                for f, v in vals_.items():
                    G.add(int(l), face_of[f], idx[s_], v[s_])
        tot = max(sum(counts.values()), 1)
        info.append({'patch': p.name, 'orientation': int(o_p[k]), 'use_fraction': round(float(use.mean()), 3),
                     'layers': {str(l): round(n / tot, 3) for l, n in sorted(counts.items())}})
        log('voted', p.name, f'use {use.mean():.2f}', 'layers', {l: round(n / tot, 2) for l, n in sorted(counts.items())})
    np.savez_compressed(out / 'vote.npz', nu=G.nu, nv=G.nv, grid=G.g, p0=G.p0, e1=G.e1, e2=G.e2, umin=G.umin, vmin=G.vmin,
                        **{f'{q}_L{l:+d}_{f}': A[q] for (l, f), A in G.acc.items() for q in A})
    summary = summarise(G)
    summary.update({'winding_spacing_vox': round(spacing, 1), 'tiles_per_layer': layer_counts, 'patches': len(info),
                    'grid_vox': G.g, 'per_patch': info})
    (out / 'vote.json').write_text(json.dumps(summary, indent=1))
    draw(G, out)
    return summary


def summarise(G, min_cov=3):
    res = {}
    cm2 = (G.g * 9.362e-4) ** 2
    for (l, f), A in sorted(G.acc.items()):
        cov = A['cov']; k = cov >= min_cov
        res[f'L{l:+d}_{f}'] = {'area_cm2': round(float((cov >= 1).sum() * cm2), 2), f'area_cm2_cov>={min_cov}': round(float(k.sum() * cm2), 2),
                               'mean_rank': round(float((A['sum'][k] / cov[k]).mean()), 4) if k.any() else None,
                               'share_top_in_half': round(float((A['top'][k] / cov[k] >= 0.5).mean()), 4) if k.any() else None}
    return res


def vote_images(G, layer, face):
    A = G._get(layer, face)
    cov = G.image(A['cov']).astype(np.float32)
    mean = np.where(cov > 0, G.image(A['sum']) / np.maximum(cov, 1), np.nan)
    top = np.where(cov > 0, G.image(A['top']) / np.maximum(cov, 1), np.nan)
    return cov, mean, top


def draw(G, out: Path, min_cov=3):
    """per (layer, face): vote_mean (mean agreed rank where >= min_cov patches; 1/3 = chance for two independent ranks),
    vote_top (share of covering patches with both models in their top 3 %), coverage. Images are cropped to coverage."""
    from PIL import Image
    for (l, f) in sorted(G.acc):
        cov, mean, top = vote_images(G, l, f)
        if (cov >= min_cov).sum() < 1000:
            continue
        ys, xs = np.nonzero(cov > 0); sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        cov, mean, top = cov[sl], mean[sl], top[sl]
        k = cov >= min_cov
        for name, a, lo, hi in (('mean', mean, 1 / 3, 0.8), ('top', top, 0.0, 0.5)):
            g = np.clip((np.nan_to_num(a) - lo) / (hi - lo), 0, 1)
            img = np.zeros(cov.shape + (3,), np.uint8); img[cov > 0] = (40, 40, 40)
            img[k] = np.stack([g[k] * 255, g[k] * 220, np.full(k.sum(), 60)], -1).astype(np.uint8)
            Image.fromarray(img[::-1]).save(out / f'vote_{name}_L{l:+d}_{f}.png')
        Image.fromarray((np.clip(cov / 8, 0, 1)[::-1] * 255).astype(np.uint8)).save(out / f'coverage_L{l:+d}_{f}.png')


# ----------------------------------------------------------------------------------------------- patch selection
def select_patches(stitch: Path, cover=3, min_new=0.2, log=print):
    """Greedy: biggest patches first, take a patch when >= min_new of its 20-voxel cells are still under ``cover``.
    The chosen patches are returned nearest-to-the-seed first, so the area around the seed is covered first."""
    rep = json.loads((stitch / 'same_report.json').read_text())
    names = sorted(n for n, s in rep['state'].items() if s == 'accepted' and (stitch / 'paths' / n / 'z.tif').exists())
    cells, cent = {}, {}
    for n in names:
        pts, valid, _ = load_mesh(stitch / 'paths' / n)
        q = np.unique(np.floor(pts[valid] / 20).astype(np.int64), axis=0)
        cells[n] = [tuple(k) for k in q.tolist()]
        cent[n] = pts[valid].mean(0)
    count, chosen = {}, []
    for n in sorted(names, key=lambda n: (-len(cells[n]), n)):
        new = sum(count.get(k, 0) < cover for k in cells[n])
        if new >= min_new * len(cells[n]):
            chosen.append(n)
            for k in cells[n]:
                count[k] = count.get(k, 0) + 1
    seed = rep['seed'] if rep['seed'] in cent else chosen[0]
    chosen.sort(key=lambda n: (float(np.linalg.norm(cent[n] - cent[seed])), n))
    cov = np.array(list(count.values()))
    log(f'{len(chosen)} of {len(names)} patches; 20-voxel cells {len(cov)}, covered >= {cover}: {(cov >= cover).mean():.2f}')
    return chosen


def queue_surveys(stitch: Path, survey: Path, ct_url: str, tag: str, cover=3, limit=None, log=print):
    """Write one survey dir per chosen patch (mesh, ct.txt, size.txt): the laptop worker then renders 28 layers and runs
    seed42 + dense_native on each, as for every survey patch. Dir names sort in the chosen order."""
    chosen = select_patches(stitch, cover, log=log)[:limit]
    for i, n in enumerate(chosen):
        d = survey / f'{tag}_{i:04d}_{n}'
        if (d / 'size.txt').exists():
            continue
        (d / 'mesh').mkdir(parents=True, exist_ok=True)
        for f in ('x.tif', 'y.tif', 'z.tif', 'meta.json'):
            shutil.copy2(stitch / 'paths' / n / f, d / 'mesh' / f)
        x = tifffile.imread(d / 'mesh' / 'x.tif')
        (d / 'ct.txt').write_text(ct_url + '\n')
        (d / 'size.txt').write_text(f'{x.shape[0] * 20} {x.shape[1] * 20}\n')
    log(f'queued {len(chosen)} survey dirs {tag}_0000.. in {survey}')
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('vote'); v.add_argument('out', type=Path); v.add_argument('patches', nargs='+', type=Path)
    v.add_argument('--grid', type=float, default=3.0); v.add_argument('--stride', type=int, default=2)
    q = sub.add_parser('queue'); q.add_argument('stitch', type=Path); q.add_argument('survey', type=Path); q.add_argument('ct')
    q.add_argument('--tag', default='V'); q.add_argument('--cover', type=int, default=3); q.add_argument('--limit', type=int)
    a = ap.parse_args(argv)
    if a.cmd == 'vote':
        import glob
        pats = []
        for p in a.patches:                       # a quoted pattern (work/survey/V1545_*) is expanded here
            pats += sorted(Path(x) for x in glob.glob(str(p))) if any(ch in str(p) for ch in '*?[') else [p]
        pats = [p for p in pats if (p / 'finished.txt').exists() or (p / 'seed42_s6_reverse.tif').exists() and (p / 'ct_mid.tif').exists()]
        print(json.dumps(vote(a.out, pats, a.grid, a.stride), indent=1)[:2000])
    else:
        queue_surveys(a.stitch, a.survey, a.ct, a.tag, a.cover, a.limit)


if __name__ == '__main__':
    main()
