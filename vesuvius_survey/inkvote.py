"""inkvote.py - keep only ink that repeats where independently traced patches overlap.

Many patches on one winding cover the same papyrus. Real ink sits at the same 3D spot in every patch that covers it;
crack hits and model noise move around from patch to patch. For every patch this module

  1. masks cracks: off-sheet pixels (report card sheet check) and the darkest DARK_Q of on-sheet CT, dilated,
  2. turns seed42 and dense_native into within-patch percentile ranks and keeps the lower of the two (both must agree),
  3. maps every pixel to 3D with the mesh and then to an unrolled (around-the-scroll, height) grid of the winding,

and averages per grid cell over the patches covering it. The two faces of the sheet are kept apart: each patch's
forward/reverse map is assigned to the scroll's outer or inner face using its normals and the scroll centre.

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
def scroll_centre(pts_list, nrm_list):
    """Least-squares point (x, y) closest to all normal lines in the xy plane (sheet normals point radially)."""
    A = np.zeros((2, 2)); b = np.zeros(2)
    for P, N in zip(pts_list, nrm_list):
        n = N[:, :2].astype(np.float64)
        ln = np.linalg.norm(n, axis=1)
        k = ln > 0.5                      # skip nodes whose normal is mostly along z
        n = n[k] / ln[k, None]; p = P[k, :2].astype(np.float64)
        M = np.eye(2)[None] - n[:, :, None] * n[:, None, :]
        A += M.sum(0); b += np.einsum('nij,nj->i', M, p)
    return np.linalg.solve(A, b)


def mesh_nodes(mesh_dir: Path):
    pts, valid, scale = load_mesh(mesh_dir)
    nrm = node_normals(pts, valid)
    ok = valid & np.isfinite(nrm).all(-1)
    return pts, valid, scale, nrm, ok


def orientation(pts, nrm, ok, c) -> int:
    """+1 when the mesh normal points away from the scroll centre (outward), else -1."""
    d = pts[ok][:, :2] - c
    return 1 if np.median((nrm[ok][:, :2] * d).sum(1)) > 0 else -1


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
LAYERS = range(-3, 4)   # windings kept: 0 = the best-covered winding, +-1 its neighbours, ...


class Grid:
    """Unrolled grid of the scroll: u = angle around the centre x mean radius, v = height (z), ``grid`` voxels per cell.
    One set of sums per (winding layer, face)."""

    def __init__(self, c, theta0, radius, z0, z1, grid):
        self.c, self.theta0, self.R, self.g = np.asarray(c, np.float64), theta0, radius, grid
        self.z0 = z0
        self.nu = int(math.ceil(2 * math.pi * radius / grid)) + 1
        self.nv = int(math.ceil((z1 - z0) / grid)) + 1
        self.acc = {}

    def index(self, xyz):
        dx = xyz[:, 0] - self.c[0]; dy = xyz[:, 1] - self.c[1]
        th = (np.arctan2(dy, dx) - self.theta0) % (2 * np.pi)
        u = np.floor(th * self.R / self.g).astype(np.int64)
        v = np.floor((xyz[:, 2] - self.z0) / self.g).astype(np.int64)
        ok = (u >= 0) & (u < self.nu) & (v >= 0) & (v < self.nv)
        return v * self.nu + u, ok, np.hypot(dx, dy)

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


def plan_grid(patches, grid):
    nodes = [mesh_nodes(Path(p) / 'mesh') for p in patches]
    c = scroll_centre([n[0][n[4]] for n in nodes], [n[3][n[4]] for n in nodes])
    th = np.concatenate([np.arctan2(n[0][n[4]][:, 1] - c[1], n[0][n[4]][:, 0] - c[0]) for n in nodes])
    r = np.concatenate([np.hypot(n[0][n[4]][:, 0] - c[0], n[0][n[4]][:, 1] - c[1]) for n in nodes])
    z = np.concatenate([n[0][n[4]][:, 2] for n in nodes])
    # cut the circle at the middle of the largest empty angular gap so the covered part is not split
    hist, edges = np.histogram(th, bins=360, range=(-np.pi, np.pi))
    empty = np.flatnonzero(hist == 0)
    theta0 = float(edges[0])
    if len(empty):
        e2 = np.concatenate([empty, empty + 360]); best, run, bs, start = 1, 1, e2[0], e2[0]
        for i in range(1, len(e2)):
            run, start = (run + 1, start) if e2[i] == e2[i - 1] + 1 else (1, e2[i])
            if run > best:
                best, bs = run, start
        theta0 = float(edges[(bs + best // 2) % 360])
    tt = (th - theta0) % (2 * np.pi)
    return nodes, c, theta0, float(np.median(r)), float(z.min()), float(z.max()), float(tt.max() - tt.min())


TILE_NODES = 5      # tile = 5 x 5 mesh nodes (100 voxels, ~0.9 mm)


def make_tiles(nodes, c, tile=TILE_NODES):
    """Cut every patch into tiles; per tile: centroid and outward normal (away from the scroll centre).
    Returns (tile table dict of arrays, per-patch tile-id grids)."""
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
                ctr = P.mean(0)
                if np.dot(n[:2], ctr[:2] - c) < 0:
                    n = -n
                g[i, j] = len(cent); cent.append(ctr); nrm.append(n); pid.append(k)
        grids.append(g)
    return {'c': np.array(cent).reshape(-1, 3), 'n': np.array(nrm).reshape(-1, 3), 'p': np.array(pid)}, grids


def tile_edges(T, radius=None, lateral=None):
    """tile pairs and their signed offset along the shared (outward) normal, voxels. Pairs: tiles of different patches
    lying on top of each other (lateral distance < 0.6 tile) and neighbouring tiles of the same patch."""
    from scipy.spatial import cKDTree
    w = TILE_NODES * 20.0
    radius = 1.15 * w if radius is None else radius
    lateral = 0.6 * w if lateral is None else lateral
    pairs = cKDTree(T['c']).query_pairs(radius, output_type='ndarray')
    if len(pairs) == 0:
        return pairs, np.zeros(0)
    a, b = pairs[:, 0], pairs[:, 1]
    n = T['n'][a] + T['n'][b]; n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    d = T['c'][b] - T['c'][a]
    o = (d * n).sum(1)
    lat = np.linalg.norm(d - o[:, None] * n, axis=1)
    same = T['p'][a] == T['p'][b]
    k = np.where(same, lat < 1.15 * w, lat < lateral) & ((T['n'][a] * T['n'][b]).sum(1) > 0.7)
    return pairs[k], o[k]


def sheet_spacing(o, lo=9.0, hi=24.0, default=14.0):
    """distance between neighbouring windings: peak of |offset| between lo and hi voxels (PHerc. 1545: ~15)."""
    x = np.abs(o); x = x[(x > lo) & (x < hi)]
    if len(x) < 200:
        return default
    h, e = np.histogram(x, bins=int((hi - lo) / 1.5), range=(lo, hi))
    h = np.convolve(h, [1, 2, 3, 2, 1], 'same')
    return float((e[np.argmax(h)] + e[np.argmax(h) + 1]) / 2)


def tile_layers(T, pairs, o, spacing, tol=None, iters=4):
    """Number the windings: an edge says 'b is round(o / spacing) windings outward of a' when o is within tol of a
    multiple of the spacing (else no edge). Breadth-first labelling from the best-connected tile, then a few rounds
    of majority vote over each tile's neighbours. Returns layer per tile (int, or -999 when unconnected)."""
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
    order = sorted(range(n), key=lambda t: -sum(1 for _, d in nbr[t] if d == 0))
    comp = 0
    for s in order:                                     # every connected component, biggest-degree tile first
        if lay[s] != -999 or not nbr[s]:
            continue
        lay[s] = 0; q = deque([s]); comp += 1
        while q:
            t = q.popleft()
            for u, d in sorted(nbr[t], key=lambda e: abs(e[1])):
                if lay[u] == -999:
                    lay[u] = lay[t] + d; q.append(u)
        if comp == 1:
            first = lay != -999
    for _ in range(iters):
        new = lay.copy()
        for t in range(n):
            if lay[t] == -999 or not nbr[t]:
                continue
            v = [lay[u] - d for u, d in nbr[t] if lay[u] != -999]
            if v:
                vals, cnt = np.unique(v, return_counts=True)
                if cnt.max() > (np.array(v) == lay[t]).sum():
                    new[t] = vals[np.argmax(cnt)]
        lay = new
    # keep only the biggest component's numbering (others have an arbitrary zero)
    if comp > 1:
        lay[~first] = -999
    return lay


def vote(out: Path, patches, grid=3.0, stride=2, log=print):
    out.mkdir(parents=True, exist_ok=True)
    nodes, c, theta0, R, z0, z1, span = plan_grid(patches, grid)
    log(f'centre {c.round(1).tolist()} radius {R:.0f} vox, angular span {math.degrees(span):.0f} deg, z {z0:.0f}-{z1:.0f}')
    G = Grid(c, theta0, R, z0, z1, grid)
    T, tgrids = make_tiles(nodes, c)
    pairs, o = tile_edges(T)
    spacing = sheet_spacing(o)
    lay = tile_layers(T, pairs, o, spacing)
    # layer 0 = the layer with the most tiles
    vals, cnt = np.unique(lay[lay != -999], return_counts=True)
    zero = int(vals[np.argmax(cnt)]) if len(vals) else 0
    lay = np.where(lay == -999, -999, lay - zero)
    log(f'tiles {len(lay)}, edges {len(pairs)}, winding spacing {spacing:.1f} vox, layers ' +
        str({int(v - zero): int(n) for v, n in zip(vals, cnt)}))
    info = []
    for k, (p, (pts, valid, scale, nrm, ok)) in enumerate(zip(patches, nodes)):
        p = Path(p)
        try:
            ct, use, ranks = patch_scores(p)
        except Exception as e:  # noqa
            log('skip', p.name, e); continue
        o_ = orientation(pts, nrm, ok, c)
        face_of = {'forward': 'outer' if o_ > 0 else 'inner', 'reverse': 'inner' if o_ > 0 else 'outer'}
        tl = np.where(tgrids[k] >= 0, lay[np.maximum(tgrids[k], 0)], -999)      # layer per tile of this patch
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
            idx, inside, rad = G.index(xyz[u].astype(np.float64))
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
        main = max(counts, key=counts.get) if counts else None
        info.append({'patch': p.name, 'orientation': o_, 'use_fraction': round(float(use.mean()), 3),
                     'layers': {str(l): round(n / tot, 3) for l, n in sorted(counts.items())}})
        log('voted', p.name, 'outward' if o_ > 0 else 'inward', f'use {use.mean():.2f}',
            'layers', {l: round(n / tot, 2) for l, n in sorted(counts.items())})
    np.savez_compressed(out / 'vote.npz', nu=G.nu, nv=G.nv, grid=grid, centre=c, theta0=theta0, radius=R, z0=z0,
                        **{f'{q}_L{l:+d}_{f}': A[q] for (l, f), A in G.acc.items() for q in A})
    summary = summarise(G)
    summary.update({'winding_spacing_vox': round(spacing, 1), 'patches': len(info), 'centre': c.round(1).tolist(), 'radius_vox': round(R), 'grid_vox': grid,
                    'angular_span_deg': round(math.degrees(span)), 'z': [round(z0), round(z1)], 'per_patch': info})
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
