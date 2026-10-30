"""inkvote.py - keep only ink that repeats where independently traced patches overlap.

Many patches cover the same papyrus. Real ink sits at the same 3D spot in every patch that covers it; crack hits and
model noise move around from patch to patch. For every patch this module

  1. masks cracks: off-sheet pixels (report card sheet check) and the darkest DARK_Q of on-sheet CT, dilated,
  2. turns seed42 and dense_native into within-patch percentile ranks and keeps the lower of the two (both must agree),
  3. maps the pixels to 3D with the mesh and counts, per 6-voxel cell, how many patches cover it and how many of them
     have top ink there. Patches on different windings (>= ~6 voxels apart) do not share cells, so no winding sorting
     is needed; on PHerc. 1545 (crushed flat, windings touching) sorting patches into windings did not hold up.

The two faces of the sheet are kept apart (patch normals are made to agree through their overlaps). Control: the same
vote with every ink map turned 180 degrees on its own patch. Output: vote3d.json and, for the patches with the most
agreed cells (and any --show surfaces), the CT with the other patches' agreement drawn on it.

  python -m vesuvius_survey.inkvote vote OUT_DIR PATCH_DIR [PATCH_DIR ...] [--stride 2] [--show SURFACE_DIR ...]
      PATCH_DIR holds mesh/{x,y,z.tif,meta.json}, ct_mid.tif, seed42_s6[_reverse].tif, dnative_s6[_reverse].tif;
      a quoted pattern (work/survey/V1545_*) is expanded
  python -m vesuvius_survey.inkvote queue STITCH_DIR SURVEY_DIR CT_URL [--tag V] [--cover 3]
      picks patches so most of the patch set is covered >= 3 times and writes survey dirs for the laptop worker
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
TILE_NODES = 5          # tile = 5 x 5 mesh nodes (100 voxels, ~0.9 mm)


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
    jumps sheets at a border shows the full offset there."""
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


def patch_pieces(n_tiles, ipairs, ioff, smooth=3.0):
    """Split patches into smooth pieces: neighbouring tiles of one patch whose border offset is under ``smooth`` voxels
    are joined (union-find). A patch traced on one sheet is one piece; one that jumps sheets splits at the jump."""
    parent = np.arange(n_tiles)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (x, y), o in zip(ipairs.tolist(), ioff.tolist()):
        if abs(o) < smooth:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
    roots = np.array([find(x) for x in range(n_tiles)])
    _, piece = np.unique(roots, return_inverse=True)
    return piece


def piece_pair_stats(piece, pairs, off, close=2.5):
    """Per pair of different pieces with overlapping tiles: (a, b, median offset b - a, tile pairs agreeing with the
    median within ``close`` voxels)."""
    A, B = piece[pairs[:, 0]], piece[pairs[:, 1]]
    sw = A > B
    A, B = np.where(sw, B, A), np.where(sw, A, B)
    o = np.where(sw, -off, off)
    k = A != B
    A, B, o = A[k], B[k], o[k]
    if len(A) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0, int)
    key = A.astype(np.int64) * (int(piece.max()) + 1) + B
    order = np.argsort(key, kind='stable'); key, o, A, B = key[order], o[order], A[order], B[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]); ends = np.r_[starts[1:], len(key)]
    med = np.array([np.median(o[s0:e0]) for s0, e0 in zip(starts.tolist(), ends.tolist())])
    agree = np.array([int((np.abs(o[s0:e0] - m) < close).sum()) for s0, e0, m in zip(starts.tolist(), ends.tolist(), med.tolist())])
    return A[starts], B[starts], med, agree


SAME_SHEET = 2.5    # voxels: overlapping pieces closer than this (median) are the same sheet
OTHER_SHEET = 7.0   # voxels: ... farther than this are different sheets (in between: no decision)


def sheet_groups(n_pieces, A, B, med, agree, min_agree=3):
    """Group pieces into sheets: join pairs that lie on top of each other (|median| < SAME_SHEET), strongest overlaps
    first, but never join two groups that are clearly apart (|median| > OTHER_SHEET) anywhere they overlap. Unlike a
    chain of small steps or integer winding numbers, this cannot creep across touching windings as long as the two
    windings overlap clearly somewhere. Returns group id per piece."""
    parent = list(range(n_pieces))
    apart = [dict() for _ in range(n_pieces)]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for x, y, m, w in zip(A.tolist(), B.tolist(), med.tolist(), agree.tolist()):
        if abs(m) > OTHER_SHEET and w >= min_agree:
            apart[x][y] = apart[x].get(y, 0) + w; apart[y][x] = apart[y].get(x, 0) + w
    order = sorted((i for i in range(len(A)) if abs(med[i]) < SAME_SHEET and agree[i] >= min_agree), key=lambda i: -agree[i])
    for i in order:
        rx, ry = find(int(A[i])), find(int(B[i]))
        if rx == ry:
            continue
        if any(find(z) == ry for z in apart[rx]) or any(find(z) == rx for z in apart[ry]):
            continue                                     # these two sheets are clearly apart somewhere
        if len(apart[rx]) < len(apart[ry]):
            rx, ry = ry, rx
        parent[ry] = rx
        for z, w in apart[ry].items():
            apart[rx][z] = apart[rx].get(z, 0) + w
        apart[ry] = {}
    return np.array([find(x) for x in range(n_pieces)])


def sheet_numbers(nodes, anchor=None):
    """Tiles, patch orientation (+-1), sheet id per tile and the sheets' tile counts for a set of patch meshes
    (``mesh_nodes`` tuples). Patches are split into smooth pieces, pieces grouped into sheets (``sheet_groups``).
    Sheet 0 is the sheet holding most of patch ``anchor`` (index), else the biggest; the rest are numbered by size.
    Returns (T, tile grids, o_p, sheet id per tile, {sheet: tiles})."""
    T, tgrids = make_tiles(nodes)
    pairs, dot = tile_pairs(T)
    ipairs, ioff = intra_pairs(nodes, tgrids, T)
    o_p = orient_patches(T, pairs, dot, len(nodes))
    T['n'] = T['n'] * o_p[T['p']][:, None]                       # all tile normals now point to the same side
    pairs = pairs[(T['n'][pairs[:, 0]] * T['n'][pairs[:, 1]]).sum(1) > 0.7]
    off = tile_offsets(T, pairs)
    ioff = ioff * (o_p[T['p'][ipairs[:, 0]]] if len(ipairs) else 1)
    piece = patch_pieces(len(T['c']), ipairs, ioff)
    n_pieces = int(piece.max()) + 1 if len(piece) else 0
    # overlaps between pieces of different patches, plus steps inside a patch (a jump to another sheet)
    allp = np.concatenate([pairs, ipairs]) if len(ipairs) else pairs
    allo = np.concatenate([off, ioff]) if len(ipairs) else off
    A, B, med, agree = piece_pair_stats(piece, allp, allo)
    grp = sheet_groups(n_pieces, A, B, med, agree, min_agree=2) if n_pieces else np.zeros(0, int)
    g_tile = grp[piece] if len(piece) else np.zeros(0, int)
    ids, cnt = np.unique(g_tile, return_counts=True)
    order = list(ids[np.argsort(-cnt, kind='stable')])
    if anchor is not None and (T['p'] == anchor).any():
        v, c = np.unique(g_tile[T['p'] == anchor], return_counts=True)
        a0 = v[np.argmax(c)]; order.remove(a0); order.insert(0, a0)
    rank = {g: i for i, g in enumerate(order)}
    sheet = np.array([rank[g] for g in g_tile.tolist()], int)
    T['piece'] = piece
    # sheets seen clearly apart from sheet 0 somewhere (a winding or more away)
    sp = np.full(n_pieces, -1, int); sp[piece] = sheet
    far = (np.abs(med) > OTHER_SHEET) & (agree >= 2)
    sa, sb = sp[A[far]], sp[B[far]]
    T['apart_from_0'] = set(sb[sa == 0].tolist()) | set(sa[sb == 0].tolist())
    v, c = np.unique(sheet, return_counts=True)
    return T, tgrids, o_p, sheet, {int(a): int(b) for a, b in zip(v, c)}


CELL = 6.0          # voxels: two patches cover the same spot when their samples fall in the same 3D cell
N_SHOW = 12         # patches drawn (most cells where the other patches agree)


def _keys(xyz, cell=CELL):
    q = np.floor(np.asarray(xyz, np.float64) / cell).astype(np.int64)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def _samples(d: Path, nodes_k, stride):
    """per patch: 3D positions of the sampled pixels, which are usable, agreed-rank maps per face, and the CT."""
    ct, use, ranks = patch_scores(d)
    pts, valid, scale, nrm, ok = nodes_k
    rows, cols = np.arange(0, ct.shape[0], stride), np.arange(0, ct.shape[1], stride)
    xyz, okp = pixel_xyz(pts, valid, scale, rows, cols)
    u = use[np.ix_(rows, cols)] & okp
    return ct, rows, cols, xyz, u, ranks


def vote(out: Path, patches, stride=2, show=(), log=print, cell=CELL):
    """3D vote. For every patch and face, per 3D cell: share of the cell's samples where both models are in their top
    3 %. Per cell over patches: coverage and summed share. A cell is 'agreed' when >= 3 patches cover it and on
    average >= half of their samples there are top. Control: the same with every patch's ink map turned 180 degrees on
    its own surface (same maps, same statistics, 3D alignment destroyed); real repeating ink beats the control.
    ``show``: extra surfaces (e.g. big stitched ones) that are coloured with the vote but do not vote."""
    out.mkdir(parents=True, exist_ok=True)
    allp = [Path(p) for p in patches] + [Path(p) for p in show]
    nodes = [mesh_nodes(p / 'mesh') for p in allp]
    T, tgrids = make_tiles(nodes)
    pairs, dot = tile_pairs(T)
    o_p = orient_patches(T, pairs, dot, len(nodes))
    combos = [(f, n) for f in ('front', 'back') for n in (False, True)]
    keys, pid = [], []
    tops = {c: [] for c in combos}
    info = []
    for k, p in enumerate(allp[:len(patches)]):
        try:
            ct, rows, cols, xyz, u, ranks = _samples(p, nodes[k], stride)
        except Exception as e:  # noqa
            log('skip', p.name, e); continue
        if not u.any():
            continue
        uk, inv = np.unique(_keys(xyz[u]), return_inverse=True)
        cnt = np.bincount(inv).astype(np.float64)
        keys.append(uk); pid.append(np.full(len(uk), k, np.int32))
        local = {'front': 'forward' if o_p[k] > 0 else 'reverse', 'back': 'reverse' if o_p[k] > 0 else 'forward'}
        for f, null in combos:
            a = ranks[local[f]]
            a = a[::-1, ::-1] if null else a
            v = a[np.ix_(rows, cols)][u]
            tops[(f, null)].append((np.bincount(inv, weights=(v >= TOP).astype(np.float64)) / cnt).astype(np.float32))
        info.append({'patch': p.name, 'orientation': int(o_p[k]), 'use_fraction': round(float(u.mean()), 3), 'cells': int(len(uk))})
        log('sampled', p.name, f'cells {len(uk)}')
    K = np.concatenate(keys); P = np.concatenate(pid)
    uq, inv = np.unique(K, return_inverse=True)
    cov = np.bincount(inv)
    res = {'patches': len(info), 'cell_vox': cell, 'cells': int(len(uq)),
           'cells_cov>=3': int((cov >= 3).sum()), 'area_cm2_cov>=3_approx': round(float((cov >= 3).sum() * (cell * 9.362e-4) ** 2), 2)}
    best = {}
    for f, null in combos:
        t = np.concatenate(tops[(f, null)]).astype(np.float64)
        ts = np.bincount(inv, weights=t, minlength=len(uq))
        k3 = cov >= 3
        agreed = k3 & (ts / np.maximum(cov, 1) >= 0.5)
        tag = f'{f}_control' if null else f
        res[tag] = {'agreed_cells': int(agreed.sum()), 'agreed_share': round(float(agreed.sum() / max(k3.sum(), 1)), 5),
                    'mean_top_share_cov>=3': round(float((ts[k3] / cov[k3]).mean()), 4) if k3.any() else None}
        if not null:
            best[f] = (ts, t)
    for f in ('front', 'back'):
        a, c = res[f]['agreed_cells'], res[f + '_control']['agreed_cells']
        res[f]['vs_control'] = round(a / max(c, 1), 2)
    # draw: per patch, consensus of the OTHER patches at each pixel's cell (best face = more agreed cells)
    face = max(('front', 'back'), key=lambda f: res[f]['agreed_cells'])
    ts, t_own = best[face]
    other_cov = cov[inv] - 1
    other_top = ts[inv] - t_own
    agree_e = (other_cov >= 2) & (other_top / np.maximum(other_cov, 1) >= 0.5) & (t_own >= 0.5)
    per_patch = np.bincount(P[agree_e], minlength=len(allp))
    res['face_drawn'] = face
    res['top_patches'] = [{'patch': allp[k].name, 'agreeing_cells': int(per_patch[k])} for k in np.argsort(-per_patch)[:N_SHOW] if per_patch[k] > 0]
    res['per_patch'] = info
    (out / 'vote3d.json').write_text(json.dumps(res, indent=1))
    lookup = (uq, cov, ts)
    for k in [int(i) for i in np.argsort(-per_patch)[:N_SHOW] if per_patch[i] > 0] + list(range(len(patches), len(allp))):
        try:
            draw_consensus(allp[k], nodes[k], o_p[k], face, lookup, out / f'consensus_{face}_{allp[k].name}.png', stride,
                           exclude_self=k < len(patches))
        except Exception as e:  # noqa
            log('draw failed', allp[k].name, e)
    log(json.dumps({k: v for k, v in res.items() if k not in ('per_patch', 'top_patches')}))
    return res


def draw_consensus(d: Path, nodes_k, o, face, lookup, png: Path, stride, exclude_self=True):
    """CT (grey) with this patch's own top ink (blue) and the other patches' agreement at each pixel's 3D cell
    (yellow -> red as the share of covering patches with top ink rises; only where >= 2 others cover)."""
    from PIL import Image
    uq, cov, ts = lookup
    ct, rows, cols, xyz, u, ranks = _samples(d, nodes_k, stride)
    local = 'forward' if (o > 0) == (face == 'front') else 'reverse'
    own = ranks[local][np.ix_(rows, cols)] >= TOP
    kk = _keys(xyz.reshape(-1, 3))
    j = np.clip(np.searchsorted(uq, kk), 0, len(uq) - 1); hit = (uq[j] == kk) & u.reshape(-1)
    c = np.where(hit, cov[j], 0).astype(np.float64); s = np.where(hit, ts[j], 0.0)
    if exclude_self:
        # remove this patch's own contribution: its share in the cell is approximated by the pixel's own flag
        c = c - hit; s = s - (own.reshape(-1) & hit)
    share = np.where(c >= 2, s / np.maximum(c, 1), np.nan).reshape(u.shape)
    g = ct[np.ix_(rows, cols)].astype(np.float32)
    img = np.stack([g, g, g], -1) * 0.8
    img[own & u] = img[own & u] * 0.4 + np.array([60, 110, 255]) * 0.6
    sh = np.nan_to_num(share)
    m = (sh >= 0.25) & u
    col = np.stack([np.full(m.sum(), 255.0), 230 * (1 - np.clip((sh[m] - 0.25) / 0.5, 0, 1)), np.zeros(m.sum())], -1)
    img[m] = img[m] * 0.3 + col * 0.7
    img[~u] *= 0.35
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(png)


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
    v.add_argument('--stride', type=int, default=2); v.add_argument('--show', nargs='*', type=Path, default=[])
    q = sub.add_parser('queue'); q.add_argument('stitch', type=Path); q.add_argument('survey', type=Path); q.add_argument('ct')
    q.add_argument('--tag', default='V'); q.add_argument('--cover', type=int, default=3); q.add_argument('--limit', type=int)
    a = ap.parse_args(argv)
    if a.cmd == 'vote':
        import glob
        pats = []
        for p in a.patches:                       # a quoted pattern (work/survey/V1545_*) is expanded here
            pats += sorted(Path(x) for x in glob.glob(str(p))) if any(ch in str(p) for ch in '*?[') else [p]
        pats = [p for p in pats if (p / 'finished.txt').exists() or (p / 'seed42_s6_reverse.tif').exists() and (p / 'ct_mid.tif').exists()]
        res = vote(a.out, pats, a.stride, show=a.show)
        print(json.dumps({k: v for k, v in res.items() if k != 'per_patch'}, indent=1))
    else:
        queue_surveys(a.stitch, a.survey, a.ct, a.tag, a.cover, a.limit)


if __name__ == '__main__':
    main()
