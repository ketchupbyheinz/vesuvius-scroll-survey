import json

import numpy as np
import tifffile

from vesuvius_survey import inkvote


def _arc(root, name, r, a0, a1, z0, z1, c=(0.0, 0.0), step=20.0):
    """tifxyz patch on a cylinder of radius r around c (a vertical sheet of one winding), angles in radians."""
    d = root / name / 'mesh'
    d.mkdir(parents=True)
    na = int((a1 - a0) * r / step) + 1; nz = int((z1 - z0) / step) + 1
    a = np.linspace(a0, a0 + (na - 1) * step / r, na); z = z0 + np.arange(nz) * step
    A, Z = np.meshgrid(a, z)
    for k, v in zip('xyz', (c[0] + r * np.cos(A), c[1] + r * np.sin(A), Z)):
        tifffile.imwrite(d / f'{k}.tif', v.astype(np.float32))
    (d / 'meta.json').write_text(json.dumps({'scale': [0.05, 0.05], 'format': 'tifxyz'}))
    return d.parent, (nz * 20, na * 20)


def test_rank_map_is_a_percentile():
    a = np.arange(100, dtype=np.uint8).reshape(10, 10)
    r = inkvote.rank_map(a, np.ones_like(a, bool))
    assert abs(r[0, 0] - 0.005) < 1e-6 and abs(r[9, 9] - 0.995) < 1e-6


def test_scroll_centre_from_normals(tmp_path):
    d, _ = _arc(tmp_path, 'p', 1000, 0.0, 1.0, 0, 400, c=(300.0, -200.0))
    pts, valid, scale, nrm, ok = inkvote.mesh_nodes(d / 'mesh')
    c = inkvote.scroll_centre([pts[ok]], [nrm[ok]])
    assert np.allclose(c, (300, -200), atol=1.0)


def test_windings_are_numbered_by_offset(tmp_path):
    # three overlapping patches: two on the same sheet (r = 1000), one on the next winding out (r = 1012),
    # plus one bridging patch on r = 1000 so the graph is connected
    ds = [_arc(tmp_path, n, r, a0, a0 + 0.4, 0, 600)[0]
          for n, r, a0 in (('a', 1000, 0.0), ('b', 1000, 0.2), ('c', 1012, 0.1), ('d', 1000.5, 0.3))]
    nodes = [inkvote.mesh_nodes(d / 'mesh') for d in ds]
    T, grids = inkvote.make_tiles(nodes, np.zeros(2))
    pairs, o = inkvote.tile_edges(T)
    lay = inkvote.tile_layers(T, pairs, o, 12.0)
    per = [set(lay[g[g >= 0]].tolist()) for g in grids]
    assert per[0] == per[1] == per[3] and len(per[0]) == 1
    (l0,), (l2,) = per[0], per[2]
    assert l2 == l0 + 1                                  # outward = +1


def test_vote_finds_ink_that_repeats_across_patches(tmp_path):
    rng = np.random.default_rng(0)
    ds = []
    for i, a0 in enumerate((0.0, 0.05, 0.1, 0.15)):
        d, (H, W) = _arc(tmp_path, f'p{i}', 1000, a0, a0 + 0.5, 0, 800)
        ds.append(d)
        pts, valid, scale, nrm, ok = inkvote.mesh_nodes(d / 'mesh')
        rows, cols = np.arange(H), np.arange(W)
        xyz, _ = inkvote.pixel_xyz(pts, valid, scale, rows, cols)
        ang = np.arctan2(xyz[..., 1], xyz[..., 0])
        ink = (np.abs(ang - 0.3) < 0.02) & (np.abs(xyz[..., 2] - 400) < 60)       # same 3D spot in every patch
        ct = rng.normal(120, 8, (H, W))
        ct += 25 * np.sin(np.arange(W) / 3.0)[None] + 25 * np.sin(np.arange(H) / 3.0)[:, None]   # crossing fibres
        tifffile.imwrite(d / 'ct_mid.tif', np.clip(ct, 0, 255).astype(np.uint8))
        for m in ('seed42', 'dnative'):
            for suf in ('', '_reverse'):
                a = rng.integers(0, 120, (H, W)).astype(np.uint8)
                if suf == '':
                    a[ink] = 230
                tifffile.imwrite(d / f'{m}_s6{suf}.tif', a)
    G_out = tmp_path / 'vote'
    res = inkvote.vote(G_out, ds, grid=4.0, stride=2, log=lambda *a: None)
    z = np.load(G_out / 'vote.npz')
    face = 'outer' if res['per_patch'][0]['orientation'] > 0 else 'inner'
    cov = z[f'cov_L+0_{face}']; top = z[f'top_L+0_{face}']
    k = cov >= 3
    frac = top[k] / cov[k]
    assert (frac > 0.9).sum() > 50                      # the repeated spot comes out in every covering patch
    assert np.mean(frac > 0.5) < 0.1                    # and not much else
