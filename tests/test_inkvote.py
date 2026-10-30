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


def test_sheets_are_grouped_by_offset(tmp_path):
    # three overlapping patches: two on the same sheet (r = 1000), one on the next winding out (r = 1012),
    # plus one bridging patch on r = 1000 so the graph is connected
    ds = [_arc(tmp_path, n, r, a0, a0 + 0.4, 0, 600)[0]
          for n, r, a0 in (('a', 1000, 0.0), ('b', 1000, 0.2), ('c', 1012, 0.1), ('d', 1000.5, 0.3))]
    T, grids, o_p, lay, counts = inkvote.sheet_numbers([inkvote.mesh_nodes(d / 'mesh') for d in ds])
    per = [set(lay[g[g >= 0]].tolist()) for g in grids]
    assert per[0] == per[1] == per[3] == {0}             # the well-covered sheet is sheet 0
    assert len(per[2]) == 1 and per[2] != per[0]         # the next winding is a sheet of its own


def test_patch_orientation_follows_overlaps(tmp_path):
    # same sheet, but patch b is stored mirrored (columns reversed), so its mesh normal points the other way
    a, _ = _arc(tmp_path, 'a', 1000, 0.0, 0.4, 0, 600)
    b, _ = _arc(tmp_path, 'b', 1000, 0.2, 0.6, 0, 600)
    for k in 'xyz':
        t = tifffile.imread(b / 'mesh' / f'{k}.tif'); tifffile.imwrite(b / 'mesh' / f'{k}.tif', t[:, ::-1].copy())
    T, grids, o_p, lay, counts = inkvote.sheet_numbers([inkvote.mesh_nodes(d / 'mesh') for d in (a, b)])
    assert o_p.tolist() == [1, -1]
    assert set(lay[grids[0][grids[0] >= 0]].tolist()) == set(lay[grids[1][grids[1] >= 0]].tolist()) == {0}


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
    res = inkvote.vote(tmp_path / 'vote', ds, stride=2, log=lambda *a: None)
    assert res['front']['agreed_cells'] > 50                         # the repeated spot is agreed in 3D
    assert res['front']['vs_control'] > 5                            # far more than with the maps turned around
    assert res['back']['agreed_cells'] <= 5
    assert res['face_drawn'] == 'front' and list((tmp_path / 'vote').glob('consensus_front_*.png'))
