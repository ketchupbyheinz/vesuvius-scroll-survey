import json

import numpy as np
import tifffile

from vesuvius_survey import patchset


def _plane(root, name, x0, y0, z, n=20, step=20.0):
    d = root / f'auto_grown_{name}'
    d.mkdir(parents=True)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    for c, a in zip('xyz', (x0 + xx * step, y0 + yy * step, np.full_like(xx, z))):
        tifffile.imwrite(d / f'{c}.tif', a)
    (d / 'meta.json').write_text(json.dumps({'scale': [0.05, 0.05], 'format': 'tifxyz', 'bbox': [[0, 0, 0], [1, 1, 1]]}))
    return d.name


def test_same_winding_accepts_the_seed_sheet_rejects_parallel_ones_and_spreads(tmp_path):
    seed = _plane(tmp_path, 'seed', 0, 0, 100)
    on = _plane(tmp_path, 'on', 150, 0, 101.5)          # same sheet, overlapping, 1.5 voxels off
    chain = _plane(tmp_path, 'chain', 400, 0, 101)      # overlaps only `on` (x 400-530): accepted in round two
    nb = _plane(tmp_path, 'neighbour', 100, 100, 113)   # parallel, one winding (13 voxels) away
    far = _plane(tmp_path, 'far', 5000, 5000, 100)      # overlaps nothing
    rep = patchset.same_winding(tmp_path, seed, thr=4)
    st = rep['state']
    assert st[seed] == st[on] == st[chain] == 'accepted'
    assert st[nb] == 'rejected' and st[far] == 'undecided'
    assert rep['counts'] == {'accepted': 3, 'rejected': 1, 'undecided': 1}
    assert rep['median_abs_offset'][nb] > 10 and rep['median_abs_offset'][on] < 2


def test_move_rejected_and_previous_acceptances_are_kept(tmp_path):
    paths = tmp_path / 'paths'
    seed = _plane(paths, 'seed', 0, 0, 100)
    nb = _plane(paths, 'neighbour', 100, 100, 113)
    rep = patchset.same_winding(paths, seed)
    assert patchset.move_rejected(paths, rep, tmp_path / 'off') == 1
    assert not (paths / nb).exists() and (tmp_path / 'off' / nb / 'z.tif').exists()
    later = _plane(paths, 'later', 5000, 5000, 100)
    rep2 = patchset.same_winding(paths, seed, previous={later: 'accepted'})
    assert rep2['state'][later] == 'accepted'      # an earlier acceptance is not re-decided


def test_stub_volume_has_the_shape_and_voxel_size(tmp_path):
    patchset.write_stub(tmp_path, (100, 50, 40), 8.64)
    z = json.loads((tmp_path / 'stub' / '0' / '.zarray').read_text())
    assert z['shape'] == [100, 50, 40] and json.loads((tmp_path / 'stub' / 'meta.json').read_text())['voxelsize'] == 8.64


FAKE_VC = '''
import sys, json, numpy as np, tifffile, pathlib
a = sys.argv; t = pathlib.Path(a[a.index("-t") + 1]); s = int(a[a.index("--rng-seed") + 1])
d = t / f"auto_grown_{s}"; d.mkdir()
yy, xx = np.mgrid[0:20, 0:20].astype(np.float32)
z = 100.5 if s % 2 == 0 else 113.0            # even seeds: on the seed winding; odd: one winding away
for c, v in zip("xyz", (100 + xx * 20, yy * 20, np.full_like(xx, z))):
    tifffile.imwrite(d / f"{c}.tif", v)
(d / "meta.json").write_text(json.dumps({"scale": [0.05, 0.05], "format": "tifxyz", "bbox": [[0, 0, 0], [1, 1, 1]]}))
print("saving", d)
'''


def test_steer_keeps_only_seed_winding_patches_and_parks_undecided(tmp_path):
    import sys
    st = tmp_path / 'stitch'
    seed = _plane(st / 'paths', 'seed', 0, 0, 100)
    lonely = _plane(st / 'paths', 'lonely', 9000, 9000, 100)      # overlaps nothing: parked, then returned
    (st / 'logs').mkdir()
    (st / 'expand.json').write_text('{}')
    fake = tmp_path / 'fake_vc.py'
    fake.write_text(FAKE_VC)
    patchset.steer(f'{sys.executable} {fake}', st, 'unused-url', seed, n_parallel=2, per_round=2, hours=2 / 3600, rng_base=10)
    kept = sorted(p.name for p in (st / 'paths').iterdir())
    assert 'auto_grown_10' in kept and 'auto_grown_11' not in kept and lonely in kept
    assert (st / 'off' / 'auto_grown_11').exists() and not any((st / 'parked').iterdir())
