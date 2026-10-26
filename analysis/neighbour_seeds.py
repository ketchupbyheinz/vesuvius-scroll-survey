"""Seeds on neighbouring windings over the most face-consistent tile of a patch.
Real text continues on the windings either side; texture artefacts need not."""
import sys, json, numpy as np, tifffile, zarr
sys.path.insert(0, 'tools'); sys.path.insert(0, 'repo')
from tile_asym import tiles
from vesuvius_survey.trace import scroll_info
if len(sys.argv) < 2 or not __import__('os').environ.get('OUT'):
    raise SystemExit('usage: OUT=seeds_x.json neighbour_seeds.py TAG [TAG...]')
U = '/mnt/user-data/uploads/Vesuvius/work/survey/'
O = '/mnt/user-data/outputs/survey/'
out = []
for tag in sys.argv[1:]:
    scroll = tag.split('_')[0]
    info = scroll_info(scroll); vol = info['ct_url'].split('/')[-1]
    o = tiles(U + tag)
    F = sum(1 for t in o if t[2] > 0.1 and t[2] > 3 * t[3]); R = sum(1 for t in o if t[3] > 0.1 and t[3] > 3 * t[2])
    k, j = (2, 3) if F >= R else (3, 2)
    y, x = max(o, key=lambda t: t[k] - t[j])[:2]
    X, Y, Z = [tifffile.imread(f'{O}{tag}/mesh/{c}.tif') for c in 'xyz']
    iy, ix = (y + 384) // 20, (x + 384) // 20
    P = np.array([X[iy, ix], Y[iy, ix], Z[iy, ix]])
    du = np.array([X[iy, ix + 1] - X[iy, ix - 1], Y[iy, ix + 1] - Y[iy, ix - 1], Z[iy, ix + 1] - Z[iy, ix - 1]])
    dv = np.array([X[iy + 1, ix] - X[iy - 1, ix], Y[iy + 1, ix] - Y[iy - 1, ix], Z[iy + 1, ix] - Z[iy - 1, ix]])
    n = np.cross(du, dv); n /= np.linalg.norm(n)
    sp = zarr.open_group(info['surface_url'], mode='r')['0']
    for d in (-45, -30, -15, 15, 30, 45):
        q = P + d * n
        cx, cy, cz = (int(round(v)) for v in q); r = 5
        blk = np.asarray(sp[cz - r:cz + r + 1, cy - r:cy + r + 1, cx - r:cx + r + 1])
        idx = np.argwhere(blk > 0)
        if not len(idx):
            print(tag, d, 'no surface'); continue
        b = idx[np.linalg.norm(idx - r, axis=1).argmin()]
        s = [scroll, info['scan'], int(cx - r + b[2]), int(cy - r + b[1]), int(cz - r + b[0]), vol, '%.3f' % info['voxel_um']]
        out.append(s); print(tag, d, s[2:5], flush=True)
json.dump(out, open(sys.argv[0].replace('tools/neighbour_seeds.py','') + (__import__('os').environ.get('OUT') or 'seeds_leads2.json'), 'w'))
print(len(out))
