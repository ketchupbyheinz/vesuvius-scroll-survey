"""Seeds that map the extent of a hotspot: farther windings along the normal, and the same sheet shifted sideways."""
import sys, os, json, numpy as np, tifffile, zarr
sys.path.insert(0, 'tools'); sys.path.insert(0, 'repo')
from tile_asym import tiles
from vesuvius_survey.trace import scroll_info
U = '/mnt/user-data/uploads/Vesuvius/work/survey/'; O = '/mnt/user-data/outputs/survey/'
tag = sys.argv[1]; out_file = sys.argv[2]
scroll = tag.split('_')[0]; info = scroll_info(scroll); vol = info['ct_url'].split('/')[-1]
o = tiles(U + tag)
F = sum(1 for t in o if t[2] > 0.1 and t[2] > 3 * t[3]); R = sum(1 for t in o if t[3] > 0.1 and t[3] > 3 * t[2])
k, j = (2, 3) if F >= R else (3, 2)
y, x = max(o, key=lambda t: t[k] - t[j])[:2]
X, Y, Z = [tifffile.imread(f'{O}{tag}/mesh/{c}.tif') for c in 'xyz']
iy, ix = (y + 384) // 20, (x + 384) // 20
def P(a, b): return np.array([X[a, b], Y[a, b], Z[a, b]], float)
du = (P(iy, ix + 1) - P(iy, ix - 1)) / 2; dv = (P(iy + 1, ix) - P(iy - 1, ix)) / 2
n = np.cross(du, dv); n /= np.linalg.norm(n); du /= np.linalg.norm(du); dv /= np.linalg.norm(dv)
c = P(iy, ix)
sp = zarr.open_group(info['surface_url'], mode='r')['0']
offs = [(d, 0, 0) for d in (-90, -75, -60, 60, 75, 90)] + [(0, s * 500, 0) for s in (-1, 1)] + [(0, 0, s * 500) for s in (-1, 1)]
res = []
for d, a, b in offs:
    q = c + d * n + a * du + b * dv
    cx, cy, cz = (int(round(v)) for v in q); r = 5
    blk = np.asarray(sp[cz - r:cz + r + 1, cy - r:cy + r + 1, cx - r:cx + r + 1]); idx = np.argwhere(blk > 0)
    if not len(idx): print(d, a, b, 'no surface'); continue
    bb = idx[np.linalg.norm(idx - r, axis=1).argmin()]
    s = [scroll, info['scan'], int(cx - r + bb[2]), int(cy - r + bb[1]), int(cz - r + bb[0]), vol, '%.3f' % info['voxel_um']]
    res.append(s); print(d, a, b, s[2:5], flush=True)
old = json.load(open(out_file)) if os.path.exists(out_file) else []
json.dump(old + res, open(out_file, 'w'))
