"""Agreement of seed43 (cloud run) with seed42 and dense_native on both faces, on sheet-check-good pixels,
plus tile face-consistency of each model pair."""
import sys, numpy as np, tifffile, cv2
sys.path.insert(0, 'tools')
from inkcheck import _norm, _top_iou, crossness_map, CROSS_THRESHOLD
t = sys.argv[1]; S43 = sys.argv[2] if len(sys.argv) > 2 else f'leads/{t}_seed43'
U = f'/mnt/user-data/uploads/Vesuvius/work/survey/{t}/'
ct = tifffile.imread(U + 'ct_mid.tif').astype(np.float32)
cross, _, valid = crossness_map(ct)
cross = cv2.blur(np.where(valid, cross, 0).astype(np.float32), (129, 129)) / np.maximum(cv2.blur(valid.astype(np.float32), (129, 129)), 1e-3)
m = (cross >= CROSS_THRESHOLD) & valid
def L(p): a = tifffile.imread(p).astype(np.float32); h, w = min(a.shape[0], ct.shape[0]), min(a.shape[1], ct.shape[1]); z = np.zeros_like(ct); z[:h, :w] = a[:h, :w]; return _norm(z, m)
res = {}
for face, suf in (('forward', ''), ('reverse', '_reverse')):
    a, b, c = L(U + f'seed42_s6{suf}.tif'), L(U + f'dnative_s6{suf}.tif'), L(f'{S43}{suf}.tif')
    res[face] = {'42-dn': _top_iou(a[m], b[m]), '43-42': _top_iou(c[m], a[m]), '43-dn': _top_iou(c[m], b[m])}
for k in res['forward']:
    f, r = res['forward'][k], res['reverse'][k]
    print(f'{t} {k}: forward {f:.3f} reverse {r:.3f} ratio {max(f, r) / max(min(f, r), 1e-3):.2f} ({"forward" if f > r else "reverse"})')
