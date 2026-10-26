"""Per-tile face asymmetry: where in a patch do the two models agree on one face only?"""
import sys, numpy as np, tifffile, cv2
sys.path.insert(0, '/home/claude/vesuvius/tools')
from inkcheck import crossness_map, CROSS_THRESHOLD, _norm
from pathlib import Path
def tiles(d, T=768):
    d = Path(d); ct = tifffile.imread(d / 'ct_mid.tif').astype(np.float32)
    F = {}
    for side, suf in (('forward', ''), ('reverse', '_reverse')):
        a = tifffile.imread(d / f'seed42_s6{suf}.tif').astype(np.float32); b = tifffile.imread(d / f'dnative_s6{suf}.tif').astype(np.float32)
        h, w = min(a.shape[0], ct.shape[0]), min(a.shape[1], ct.shape[1]); F[side] = (a[:h, :w], b[:h, :w])
    ct = ct[:h, :w]
    cross, _, valid = crossness_map(ct)
    cross = cv2.blur(np.where(valid, cross, 0).astype(np.float32), (129, 129)) / np.maximum(cv2.blur(valid.astype(np.float32), (129, 129)), 1e-3)
    m = (cross >= CROSS_THRESHOLD) & valid
    top = {}
    for s, (a, b) in F.items():
        na, nb = _norm(a, m), _norm(b, m)
        top[s] = (na >= np.quantile(na[m], .97)) & (nb >= np.quantile(nb[m], .97)) & m, (na >= np.quantile(na[m], .97)) | (nb >= np.quantile(nb[m], .97))
    out = []
    for y in range(0, h - T + 1, T // 2):
        for x in range(0, w - T + 1, T // 2):
            sl = (slice(y, y + T), slice(x, x + T))
            if m[sl].mean() < 0.5: continue
            iou = {s: top[s][0][sl].sum() / max((top[s][1][sl] & m[sl]).sum(), 1) for s in top}
            out.append((y, x, iou['forward'], iou['reverse'], m[sl].mean()))
    return out
if __name__ == '__main__':
    for d in sys.argv[1:]:
        o = tiles(d)
        o.sort(key=lambda t: -max(t[2], t[3]) / max(min(t[2], t[3]), 0.02))
        print(Path(d).name, len(o), 'tiles')
        for y, x, f, r, g in o[:6]:
            print(f'  y{y:5d} x{x:5d}  fwd {f:.3f} rev {r:.3f}  ratio {max(f,r)/max(min(f,r),0.02):.2f}  good {g:.2f}')
