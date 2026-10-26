"""Which ink face (forward/reverse) of a patch faces the scroll's centre? Forward = +normal (render_patch normals)."""
import sys, json, functools, numpy as np, zarr
from pathlib import Path
from scipy import ndimage
sys.path.insert(0, 'tools'); sys.path.insert(0, 'repo')
from render_patch import load_mesh, node_normals
from vesuvius_survey.trace import scroll_info
O = Path('/mnt/user-data/outputs/survey')

@functools.lru_cache(None)
def centre(scroll, zb, level=3):
    ct = zarr.open_group(scroll_info(scroll)['ct_url'], mode='r'); f = 2 ** level
    a = np.asarray(ct[str(level)][zb // f])
    m = ndimage.binary_fill_holes(ndimage.binary_opening(a > 40, iterations=2))
    lab, k = ndimage.label(m); m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    cy, cx = ndimage.center_of_mass(m)
    return cx * f, cy * f

def forward_is_inward(tag, scroll):
    p, v, _ = load_mesh(O / tag / 'mesh'); n = node_normals(p, v)
    ok = np.isfinite(n).all(-1)
    P, N = p[ok], n[ok]
    zb = int(np.median(P[:, 2])) // 64 * 64
    cx, cy = centre(scroll, zb)
    inward = np.stack([cx - P[:, 0], cy - P[:, 1]], 1); inward /= np.linalg.norm(inward, axis=1, keepdims=True)
    d = (N[:, :2] * inward).sum(1)
    return float(np.mean(np.sign(d)))

if __name__ == '__main__':
    rows = json.load(open('repo/results/survey.json'))
    res = {}
    for r in rows:
        if not r['inked'] or not (O / r['tag'] / 'mesh').exists(): continue
        from vesuvius_survey.side import forward_inward
        s = forward_inward(O / r['tag'] / 'mesh', r['scroll'])
        from vesuvius_survey.side import side_of
        cf = r['consistent_face']
        res[r['tag']] = {'forward_inward_score': round(s, 3), 'consistent_face': cf, 'consistent_side': side_of(cf, s),
                         'best_face_side': side_of(r['face'], s)}
        print(r['tag'], res[r['tag']], 'cons', r['face_consistency'], flush=True)
    json.dump(res, open('survey/face_side.json', 'w'), indent=1)
