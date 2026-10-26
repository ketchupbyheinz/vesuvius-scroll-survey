"""Second, geometry-only estimate of which way 'forward' points: normals of a rolled sheet converge towards the roll
axis. s = mean over node pairs of (n_i - n_j).(p_i - p_j) / |p_i - p_j|^2 : > 0 normals diverge (forward = outward)."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, 'tools'); sys.path.insert(0, 'repo')
from render_patch import load_mesh, node_normals
O = Path('/mnt/user-data/outputs/survey')
def curv(tag, rng=np.random.default_rng(0)):
    p, v, _ = load_mesh(O / tag / 'mesh'); n = node_normals(p, v); ok = np.isfinite(n).all(-1)
    P, N = p[ok].astype(float), n[ok].astype(float)
    i, j = rng.integers(0, len(P), 20000), rng.integers(0, len(P), 20000)
    d = P[i] - P[j]; L = (d ** 2).sum(1); m = L > 200 ** 2
    return float(np.median(((N[i] - N[j]) * d).sum(1)[m] / L[m]) * 1000)
rows = json.load(open('repo/results/survey.json'))
agree = dis = 0; out = {}
for r in rows:
    if not r['inked'] or r.get('forward_inward_score') is None or not (O / r['tag'] / 'mesh').exists(): continue
    from vesuvius_survey.side import curvature_sign
    c = curvature_sign(O / r['tag'] / 'mesh'); fi = r['forward_inward_score']
    out[r['tag']] = c
    if abs(fi) > 0.5 and abs(c) > 0.05:
        same = (fi > 0) == (c < 0)   # forward inward <=> normals converge
        agree += same; dis += not same
        if not same or r['scroll'] in ('PHerc0846B', 'PHerc0306B', 'PHerc0343'):
            print(f"{r['tag']:28s} COM {fi:+.2f}  curvature {c:+.2f}  {'agree' if same else 'DISAGREE'}")
print('agree', agree, 'disagree', dis)
json.dump(out, open('survey/curv_side.json', 'w'))
