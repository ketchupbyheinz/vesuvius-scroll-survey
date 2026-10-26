"""Pairwise model agreement (top-3 % IoU) on each face at d=0 for seed42, dense_native (survey maps), seed43 and
Hecate (deep maps; Hecate PNG resized to the render grid). Prints inner/outer ratios per pair.
four_model.py TAG"""
import sys, json, itertools, numpy as np, tifffile, cv2
from pathlib import Path
sys.path.insert(0, 'tools')
from inkcheck import _norm, _top_iou, crossness_map, CROSS_THRESHOLD
t = sys.argv[1]
S = Path(f'/mnt/user-data/uploads/Vesuvius/work/survey/{t}'); D = Path(f'/mnt/user-data/uploads/Vesuvius/work/deep/{t}/maps')
r = {x['tag']: x for x in json.load(open('repo/results/survey.json'))}[t]
fi = r['forward_inward_score']; inner = 'forward' if fi > 0 else 'reverse'
ct = tifffile.imread(S / 'ct_mid.tif').astype(np.float32); H, W = ct.shape
cross, _, valid = crossness_map(ct)
cross = cv2.blur(np.where(valid, cross, 0).astype(np.float32), (129, 129)) / np.maximum(cv2.blur(valid.astype(np.float32), (129, 129)), 1e-3)
m = (cross >= CROSS_THRESHOLD) & valid
def load(p):
    if not p.exists() or p.stat().st_size == 0: return None
    a = (cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.suffix == '.png' else tifffile.imread(p)).astype(np.float32)
    if a.ndim == 3: a = a[..., 0]
    if p.suffix == '.png': a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
    return a
raw = {}
for face, suf in (('forward', ''), ('reverse', '_reverse')):
    raw[face] = {'seed42': load(S / f'seed42_s6{suf}.tif'), 'dnative': load(S / f'dnative_s6{suf}.tif'),
                 'seed43': load(D / f'seed43_d+0{suf}.tif'), 'hecate': load(D / f'hecate_d+0{suf}.png')}
# crop everything to the common extent, as report_card does with seed42/dense_native
hh = min([H] + [a.shape[0] for f in raw.values() for a in f.values() if a is not None])
ww = min([W] + [a.shape[1] for f in raw.values() for a in f.values() if a is not None])
m = m[:hh, :ww]
maps = {f: {k: (None if a is None else _norm(a[:hh, :ww], m)) for k, a in d.items()} for f, d in raw.items()}
names = [k for k in maps['forward'] if maps['forward'][k] is not None and maps['reverse'][k] is not None]
outer = 'reverse' if inner == 'forward' else 'forward'
res = {}
for a, b in itertools.combinations(names, 2):
    i = round(_top_iou(maps[inner][a][m], maps[inner][b][m]), 4); o = round(_top_iou(maps[outer][a][m], maps[outer][b][m]), 4)   # rounded as on the card
    res[f'{a}-{b}'] = round(i / max(o, 1e-3), 3)
    print(f'{t} {a:8s}-{b:8s} inner {i:.3f} outer {o:.3f} ratio {i / max(o, 1e-3):.2f}')
json.dump({'patch': t, 'inner_face': inner, 'ratio_inner_over_outer': res, 'note': 'top-3 % IoU of each model pair on sheet-check-good pixels, d=0'}, open(f'repo/results/leads/model_pairs_{t}.json', 'w'), indent=1)
