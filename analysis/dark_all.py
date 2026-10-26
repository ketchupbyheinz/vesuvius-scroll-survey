"""ink_on_dark / ink_ct_z for every staged patch -> survey/dark.json (same code as the report card)."""
import sys, os, json, numpy as np, tifffile, cv2
sys.path.insert(0, 'tools')
from inkcheck import _norm, crossness_map, CROSS_THRESHOLD
from report_card import ink_on_dark
U = '/mnt/user-data/uploads/Vesuvius/work/survey/'
rows = {r['tag']: r for r in json.load(open('repo/results/survey.json'))}
out = json.load(open('survey/dark.json')) if os.path.exists('survey/dark.json') else {}
Q = '/mnt/user-data/uploads/Vesuvius/work/queue/'
for t, r in rows.items():
    if t in out or not r['inked']: continue
    base = U if os.path.exists(U + t + '/ct_mid.tif') else (Q if os.path.exists(Q + t + '/ct_mid.tif') else None)
    if base is None: continue
    ct = tifffile.imread(base + t + '/ct_mid.tif').astype(np.float32)
    cross, _, valid = crossness_map(ct)
    cross = cv2.blur(np.where(valid, cross, 0).astype(np.float32), (129, 129)) / np.maximum(cv2.blur(valid.astype(np.float32), (129, 129)), 1e-3)
    good = (cross >= CROSS_THRESHOLD) & valid
    m = good if good.mean() > 0.05 else valid
    suf = '' if r['face'] == 'forward' else '_reverse'
    a = _norm(tifffile.imread(base + t + f'/seed42_s6{suf}.tif').astype(np.float32)[:ct.shape[0], :ct.shape[1]], m)
    b = _norm(tifffile.imread(base + t + f'/dnative_s6{suf}.tif').astype(np.float32)[:ct.shape[0], :ct.shape[1]], m)
    h, w = a.shape; ct2, m2 = ct[:h, :w], m[:h, :w]
    agreed = (a >= np.quantile(a[m2], .97)) & (b >= np.quantile(b[m2], .97)) & m2
    out[t] = ink_on_dark(ct2, agreed, m2)
    json.dump(out, open('survey/dark.json', 'w'))
print(len(out))
