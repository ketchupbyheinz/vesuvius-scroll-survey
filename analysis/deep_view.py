"""deep_view.py TAG [y0 y1 x0 x1] - ensemble of the depth-sweep maps of one patch (Nieuwlaar-style):
per model one z-score over the reading face pooled over all depths, then the mean over models, per depth.
Writes /tmp/claude-0/deep_<TAG>.png (CT + ensemble per depth for the crop) and prints a per-depth summary."""
import sys, re, json, glob, numpy as np, tifffile, cv2
from pathlib import Path
sys.path.insert(0, '/home/claude/vesuvius/tools')
from inkcheck import _norm

tag = sys.argv[1]
D = Path(f'/mnt/user-data/uploads/Vesuvius/work/deep/{tag}/maps')
S = Path(f'/mnt/user-data/uploads/Vesuvius/work/survey/{tag}')
rows = {r['tag']: r for r in json.load(open('/home/claude/vesuvius/repo/results/survey.json'))}
face = rows[tag.replace('L44', '')]['consistent_face']
ct = tifffile.imread(S / 'ct_mid.tif').astype(np.float32)
v = ct > 0
maps = {}
for f in sorted(D.glob('*.tif')):
    m = re.match(r'(\w+?)_d([+-]\d+)(_reverse)?\.tif', f.name)
    if not m: continue
    fc = 'reverse' if m.group(3) else 'forward'
    maps.setdefault(fc, {}).setdefault(m.group(1), {})[int(m.group(2))] = f
def zstack(fc):
    out = {}
    for model, dd in maps[fc].items():
        arr = {d: tifffile.imread(p).astype(np.float32)[:ct.shape[0], :ct.shape[1]] for d, p in dd.items()}
        pool = np.concatenate([a[v] for a in arr.values()])
        mu, sd = pool.mean(), pool.std() + 1e-6
        for d, a in arr.items():
            out.setdefault(d, []).append((a - mu) / sd)
    return {d: np.mean(z, 0) for d, z in out.items() if len(z) == len(maps[fc])}
read, blind = zstack(face), zstack('reverse' if face == 'forward' else 'forward')
depths = sorted(read)
print(tag, 'reading face', face, 'models', sorted(maps[face]), 'depths', depths)
prof = {}
for d in depths:
    r, b = read[d][v], blind.get(d, read[d])[v]
    prof[d] = {'reading_frac_z_gt_2': round(float(np.mean(r > 2)), 4), 'blind_frac_z_gt_2': round(float(np.mean(b > 2)), 4)}
    print(f'd={d:+d}  reading p99.5 {np.quantile(r, .995):.2f}  blind p99.5 {np.quantile(b, .995):.2f}  frac>2 read {np.mean(r > 2):.4f} blind {np.mean(b > 2):.4f}')
import os
os.makedirs('/home/claude/vesuvius/repo/results/leads', exist_ok=True)
json.dump({'patch': tag, 'reading_face': face, 'models': sorted(maps[face]), 'per_model_zscore': 'pooled over all depths, reading face',
           'depth_profile': prof}, open(f'/home/claude/vesuvius/repo/results/leads/depth_sweep_{tag}.json', 'w'), indent=1)
if len(sys.argv) > 5:
    y0, y1, x0, x1 = map(int, sys.argv[2:6])
    C = (_norm(ct, v)[y0:y1, x0:x1] * 255).astype(np.uint8)
    tiles = [C]
    for d in depths:
        e = np.clip((read[d][y0:y1, x0:x1] - 0.5) / 3.0, 0, 1)
        t = (e * 255).astype(np.uint8)
        cv2.putText(t, f'd={d:+d}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 2); tiles.append(t)
    mean3 = np.mean([read[d] for d in depths if abs(d) <= 1], 0)[y0:y1, x0:x1]
    t = (np.clip((mean3 - 0.5) / 3.0, 0, 1) * 255).astype(np.uint8)
    cv2.putText(t, 'mean d-1..+1', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 2); tiles.append(t)
    while len(tiles) % 3: tiles.append(np.zeros_like(C))
    grid = np.concatenate([np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)], 0)
    out = f'/tmp/claude-0/deep_{tag}.png'
    cv2.imwrite(out, grid); print(out)
