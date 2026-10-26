"""Whole-patch overlay (Nieuwlaar style): ensemble ink (mean z over models, mean over depths LO..HI) in yellow over a
dimmed CT layer, only on sheet-check-good tiles. deep_overlay.py TAG LO HI OUT [scale]"""
import sys, re, numpy as np, tifffile, cv2, json
from pathlib import Path
sys.path.insert(0, '/home/claude/vesuvius/tools')
from inkcheck import _norm, crossness_map, CROSS_THRESHOLD
tag, lo, hi, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sc = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
D = Path(f'/mnt/user-data/uploads/Vesuvius/work/deep/{tag}/maps'); S = Path(f'/mnt/user-data/uploads/Vesuvius/work/survey/{tag}')
face = {r['tag']: r for r in json.load(open('/home/claude/vesuvius/repo/results/survey.json'))}[tag.replace('L44', '')]['consistent_face']
ct = tifffile.imread(S / 'ct_mid.tif').astype(np.float32); v = ct > 0
cross, _, valid = crossness_map(ct)
cross = cv2.blur(np.where(valid, cross, 0).astype(np.float32), (129, 129)) / np.maximum(cv2.blur(valid.astype(np.float32), (129, 129)), 1e-3)
good = (cross >= CROSS_THRESHOLD) & valid
by = {}
for f in list(D.glob('*.tif')) + list(D.glob('*.png')):
    m = re.match(r'(\w+?)_d([+-]\d+)(_reverse)?\.(tif|png)', f.name)
    if not m or ('reverse' if m.group(3) else 'forward') != face: continue
    by.setdefault(m.group(1), {})[int(m.group(2))] = f
zs = []
for model, dd in by.items():
    def ld(p):
        if p.suffix == '.png':
            a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32)
            a = a[..., 0] if a.ndim == 3 else a
            return cv2.resize(a, (ct.shape[1], ct.shape[0]), interpolation=cv2.INTER_LINEAR)
        a = tifffile.imread(p).astype(np.float32); z = np.zeros(ct.shape, np.float32)
        h, w = min(a.shape[0], ct.shape[0]), min(a.shape[1], ct.shape[1]); z[:h, :w] = a[:h, :w]; return z
    arr = [ld(p) for d, p in dd.items() if lo <= d <= hi and p.stat().st_size > 0]
    if not arr: continue
    a = np.mean(arr, 0); zs.append((a - a[good].mean()) / (a[good].std() + 1e-6))
e = np.mean(zs, 0)
e = np.clip((e - 1.0) / 2.0, 0, 1) * good
c = _norm(ct, v) * (0.45 + 0.25 * good)
img = c[..., None] * (1 - e[..., None]) + np.array([0.35, 0.88, 1.0])[None, None] * e[..., None]   # BGR yellow
img = cv2.resize((img * 255).clip(0, 255).astype(np.uint8), None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
cv2.imwrite(out, img); print(out, img.shape, 'models', sorted(by), 'face', face)
