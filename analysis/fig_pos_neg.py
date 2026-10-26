"""docs/img/pos_vs_neg.png: where both ink models agree (top 3 %) on each face, known text vs a typical
eligible patch, plus the face-asymmetry distribution of every inked patch."""
import json, sys
from pathlib import Path
import numpy as np, tifffile, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/claude/vesuvius/tools')
from inkcheck import crossness_map, CROSS_THRESHOLD, _norm

ROOT = Path('/home/claude/vesuvius')
U = Path('/mnt/user-data/uploads/Vesuvius/work/survey')
rows = json.load(open(ROOT / 'repo/results/survey.json'))
inked = [r for r in rows if r['inked']]
elig = [r for r in inked if r['role'] == 'first_letters_eligible' and r.get('sample') == 'survey']
follow = [r for r in inked if r['role'] == 'first_letters_eligible' and r.get('sample') == 'follow_up']
wtest = [r for r in inked if r.get('sample') == 'windows_test']
med = float(np.median([r['face_asymmetry'] for r in elig]))
neg = min((r for r in elig if r['sheet_good_fraction'] >= 0.7), key=lambda r: abs(r['face_asymmetry'] - med))


def agree_maps(tag):
    d = U / tag
    ct = tifffile.imread(d / 'ct_mid.tif').astype(np.float32)
    out = {}
    for side, suf in (('forward', ''), ('reverse', '_reverse')):
        a = tifffile.imread(d / f'seed42_s6{suf}.tif').astype(np.float32)
        b = tifffile.imread(d / f'dnative_s6{suf}.tif').astype(np.float32)
        h, w = min(a.shape[0], ct.shape[0]), min(a.shape[1], ct.shape[1])
        out[side] = (a[:h, :w], b[:h, :w])
    ct = ct[:h, :w]
    cross, _, valid = crossness_map(ct)
    cross = cv2.blur(np.where(valid, cross, 0).astype(np.float32), (129, 129)) / np.maximum(cv2.blur(valid.astype(np.float32), (129, 129)), 1e-3)
    m = (cross >= CROSS_THRESHOLD) & valid
    res = {}
    for side, (a, b) in out.items():
        na, nb = _norm(a, m), _norm(b, m)
        ta, tb = np.quantile(na[m], 0.97), np.quantile(nb[m], 0.97)
        res[side] = (na >= ta) & (nb >= tb) & m
    return _norm(ct, ct > 0), m, res


def window(both, m, size=1100):
    dens = cv2.boxFilter(both.astype(np.float32), -1, (size, size), normalize=True) + 0.2 * cv2.boxFilter(m.astype(np.float32), -1, (size, size))
    h = size // 2
    sub = dens[h:-h or None, h:-h or None]
    y, x = np.unravel_index(np.argmax(sub), sub.shape)
    return slice(y, y + size), slice(x, x + size)


def panel(ax, ct, m, both, title, win):
    ct, m, both = ct[win], m[win], both[win]
    k = 2
    small = lambda x: cv2.resize(x.astype(np.float32), (x.shape[1] // k, x.shape[0] // k), interpolation=cv2.INTER_AREA)
    c, mm, bb = small(ct), small(m), small(both)
    img = np.stack([c * 0.5] * 3, -1) * (0.35 + 0.65 * mm[..., None])
    bb = np.clip(bb * 3, 0, 1)[..., None]
    img = img * (1 - bb) + np.array([1.0, 0.82, 0.25]) * bb
    ax.imshow(img.clip(0, 1)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10, loc='left')
    for s in ax.spines.values():
        s.set_visible(False)


fig = plt.figure(figsize=(14, 7.6), dpi=110)
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.12, 1.5], wspace=0.06, hspace=0.25)
for i, (tag, label, r) in enumerate((
        ('POS_PHerc0343_erwin', 'Known text: PHerc. 343, Nieuwlaar First Letters segment', next(r for r in rows if r['tag'] == 'POS_PHerc0343_erwin')),
        (neg['tag'], f"Typical survey patch: {neg['scroll'].replace('PHerc0', 'PHerc. ').replace('PHerc1', 'PHerc. 1')} ({neg['seed_x']}, {neg['seed_y']}, {neg['seed_z']})", neg))):
    ct, m, res = agree_maps(tag)
    best = r['face']; other = 'reverse' if best == 'forward' else 'forward'
    win = window(res[best], m)
    panel(fig.add_subplot(gs[i, 0]), ct, m, res[best], f"{label}\nbest face: models agree on {r['agree_top3']:.2f}", win)
    panel(fig.add_subplot(gs[i, 1]), ct, m, res[other], f"\nother face: {r['agree_top3_other_face']:.2f}   ->  asymmetry {r['face_asymmetry']:.1f}x", win)

ax = fig.add_subplot(gs[:, 3])
KNOWN = ('POS_PHerc0343_erwin', 'PHerc0343_5911_4059_10523')
ax.scatter([r['face_consistency'] for r in follow], [r['face_asymmetry'] for r in follow], s=26, facecolor='none',
           edgecolor='#9aa3ad', linewidth=0.9, zorder=2, label='lead follow-ups (not a random sample)')
ax.scatter([r['face_consistency'] for r in wtest], [r['face_asymmetry'] for r in wtest], s=40, marker='D', facecolor='none',
           edgecolor='#5577aa', linewidth=1.1, zorder=3, label='Windows end-to-end test patch')
groups = [('PHerc. 343 known text', [r for r in inked if r['tag'] in KNOWN]),
          ('PHerc. 343 other patches', [r for r in inked if r['role'] == 'positive_control' and r['tag'] not in KNOWN]),
          (f'survey patches, 20 scrolls (n={len(elig)})', elig)]
cols = ['#d08a1e', '#c9b48a', '#5577aa']
for gi, (name, rs) in enumerate(groups):
    ax.scatter([r['face_consistency'] for r in rs], [r['face_asymmetry'] for r in rs], s=[60, 40, 36][gi],
               color=cols[gi], alpha=0.9, edgecolor='white', linewidth=0.6, zorder=3, label=name.replace('\n', ' '))
lab = sorted(elig, key=lambda r: -(r['face_consistency'] + 0.25 * r['face_asymmetry']))[:4] + [max(follow, key=lambda r: r['face_asymmetry'])] + wtest
for r in lab:
    nm = r['scroll'].replace('PHerc0', 'PHerc. ').replace('PHerc1', 'PHerc. 1')
    ax.annotate(nm, (r['face_consistency'], r['face_asymmetry']), xytext=(6, 4), textcoords='offset points', fontsize=8.5, color='#334')
ax.set_xlabel('face consistency (share of tiles favouring one face)', fontsize=10)
ax.set_ylabel('face asymmetry (whole patch)', fontsize=10)
ax.axhline(1, color='#bbb', lw=0.8, ls='--', zorder=1)
ax.set_xlim(-0.03, 0.85)
ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#eee', zorder=0)
ax.legend(frameon=False, fontsize=8.5, loc='upper left', bbox_to_anchor=(-0.02, -0.1), ncol=2)
ax.set_title('Every inked patch', fontsize=11, loc='left')
ax.set_ylim(0.8, max(r['face_asymmetry'] for r in inked) + 0.4)
fig.text(0.01, -0.07, 'Yellow: pixels in both models\' top 3 % (seed42 + dense_native), on tiles that pass the sheet check (dimmed tiles fail it). 1100 x 1100 px crops of each patch.', fontsize=9, color='#555')
out = ROOT / 'repo/docs/img/pos_vs_neg.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches='tight', facecolor='white')
print(out, 'neg =', neg['tag'], 'median', med)
