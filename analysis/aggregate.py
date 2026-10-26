"""Collect every report card + sheet grade into results/survey.json and results/survey.csv."""
import json, glob, os, csv
from pathlib import Path

ROOT = Path('/home/claude/vesuvius')
grades = json.load(open(ROOT / 'survey/grades.json'))
ELIGIBLE = {'PHerc0125', 'PHerc0175A', 'PHerc0175B', 'PHerc0191', 'PHerc0211', 'PHerc0257', 'PHerc0268', 'PHerc0306B',
            'PHerc0358', 'PHerc0483A', 'PHerc0483B', 'PHerc0490A', 'PHerc0490B', 'PHerc0800', 'PHerc0813', 'PHerc0826',
            'PHerc0846A', 'PHerc0846B', 'PHerc1203', 'PHerc1218', 'PHerc1545'}
# early patches made before the tag convention: map to scroll + seed
r2 = json.load(open(ROOT / 'seeds_round2.json'))
ALIAS = {f'p125_n{i+1}': ('PHerc0125', *r2[i][1:]) for i in range(5)}
ALIAS['p191_a'] = ('PHerc0191', *r2[5][1:])
ALIAS['POS_PHerc0343_erwin'] = ('PHerc0343', None, None, None)
for a in ALIAS:
    c = json.load(open(ROOT / f'survey/cards/{a}.json'))
    grades.setdefault(a, {'good_fraction': c['sheet_good_fraction'], 'median_cross': None})
DARK = json.load(open(ROOT / 'survey/dark.json')) if (ROOT / 'survey/dark.json').exists() else {}
CURV = json.load(open(ROOT / 'survey/curv_side.json')) if (ROOT / 'survey/curv_side.json').exists() else {}
def _tags(f):
    try:
        return {'%s_%d_%d_%d' % (e[0], e[2], e[3], e[4]) for e in json.load(open(ROOT / f))}
    except FileNotFoundError:
        return set()
FOLLOW = set().union(*[_tags(f) for f in ('seeds_lead211.json', 'seeds_leads2.json', 'seeds_leads3.json', 'seeds_leads4.json',
                                           'seeds_leads5.json', 'seeds_leads6.json', 'seeds_leads7.json')])
SIDE = json.load(open(ROOT / 'survey/face_side.json')) if (ROOT / 'survey/face_side.json').exists() else {}
rows = []
for tag, g in sorted(grades.items()):
    if tag in ALIAS:
        scroll, x, y, z = ALIAS[tag]
    else:
        scroll = tag.split('_')[0]
        x, y, z = (int(v) for v in tag.split('_')[1:4])
    card = ROOT / f'survey/cards/{tag}.json'
    c = json.load(open(card)) if card.exists() else {}
    um = 9.362
    ctf = Path(f'/mnt/user-data/outputs/survey/{tag}/ct.txt')
    if ctf.exists() and '8.640um' in ctf.read_text():
        um = 8.64
    elif scroll in ('PHerc0175A', 'PHerc0175B', 'PHerc0306B', 'PHerc0483A', 'PHerc0490A', 'PHerc0490B', 'PHerc0800', 'PHerc1218', 'PHerc0343', 'PHerc0268', 'PHerc0483B'):
        um = 8.64
    role = 'positive_control' if scroll == 'PHerc0343' else ('first_letters_eligible' if scroll in ELIGIBLE else 'other')
    sample = 'follow_up' if tag in FOLLOW else ('windows_test' if tag == 'PHerc1545_3499_4074_9431' else 'survey')
    r = {'tag': tag, 'role': role, 'sample': sample, 'scroll': scroll, 'seed_x': x, 'seed_y': y, 'seed_z': z, 'voxel_um': um,
         'trace_sheet_good_fraction': g['good_fraction'], 'trace_median_cross': g['median_cross'],
         'traced_ok': g['good_fraction'] >= 0.4, 'inked': bool(c)}
    for k in ('sheet_good_fraction', 'face', 'agree_top3', 'agree_top3_other_face', 'face_contrast_ratio', 'chain', 'chain_other_face', 'face_consistency', 'consistent_face', 'tiles', 'strong_tiles', 'strong_tiles_other_face'):
        r[k] = c.get(k)
    if tag in SIDE:
        r['consistent_side'] = SIDE[tag]['consistent_side']
        r['forward_inward_score'] = SIDE[tag]['forward_inward_score']
        if tag in CURV:
            r['curvature'] = round(CURV[tag], 3)
    if tag in DARK:
        r.update(DARK[tag])
    if c:
        r['face_asymmetry'] = c.get('face_asymmetry') or round(c['agree_top3'] / max(c['agree_top3_other_face'], 1e-3), 3)
    rows.append(r)
os.makedirs(ROOT / 'repo/results', exist_ok=True)
json.dump(rows, open(ROOT / 'repo/results/survey.json', 'w'), indent=1)
keys = list(rows[0].keys()) + ['face_asymmetry', 'consistent_side', 'forward_inward_score', 'curvature', 'ink_on_dark', 'ink_ct_z']
keys = list(dict.fromkeys(keys))
with open(ROOT / 'repo/results/survey.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k) for k in keys}) for r in rows]
neg = [r['face_asymmetry'] for r in rows if r['inked'] and r['role'] == 'first_letters_eligible']
import statistics
print('eligible inked', len(neg), 'median', round(statistics.median(neg), 2), 'max', max(neg), 'scrolls', len({r['scroll'] for r in rows if r['role']=='first_letters_eligible'}))
print(len(rows), 'patches;', sum(r['traced_ok'] for r in rows), 'traced ok;', sum(r['inked'] for r in rows), 'inked')
for r in sorted([r for r in rows if r['inked']], key=lambda r: -r['face_asymmetry'])[:12]:
    print(f"{r['scroll']:11s} {str(r['seed_x']):>5s} {str(r['seed_y']):>5s} {str(r['seed_z']):>5s}  sheet {r['sheet_good_fraction']:.2f}  agree {r['agree_top3']:.3f}/{r['agree_top3_other_face']:.3f}  asym {r['face_asymmetry']:.2f}  chain {r['chain']:.2f}")
