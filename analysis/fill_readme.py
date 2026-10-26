"""Fill tools/README.template.md from repo/results -> repo/README.md."""
import json, statistics, subprocess, glob
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path('/home/claude/vesuvius')
rows = json.load(open(ROOT / 'repo/results/survey.json'))
t = open(ROOT / 'tools/README.template.md').read()
elig = [r for r in rows if r['role'] == 'first_letters_eligible']
S = [r for r in elig if r.get('sample') == 'survey']          # first-pass survey patches
SI = [r for r in S if r['inked']]
F = [r for r in rows if r.get('sample') == 'follow_up']
asym = [r['face_asymmetry'] for r in SI]; cons = [r['face_consistency'] for r in SI]
med = lambda v: statistics.median(v)

# per-scroll table: survey patches only
by = defaultdict(list)
for r in S:
    by[r['scroll']].append(r)
fu = defaultdict(int)
for r in F:
    fu[r['scroll']] += 1
lines = ['| scroll | survey patches traced | pass sheet check | inked | best face asymmetry (seed x, y, z) | best face consistency (seed x, y, z) | follow-up patches |',
         '|---|---|---|---|---|---|---|']
for s in sorted(by):
    rs = by[s]; ok = [r for r in rs if r['traced_ok']]; ik = [r for r in rs if r['inked']]
    b = max(ik, key=lambda r: r['face_asymmetry']) if ik else None
    c = max(ik, key=lambda r: r['face_consistency']) if ik else None
    name = s.replace('PHerc', 'PHerc. ').replace('. 0', '. ')
    lines.append(f"| {name} | {len(rs)} | {len(ok)} | {len(ik)} | "
                 + (f"{b['face_asymmetry']:.2f} ({b['seed_x']}, {b['seed_y']}, {b['seed_z']}) | {c['face_consistency']:.2f} ({c['seed_x']}, {c['seed_y']}, {c['seed_z']}) |" if b else "— | — |")
                 + f" {fu.get(s, 0) or '—'} |")

def curv_agree(rs):
    cc = [r for r in rs if r.get('curvature') is not None and r.get('forward_inward_score') is not None
          and abs(r['forward_inward_score']) > 0.5 and abs(r['curvature']) > 0.05]
    return sum((r['forward_inward_score'] > 0) == (r['curvature'] < 0) for r in cc), len(cc)
ca, cn = curv_agree(rows); cas, cns = curv_agree(SI)

# model-pair table from results/leads/model_pairs_*.json
order = ['seed42-dnative', 'seed42-seed43', 'dnative-seed43', 'seed42-hecate', 'dnative-hecate', 'seed43-hecate']
NAMES = {'POS_PHerc0343_erwin': 'PHerc. 343 known text (Nieuwlaar segment)'}
mp = []
for f in sorted(glob.glob(str(ROOT / 'repo/results/leads/model_pairs_*.json'))):
    j = json.load(open(f)); p = j['patch']
    nm = NAMES.get(p, p.split('_')[0].replace('PHerc0', 'PHerc. ').replace('PHerc1', 'PHerc. 1') + ' (' + ', '.join(p.split('_')[1:4]) + ')')
    rr = j['ratio_inner_over_outer']
    mp.append((0 if p.startswith('POS') else 1, nm, [f"{rr[k]:.2f}" if k in rr else '—' for k in order]))
mp.sort()
mpt = ['| patch | ' + ' | '.join(k.replace('-', ' / ').replace('dnative', 'dense_native') for k in order) + ' |',
       '|---' * (len(order) + 1) + '|'] + [f"| {nm} | " + ' | '.join(v) + ' |' for _, nm, v in mp]

t = (t.replace('__N_NEG__', str(len(SI))).replace('__NEG_MED__', f'{med(asym):.2f}')
      .replace('__NEG_MAX__', f'{max(asym):.2f}').replace('__SURVEY_TABLE__', '\n'.join(lines))
      .replace('__N_SCROLLS_INKED__', str(len({r['scroll'] for r in SI})))
      .replace('__CONS_MED__', f'{med(cons):.2f}').replace('__CONS_MAX__', f'{max(cons):.2f}')
      .replace('__R_SHEET_CONS__', f"{np.corrcoef([r['sheet_good_fraction'] for r in SI], cons)[0, 1]:.2f}")
      .replace('__CURV_AGREE_S__', str(cas)).replace('__CURV_N_S__', str(cns))
      .replace('__CURV_AGREE__', str(ca)).replace('__CURV_N__', str(cn))
      .replace('__S_DARK_MED__', f"{med([r['ink_on_dark'] for r in SI if r.get('ink_on_dark') is not None]):.2f}")
      .replace('__MODEL_PAIRS_TABLE__', '\n'.join(mpt))
      .replace('__LEADS_SECTION__', subprocess.run(['python3', 'tools/leads_table.py'], cwd=ROOT, capture_output=True, text=True).stdout.strip()))
tk = ROOT / 'tools/takeaways.md'
t = t.replace('__TAKEAWAYS__', tk.read_text().strip())
t = (t.replace('__S_TRACED__', str(len(S))).replace('__S_OK__', str(sum(r['traced_ok'] for r in S)))
      .replace('__S_INKED__', str(len(SI))).replace('__F_TRACED__', str(len(F)))
      .replace('__S_ASYM_MED__', f'{med(asym):.2f}').replace('__S_CONS_MED__', f'{med(cons):.2f}'))
for sc in ('PHerc0175B', 'PHerc0268'):
    rs = [r for r in S if r['scroll'] == sc]
    t = t.replace(f'__{sc}__', f"{sum(r['traced_ok'] for r in rs)} of {len(rs)}")
open(ROOT / 'repo/README.md', 'w').write(t)
left = [w for w in t.split() if w.startswith('__') and w.endswith('__')]
print('unfilled:', left)
