"""Markdown table of every lead and all patches traced around it, from results/survey.json + the seed lists."""
import json, statistics as st
rows = {r['tag']: r for r in json.load(open('repo/results/survey.json'))}
def tags(f): return ['%s_%d_%d_%d' % (e[0], e[2], e[3], e[4]) for e in json.load(open(f))]
L = {k: tags(f'seeds_{k}.json') for k in ('lead211', 'leads2', 'leads3', 'leads4', 'leads5', 'leads6', 'leads7')}
pick = lambda k, p: [t for t in L[k] if t.startswith(p)]
LEADS = [('PHerc0846A_2765_2952_4205', L['leads5'] + L['leads6']),
         ('PHerc0846B_3561_4951_6961', pick('leads2', 'PHerc0846B') + L['leads4']),
         ('PHerc1545_3499_4074_9431', L['leads7']),
         ('PHerc0306B_2473_4325_7949', pick('leads2', 'PHerc0306B')),
         ('PHerc0813_4779_5850_8497', pick('leads2', 'PHerc0813')),
         ('PHerc0846A_1866_4064_7009', pick('leads2', 'PHerc0846A')),
         ('PHerc0257_3549_4692_13210', pick('leads2', 'PHerc0257')),
         ('PHerc0211_3449_4349_9706', L['lead211'] + L['leads3'])]
dk = lambda x: f"{x['ink_on_dark']:.2f}" if x.get('ink_on_dark') is not None else '-'
out = ['| lead (first patch) | first patch: asym / consistency / side / ink on dark | follow-up patches scored | side: inner / outer / unclear | median asym | median consistency | median ink on dark | best follow-up (asym / consistency / ink on dark) |',
       '|---|---|---|---|---|---|---|---|']
for lead, nb in LEADS:
    r = rows[lead]; n = [rows[t] for t in nb if t in rows and rows[t]['inked']]
    if not n: continue
    inner = sum(x.get('consistent_side') == 'inner' for x in n)
    outer = sum(x.get('consistent_side') == 'outer' for x in n)
    mc = st.median([x['face_consistency'] for x in n])
    b = max(n, key=lambda x: x['face_asymmetry'] * max(x['face_consistency'], .05))
    name = lead.split('_')[0].replace('PHerc0', 'PHerc. ').replace('PHerc1', 'PHerc. 1')
    out.append(f"| {name} ({lead.split('_', 1)[1].replace('_', ', ')}) | {r['face_asymmetry']:.2f} / {r['face_consistency']:.2f} / {r.get('consistent_side')} / {dk(r)} | {len(n)} | {inner} / {outer} / {len(n) - inner - outer} | "
               f"{st.median([x['face_asymmetry'] for x in n]):.2f} | {mc:.2f} | {st.median([x['ink_on_dark'] for x in n if x.get('ink_on_dark') is not None]):.2f} | {b['face_asymmetry']:.2f} / {b['face_consistency']:.2f} / {dk(b)} |")
print('\n'.join(out))
print('\nThe PHerc. 211 lead (the first lead of this project) was also checked with seed43 on CPU on one follow-up '
      'patch (3186, 4545, 9432): seed43 did not prefer either face with dense_native (1.06x).')
