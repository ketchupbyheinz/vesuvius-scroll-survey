"""seeds_survey4.json: wider coverage - 2 heights x 2 directions per eligible scroll, round-robin order."""
import json, sys
sys.path.insert(0, 'repo')
from vesuvius_survey.seeds import pick
from vesuvius_survey.trace import scroll_info
import numpy as np
SCROLLS = ['PHerc0175A', 'PHerc0175B', 'PHerc0191', 'PHerc0257', 'PHerc0268', 'PHerc0306B', 'PHerc0358', 'PHerc0483A',
           'PHerc0483B', 'PHerc0490A', 'PHerc0490B', 'PHerc0800', 'PHerc0813', 'PHerc0826', 'PHerc0846A', 'PHerc0846B',
           'PHerc1203', 'PHerc1218', 'PHerc1545', 'PHerc0125']
per = {}
for s in SCROLLS:
    try:
        info = scroll_info(s)
        vol = info['ct_url'].split('/')[-1]
        seeds = []
        for z in (0.3, 0.7):
            seeds += pick(s, z, 0.6, 2, angles=(np.pi / 4, -3 * np.pi / 4, 3 * np.pi / 4, -np.pi / 4))
        per[s] = [[s, info['scan'], d['x'], d['y'], d['z'], vol, '%.3f' % info['voxel_um']] for d in seeds]
        print(s, len(per[s]), flush=True)
    except Exception as e:
        print(s, 'ERR', e, flush=True)
out = []
for i in range(4):
    for s in SCROLLS:
        if s in per and i < len(per[s]):
            out.append(per[s][i])
json.dump(out, open('seeds_survey4.json', 'w'))
print(len(out))
