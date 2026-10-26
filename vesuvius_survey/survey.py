"""survey.py - one command from "scroll name" to scored report cards.

  python -m vesuvius_survey.survey --scroll PHerc0211 --out work/survey --vc /path/to/vc_grow_seg_from_seed \
         [--z 0.3 0.5 0.7] [--radius 0.7] [--n 2] [--ckpt-dir ckpt]

For every seed (n per height): trace a ~4-7 cm2 patch, render a 28-layer surface volume, run both ink
models on both faces, and write a report card (PNG + JSON) with:
  sheet_good_fraction   share of the patch where the CT shows the papyrus weave (trace on one sheet)
  agree_top3            overlap of the two models' strongest 3% of pixels, on the better face
  agree_top3_other_face the same on the other face (real ink is one-sided; texture responses are not)
  face_contrast_ratio   ink contrast of the better face / other face
  chain                 how strongly letter-sized ink blobs line up along a fibre direction (1 = random)
  face_consistency      share of tiles where one face wins clearly, minus those where the other face does
  consistent_side       whether that face is the inner one (towards the scroll centre, where the text is)

Everything streams from the open-data bucket; nothing larger than the rendered patch is stored.
Steps that already have outputs are skipped, so the command can be re-run to resume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from . import ink, render_patch, report_card, seeds, side, trace


def run_patch(info: dict, patch: Path, work: Path, ckpt: Path):
    x = tifffile.imread(patch / 'x.tif')
    H, W = x.shape[0] * 20, x.shape[1] * 20
    render = work / 'render.zarr'
    if not (work / 'render.done').exists():
        render_patch.main([str(patch), info['ct_url'], str(render), '--rows', '0', str(H), '--cols', '0', str(W),
                           '--layers', '28', '--tile', '512', '--workers', '8', '--resume'])
        (work / 'render.done').touch()
    ink.run(render, work, ckpt)
    import zarr
    a = zarr.open_array(str(render), mode='r')
    ct = np.asarray(a[a.shape[0] // 2]).astype(np.float32)
    res, ct, cross, valid, good, emap = report_card.analyse(work, ct, info['voxel_um'])
    res.update({'scroll': info['scroll'], 'patch': patch.name})
    fi = side.forward_inward(patch, info['scroll'])
    res.update({'forward_inward_score': round(fi, 3), 'consistent_side': side.side_of(res['consistent_face'], fi),
                'curvature': round(side.curvature_sign(patch), 3)})
    report_card.draw(res, ct, cross, valid, good, emap, f"{info['scroll']}  {patch.name}", work / 'card.png')
    (work / 'card.json').write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scroll', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--vc', default='vc_grow_seg_from_seed')
    ap.add_argument('--z', type=float, nargs='+', default=[0.5])
    ap.add_argument('--radius', type=float, default=0.7)
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--ckpt-dir', type=Path, default=Path('ckpt'))
    ap.add_argument('--generations', type=int, default=55, help='trace growth steps (55 = ~5-7 cm2)')
    a = ap.parse_args()
    info = trace.scroll_info(a.scroll)
    base = a.out / a.scroll
    results = []
    for z in a.z:
        for s in seeds.pick(a.scroll, z, a.radius, a.n):
            tag = f"{s['x']}_{s['y']}_{s['z']}"
            work = base / tag
            work.mkdir(parents=True, exist_ok=True)
            patch = next((p for p in (work / 'patches').glob('*') if (p / 'x.tif').exists()), None) \
                if (work / 'patches').exists() else None
            if patch is None:
                patch = trace.grow(info, (s['x'], s['y'], s['z']), work, a.vc, a.generations)
            if patch is None:
                print('trace failed', tag)
                continue
            r = run_patch(info, patch, work, a.ckpt_dir)
            r['seed'] = s
            results.append(r)
            print(json.dumps(r))
    (base / 'summary.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
