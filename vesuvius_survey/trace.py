"""trace.py - grow a papyrus surface patch from a seed point, streaming everything from the open-data bucket.

Wraps VC3D's vc_grow_seg_from_seed with the two inputs it needs to stay on one sheet:
  * the scroll's published surface prediction (streamed as OME-Zarr), and
  * the matching normal grids (streamed on demand through a normal-grids-remote.json marker).
Without the normal grids the tracer can wander across layers; with them it follows one sheet.

  python -m vesuvius_survey.trace --scroll PHerc0125 --seed 3699 3698 13499 --out work/PHerc0125 \
         --vc vc_grow_seg_from_seed [--generations 55]

--scroll accepts any scroll in the open-data bucket that has a published m7 surface prediction.
The scan id, voxel size and prediction names are looked up from the bucket listing.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from pathlib import Path

BUCKET = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'


def _list(prefix: str) -> list[str]:
    url = f'{BUCKET}/?list-type=2&prefix={prefix}&delimiter=/'
    xml = urllib.request.urlopen(url, timeout=60).read().decode()
    return re.findall(r'<Prefix>([^<]*)</Prefix>', xml)


def scroll_info(scroll: str) -> dict:
    """Find the masked CT volume and the m7 surface prediction of a scroll in the open-data bucket."""
    vols = [p for p in _list(f'{scroll}/volumes/') if p.endswith('-masked.zarr/')]
    preds = [p for p in _list(f'{scroll}/representations/predictions/surfaces/') if p.endswith('.zarr/')]
    if not vols or not preds:
        raise SystemExit(f'{scroll}: no masked volume or surface prediction in the open-data bucket')
    vol = vols[0].rstrip('/').split('/')[-1]
    scan = vol.split('-')[0]
    um = float(vol.split('-')[1].replace('um', ''))
    pred = next((p for p in preds if p.split('/')[-2].startswith(scan)), preds[0]).rstrip('/')
    return {'scroll': scroll, 'scan': scan, 'voxel_um': um,
            'ct_url': f'{BUCKET}/{scroll}/volumes/{vol}',
            'surface_url': f'{BUCKET}/{pred}',
            'normal_grids_url': f'{BUCKET}/{pred[:-len(".zarr")]}.normal-grids'}


def grow(info: dict, seed_xyz, out: Path, vc_bin: str, generations: int = 55, threads: int = 2) -> Path | None:
    out.mkdir(parents=True, exist_ok=True)
    ng = out / 'normal-grids'
    ng.mkdir(exist_ok=True)
    (ng / 'normal-grids-remote.json').write_text(json.dumps({'url': info['normal_grids_url']}))
    params = {'mode': 'seed', 'step_size': 20, 'generations': generations, 'min_area_cm': 0.05,
              'thread_limit': threads, 'use_cuda': False, 'voxelsize': info['voxel_um'],
              'normal_grid_path': str(ng.resolve()), 'cache_root': str((out / 'cache').resolve()),
              'cache_size': 2_000_000_000}
    pj = out / 'grow.json'
    pj.write_text(json.dumps(params))
    patches = out / 'patches'
    patches.mkdir(exist_ok=True)
    before = set(p.name for p in patches.iterdir())
    import os, shutil
    exe = [vc_bin] if os.path.exists(vc_bin) else vc_bin.split()
    if not (os.path.exists(exe[0]) or shutil.which(exe[0])):
        raise SystemExit(f'vc_grow_seg_from_seed not found: {vc_bin} (pass --vc with the path from the VC3D release)')
    cmd = [*exe, '-v', info['surface_url'], '-t', str(patches), '-p', str(pj),
           '-s', *[str(int(v)) for v in seed_xyz]]
    log = out / f'trace_{"_".join(str(int(v)) for v in seed_xyz)}.log'
    with open(log, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    new = [p for p in patches.iterdir() if p.name not in before and (p / 'x.tif').exists()]
    return new[0] if new else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scroll', required=True)
    ap.add_argument('--seed', nargs=3, type=float, required=True, metavar=('X', 'Y', 'Z'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--vc', default='vc_grow_seg_from_seed', help='path to vc_grow_seg_from_seed (VC3D release)')
    ap.add_argument('--generations', type=int, default=55)
    a = ap.parse_args()
    info = scroll_info(a.scroll)
    print(json.dumps(info, indent=1))
    p = grow(info, a.seed, a.out, a.vc, a.generations)
    print('patch:', p)


if __name__ == '__main__':
    main()
