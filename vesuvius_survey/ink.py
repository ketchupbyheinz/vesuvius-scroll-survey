"""ink.py - run two independent public ink models on both faces of a rendered surface volume.

Models (downloaded and sha256-checked on first use):
  seed42   scrollprize/ink_9um  hybrid_3d2d-seed42 step 75000   (Vesuvius Challenge team)
  dnative  Nieuwlaar/ink9um-dense-native step 16000             (fine-tune used for the PHerc. 343 First Letters win)

Both run through villa's flat inference (`vesuvius.ink_detection.inference.infer`) on a 17-layer window
centred on the surface, in both normal directions, so every patch gets four maps:
  <name>_s6.tif, <name>_s6_reverse.tif  for name in (seed42, dnative)

Works on a 6 GB laptop GPU (batch size 4 / 2); on CPU it is ~20x slower but still finishes.

  python -m vesuvius_survey.ink RENDER.zarr OUT_DIR [--ckpt-dir ckpt]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

MODELS = {
    'seed42': {
        'file': 'seed42_step-075000.pth',
        'url': 'https://huggingface.co/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed42/step-075000.pth',
        'sha256': 'e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab',
        'batch': 4,
    },
    'dnative': {
        'file': 'dense_native-016000.rebuilt.pth',
        'weights': ('https://huggingface.co/Nieuwlaar/ink9um-dense-native/resolve/main/weights/dense_native_016000.safetensors',
                    'd3d95dc41dcb24ed76b84b5e12ca5dae25d7fb34edf1dfbb93e0186f91e6fc6d'),
        'config': ('https://huggingface.co/Nieuwlaar/ink9um-dense-native/resolve/main/training/configs/train_dense_native.json',
                   '5d8459030b36cd3980a03610c684365a7407d497173eff433efdb37f75840c86'),
        'batch': 2,
    },
}
LAYER_START, LAYER_END = 6, 23   # 17 of 28 layers, centred on the surface


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _get(url: str, dst: Path, sha: str):
    if dst.exists() and _sha(dst) == sha:
        return
    print('downloading', url, flush=True)
    tmp = dst.with_suffix(dst.suffix + '.part')
    urllib.request.urlretrieve(url, tmp)
    got = _sha(tmp)
    if got != sha:
        tmp.unlink()
        raise SystemExit(f'sha256 mismatch for {url}: {got}')
    tmp.replace(dst)


def ensure_models(ckpt: Path) -> dict[str, Path]:
    ckpt.mkdir(parents=True, exist_ok=True)
    m = MODELS['seed42']
    _get(m['url'], ckpt / m['file'], m['sha256'])
    d = MODELS['dnative']
    out = ckpt / d['file']
    if not out.exists():
        w, c = ckpt / 'dense_native_016000.safetensors', ckpt / 'train_dense_native.json'
        _get(d['weights'][0], w, d['weights'][1])
        _get(d['config'][0], c, d['config'][1])
        import torch
        from safetensors.torch import load_file
        # same wrapping as Nieuwlaar's rebuild_dense_native_pth.py: weights + training config
        torch.save({'model': load_file(str(w)), 'config': json.load(open(c)), 'step': 16000}, out)
    return {'seed42': ckpt / m['file'], 'dnative': out}


def run(render: Path, out_dir: Path, ckpt: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = ensure_models(ckpt)
    for name, ck in paths.items():
        if (out_dir / f'{name}_s6_reverse.tif').exists():
            continue
        cmd = [sys.executable, '-m', 'vesuvius.ink_detection.inference.infer', str(render), str(ck),
               str(out_dir / f'{name}_s6.tif'), '--overlap', '0.5', '--blend-mode', 'hann',
               '--batch-size', str(MODELS[name]['batch']), '--direction', 'both',
               '--layer-start', str(LAYER_START), '--layer-end', str(LAYER_END),
               '--no-compile', '--num-workers', '0']
        print(' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('render', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--ckpt-dir', type=Path, default=Path('ckpt'))
    a = ap.parse_args()
    run(a.render, a.out, a.ckpt_dir)


if __name__ == '__main__':
    main()
