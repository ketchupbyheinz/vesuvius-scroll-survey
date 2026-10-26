#!/usr/bin/env bash
# Compare vesuvius_survey.render_patch with VC3D's vc_render_tifxyz on the same 600x600 px, 28-layer patch.
#
#   docs/validate_renderer.sh "VC3D.AppImage vc_render_tifxyz" [workdir]
#
# Uses a 30x30-node crop of results/meshes/PHerc0343_4551_3852_10201 (PHerc. 343, 8.64 um scan) and streams
# the CT from the open-data bucket. Expected output (VC3D 322cb08):
#   identical 0.9962  max_diff 1  corr 0.999998
set -euo pipefail
VC=${1:?command for vc_render_tifxyz, e.g. "VC3D.AppImage vc_render_tifxyz"}
W=${2:-validate_work}
HERE=$(cd "$(dirname "$0")/.." && pwd)
CT=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr
mkdir -p "$W"

python - "$HERE/results/meshes/PHerc0343_4551_3852_10201" "$W/crop" <<'EOF'
import json, sys, tifffile
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for c in 'xyz':
    tifffile.imwrite(dst / f'{c}.tif', tifffile.imread(src / f'{c}.tif')[40:70, 40:70])
meta = json.load(open(src / 'meta.json'))
meta.update(uuid='crop', name='crop')
json.dump(meta, open(dst / 'meta.json', 'w'))
EOF

$VC -v "$CT" -g 0 --scale 1 -s "$W/crop" -n 28 --zarr-output "$W/vc3d.zarr"
python -m vesuvius_survey.render_patch "$W/crop" "$CT" "$W/mine.zarr" --rows 0 600 --cols 0 600 --layers 28 --step 1 --tile 512

python - "$W" <<'EOF'
import sys, numpy as np, zarr
w = sys.argv[1]
a = np.asarray(zarr.open(f'{w}/vc3d.zarr', mode='r')['0'][:]).astype(int)
b = np.asarray(zarr.open(f'{w}/mine.zarr', mode='r')[:]).astype(int)
m = (a > 0) | (b > 0)
d = np.abs(a - b)[m]
print(f'identical {(d == 0).mean():.4f}  max_diff {d.max()}  corr {np.corrcoef(a[m], b[m])[0, 1]:.6f}  voxels {m.sum()}')
EOF
