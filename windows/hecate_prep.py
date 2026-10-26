#!/usr/bin/env python3
# From Erwin Nieuwlaar's PHerc. 343 First Letters repository, unchanged. Copyright (c) 2026 Erwin Nieuwlaar, MIT licence.
"""hecate_prep.py - turn a surface render (uint8, Z,Y,X, isotropic at the native voxel size) into a
Hecate-9.6um input: isotropic 9.6 um in x, y AND z, uint8, Z,Y,X, depth exactly --nz planes.

  python hecate_prep.py --src <zarr-group|tifdir> --native-um 8.64 --native-um-z 8.64 \
      --z-center 32.5 --nz 16 --out out.zarr [--crop y0 x0 h w]

The z samples lie at native layer positions  z_center + (i - (nz-1)/2) * (9.6/native_um_z),  i = 0..nz-1,
linearly interpolated between the two neighbouring native layers. With --nz 16 Hecate uses exactly these
16 planes (its central-window start is 16//2 - 8 = 0), so the forward and the reverse run see exactly the
same window, only in opposite order. In-plane, each plane is resized by native_um/target_um
(cv2.INTER_AREA when shrinking by more than 5 %, else cv2.INTER_LINEAR).

For the PHerc0343 66-layer renders (layer k at k - 32.5 voxels along the normal) the call is
--native-um 8.64 --native-um-z 8.64 --target-um 9.6 --nz 16 --z-center 32.5 + d.

Same algorithm and output as the script used for all maps in this submission (only comments and
messages translated; verified byte-identical output on the gate crop).
"""
import argparse, os, glob
import numpy as np, cv2


def load_src(src, crop=None):
    if os.path.isdir(src) and glob.glob(os.path.join(src, '*.tif')):
        import tifffile
        fs = sorted(glob.glob(os.path.join(src, '*.tif')))
        a = np.stack([tifffile.imread(f) for f in fs])
    else:
        import zarr
        z = zarr.open(src, mode='r')
        a = z if hasattr(z, 'shape') else z['0']
        if crop is not None:
            y0, x0, h, w = crop
            return np.asarray(a[:, y0:y0+h, x0:x0+w])
        a = np.asarray(a[:])
    if crop is not None:
        y0, x0, h, w = crop
        a = a[:, y0:y0+h, x0:x0+w]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--native-um', type=float, required=True, help='in-plane (x, y) voxel size of the render')
    ap.add_argument('--native-um-z', type=float, default=None,
                    help='voxel size along the depth; defaults to --native-um. Set it for an ANISOTROPIC stack, '
                         'otherwise Hecate receives planes at the wrong spacing.')
    ap.add_argument('--target-um', type=float, default=9.6)
    ap.add_argument('--z-center', type=float, required=True, help='window centre in NATIVE layer units')
    ap.add_argument('--nz', type=int, default=16)
    ap.add_argument('--crop', type=int, nargs=4, default=None, help='y0 x0 h w in native pixels')
    a = ap.parse_args()
    vol = load_src(a.src, a.crop)
    assert vol.ndim == 3 and vol.dtype == np.uint8, (vol.shape, vol.dtype)
    D, H, W = vol.shape
    nz_um = a.native_um_z if a.native_um_z else a.native_um
    step = a.target_um / nz_um                            # native layers per 9.6 um plane (depth voxel size)
    pos = a.z_center + (np.arange(a.nz) - (a.nz - 1) / 2.) * step
    if pos.min() < 0 or pos.max() > D - 1:
        raise SystemExit(f'z window {pos.min():.2f}..{pos.max():.2f} lies outside 0..{D-1}')
    lo = np.floor(pos).astype(int); frac = (pos - lo).astype(np.float32)
    hi = np.minimum(lo + 1, D - 1)
    zs = (vol[lo].astype(np.float32) * (1 - frac)[:, None, None] +
          vol[hi].astype(np.float32) * frac[:, None, None])
    s = a.native_um / a.target_um                          # < 1 => shrink
    nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
    interp = cv2.INTER_AREA if s < 0.95 else cv2.INTER_LINEAR
    out = np.empty((a.nz, nh, nw), np.uint8)
    for i in range(a.nz):
        out[i] = np.clip(cv2.resize(zs[i], (nw, nh), interpolation=interp), 0, 255).astype(np.uint8)
    import zarr
    g = zarr.open_group(a.out, mode='w')
    ch = (a.nz, 256, 256)
    try:                                    # zarr 2
        g.create_dataset('0', data=out, chunks=ch, dtype='u1', overwrite=True)
    except AttributeError:                  # zarr 3: create_dataset no longer exists
        d = g.create_array('0', shape=out.shape, chunks=ch, dtype='u1', overwrite=True)
        d[:] = out
    g.attrs['hecate_prep'] = dict(src=a.src, native_um=a.native_um, native_um_z=nz_um, target_um=a.target_um,
                                  z_center=a.z_center, nz=a.nz, z_positions=[float(p) for p in pos],
                                  crop=a.crop, xy_scale=s, interp=int(interp), shape=list(out.shape))
    print('written', a.out, out.shape, 'xy %.4f um -> scale %.6f' % (a.native_um, s),
          'z %.4f um -> step %.6f layers (= %.4f um)' % (nz_um, step, step * nz_um),
          'z %.2f..%.2f' % (pos.min(), pos.max()))


if __name__ == '__main__':
    main()
