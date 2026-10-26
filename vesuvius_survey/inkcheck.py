"""inkcheck - automatic diagnostics for a traced papyrus patch.

Two questions, answered per tile and for the whole patch:

1. Sheet quality: does the traced surface stay on ONE papyrus sheet?
   On a correct trace the flattened CT shows the papyrus weave: two families of straight
   fibres crossing at ~90 degrees. When a trace cuts across several layers, the render shows
   curved, swirling bands with one varying orientation instead. We measure, per tile, how much
   of the gradient energy falls into two orthogonal orientation bins (the "weave score").

2. Text-likeness of an ink prediction: does the predicted ink look like lines of writing?
   Greek text in the Herculaneum rolls runs in rows along the horizontal fibres, so a real ink
   map has a periodic row profile perpendicular to one fibre direction, with a line spacing of a
   few millimetres. Noise, texture responses and crack artefacts do not. We measure, per tile,
   the strongest autocorrelation peak of the ink row profile in the 2.5-8 mm line-spacing range,
   along both fibre directions, and compare it with the same statistic on a shuffled control.
   Optional: agreement between two models, and the reading-side vs blind-side contrast.

Usage (library):  from inkcheck import sheet_quality, text_likeness
Usage (CLI):      python inkcheck.py --ct CT.png|render.zarr[:layer] --ink a.tif [b.tif] [--blind c.tif [d.tif]]
                                     --voxel-um 8.64 --out report.json [--png report.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

TILE = 512


# ----------------------------------------------------------------------------- helpers
def _load_2d(spec: str) -> np.ndarray:
    p = spec.split(':')[0]
    if p.endswith('.zarr') or Path(p).is_dir():
        import zarr
        a = zarr.open_array(p, mode='r') if not Path(p, '0').exists() else zarr.open_group(p, mode='r')['0']
        layer = int(spec.split(':')[1]) if ':' in spec else a.shape[0] // 2
        return np.asarray(a[layer]).astype(np.float32)
    if p.lower().endswith(('.tif', '.tiff')):
        import tifffile
        return tifffile.imread(p).astype(np.float32)
    from PIL import Image
    return np.asarray(Image.open(p).convert('L')).astype(np.float32)


def _norm(a: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    v = a[valid] if valid is not None else a.ravel()
    lo, hi = np.percentile(v, [1, 99.5]) if v.size else (0, 1)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def _tiles(shape, tile=TILE):
    H, W = shape
    for y in range(0, max(H - tile // 2, 1), tile // 2):
        for x in range(0, max(W - tile // 2, 1), tile // 2):
            yield y, x, min(y + tile, H), min(x + tile, W)


# ----------------------------------------------------------------------------- 1. sheet quality
CROSS_THRESHOLD = 0.31   # calibrated on 9 hand-judged good and 7 bad traces (good 0.34-0.55, bad 0.13-0.28)


def crossness_map(ct: np.ndarray, win: int = 64, lam: float = 8.0, px_scale: float = 1.0):
    """Per-pixel 'fibre crossing' score of a flattened CT layer.

    A Gabor filter bank (12 orientations) measures oriented texture energy in a local window.
    Papyrus seen face-on shows two straight fibre families at ~90 deg, so the energy at the
    orthogonal orientation is a large fraction of the dominant one. A trace that cuts across
    layers shows a single swirling flow direction, and the orthogonal energy collapses.
    Returns (crossness, dominant_angle_deg, valid_mask). px_scale < 1 for downsampled input."""
    import cv2
    img = _norm(ct, ct > 0).astype(np.float32)
    lam = lam * px_scale
    win = max(8, int(win * px_scale))
    ths = np.arange(0, 180, 15)
    E = []
    for t in ths:
        k = cv2.getGaborKernel((int(4 * lam) | 1,) * 2, sigma=lam * 0.6, theta=np.radians(t),
                               lambd=lam, gamma=0.5, psi=0)
        k -= k.mean()
        r = cv2.filter2D(img, cv2.CV_32F, k)
        E.append(cv2.blur(r * r, (win, win)))
    E = np.stack(E)
    n = len(ths)
    i = np.argmax(E, 0)
    emax = np.take_along_axis(E, i[None], 0)[0]
    eorth = np.take_along_axis(E, ((i + n // 2) % n)[None], 0)[0]
    valid = cv2.blur((img > 0.02).astype(np.float32), (win, win)) > 0.9
    return eorth / np.maximum(emax, 1e-9), ths[i].astype(np.float32), valid


def weave_score(ct: np.ndarray, px_scale: float = 1.0) -> tuple[float, float]:
    """Median crossness of a tile and its dominant fibre angle (deg)."""
    c, ang, valid = crossness_map(ct, px_scale=px_scale)
    if valid.sum() < 100:
        return 0.0, 0.0
    a = np.radians(ang[valid] * 2)
    dom = (np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2) % 180
    return float(np.median(c[valid])), float(dom)


def sheet_quality(ct: np.ndarray, tile: int = TILE, px_scale: float = 1.0,
                  good_threshold: float = CROSS_THRESHOLD):
    c, ang, valid = crossness_map(ct, px_scale=px_scale)
    out = []
    t = int(tile * px_scale)
    for y0, x0, y1, x1 in _tiles(ct.shape, t):
        m = valid[y0:y1, x0:x1]
        if m.mean() < 0.6:
            continue
        s = float(np.median(c[y0:y1, x0:x1][m]))
        out.append({'y': int(y0 / px_scale), 'x': int(x0 / px_scale), 'cross': round(s, 4),
                    'good': s >= good_threshold})
    good = [t for t in out if t['good']]
    return {'tiles': out, 'good_fraction': round(len(good) / max(len(out), 1), 3),
            'median_cross': round(float(np.median([t['cross'] for t in out])) if out else 0.0, 4),
            'threshold': good_threshold}


# ----------------------------------------------------------------------------- 2. text-likeness
def _row_periodicity(ink: np.ndarray, angle_deg: float, px_um: float,
                     spacing_mm=(2.5, 8.0)) -> float:
    """Rotate so that the given fibre direction is horizontal, average along it, and return the
    strongest normalised autocorrelation peak of the row profile within the spacing range."""
    r = ndimage.rotate(ink, -angle_deg, reshape=True, order=1, mode='constant', cval=np.nan)
    with np.errstate(all='ignore'):
        prof = np.nanmean(r, axis=1)
    prof = prof[np.isfinite(prof)]
    if prof.size < 64:
        return 0.0
    prof = prof - ndimage.uniform_filter1d(prof, size=max(9, int(12000 / px_um)))  # remove trends > 12 mm
    prof = prof - prof.mean()
    if prof.std() < 1e-6:
        return 0.0
    ac = np.correlate(prof, prof, mode='full')[prof.size - 1:]
    ac = ac / ac[0]
    lo = int(spacing_mm[0] * 1000 / px_um)
    hi = min(int(spacing_mm[1] * 1000 / px_um), ac.size - 1)
    if hi <= lo:
        return 0.0
    seg = ac[lo:hi]
    # a real peak, not a monotone tail: value minus the minimum before it
    k = int(np.argmax(seg))
    trough = float(np.min(ac[:lo + k + 1][lo // 2:])) if lo + k + 1 > lo // 2 else 0.0
    return float(max(0.0, seg[k] - max(trough, -0.5)))


def _top_iou(a: np.ndarray, b: np.ndarray, q: float = 97) -> float:
    ta, tb = a > np.percentile(a, q), b > np.percentile(b, q)
    return float((ta & tb).sum() / max((ta | tb).sum(), 1))


def _contrast(a: np.ndarray) -> float:
    return float(np.percentile(a, 99) - np.percentile(a, 50))


def text_likeness(inks: list[np.ndarray], ct: np.ndarray, px_um: float,
                  blind: list[np.ndarray] | None = None, tile: int = 1024,
                  sheet: dict | None = None, rng_seed: int = 0):
    """Per-tile and overall text-likeness of an ink prediction.

    inks:  one or more ink maps of the side to test (same shape as ct)
    blind: the same models on the other side of the sheet (optional)
    """
    valid = ct > 0
    E = np.mean([_norm(i, valid) for i in inks], axis=0)
    Eb = np.mean([_norm(i, valid) for i in blind], axis=0) if blind else None
    rng = np.random.default_rng(rng_seed)
    rows = []
    for y0, x0, y1, x1 in _tiles(E.shape, tile):
        m = valid[y0:y1, x0:x1]
        if m.mean() < 0.6:
            continue
        c = _norm(ct[y0:y1, x0:x1], m)
        weave, ang = weave_score(c)
        e = np.where(m, E[y0:y1, x0:x1], np.nan)
        e = np.nan_to_num(e, nan=float(np.nanmean(e)))
        per = max(_row_periodicity(e, ang, px_um), _row_periodicity(e, ang + 90, px_um))
        # control: the same pixels shuffled in 64 px blocks (keeps blob statistics, breaks rows)
        bs = 64
        hh, ww = (e.shape[0] // bs) * bs, (e.shape[1] // bs) * bs
        blocks = e[:hh, :ww].reshape(hh // bs, bs, ww // bs, bs).swapaxes(1, 2).reshape(-1, bs, bs)
        blocks = blocks[rng.permutation(len(blocks))]
        sh = blocks.reshape(hh // bs, ww // bs, bs, bs).swapaxes(1, 2).reshape(hh, ww)
        ctrl = max(_row_periodicity(sh, ang, px_um), _row_periodicity(sh, ang + 90, px_um))
        row = {'y': y0, 'x': x0, 'weave': round(weave, 4), 'rows': round(per, 4),
               'rows_ctrl': round(ctrl, 4), 'contrast': round(_contrast(e), 4)}
        if len(inks) >= 2:
            row['agree'] = round(_top_iou(_norm(inks[0][y0:y1, x0:x1]), _norm(inks[1][y0:y1, x0:x1])), 4)
        if Eb is not None:
            row['side_ratio'] = round(_contrast(e) / max(_contrast(Eb[y0:y1, x0:x1]), 1e-6), 4)
        rows.append(row)
    good = [r for r in rows if r['weave'] >= CROSS_THRESHOLD] or rows
    summary = {
        'tiles': len(rows), 'good_tiles': len(good),
        'rows_max': max((r['rows'] - r['rows_ctrl'] for r in good), default=0.0),
        'rows_median': float(np.median([r['rows'] - r['rows_ctrl'] for r in good])) if good else 0.0,
    }
    if good and 'agree' in good[0]:
        summary['agree_median'] = float(np.median([r['agree'] for r in good]))
    if good and 'side_ratio' in good[0]:
        summary['side_ratio_median'] = float(np.median([r['side_ratio'] for r in good]))
    return {'summary': {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()},
            'tiles': rows}


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ct', required=True, help='CT layer: PNG/TIF, or render.zarr[:layer]')
    ap.add_argument('--ink', nargs='+', required=True, help='ink map(s) of the side to test')
    ap.add_argument('--blind', nargs='*', default=None, help='same model(s) on the other side')
    ap.add_argument('--voxel-um', type=float, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ct = _load_2d(a.ct)
    inks = [_load_2d(p) for p in a.ink]
    shape = inks[0].shape
    if ct.shape != shape:
        from PIL import Image
        ct = np.asarray(Image.fromarray(ct).resize(shape[::-1], Image.BILINEAR)).astype(np.float32)
    blind = [_load_2d(p) for p in a.blind] if a.blind else None
    rep = {'sheet': sheet_quality(ct), 'text': text_likeness(inks, ct, a.voxel_um, blind)}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({'sheet_good_fraction': rep['sheet']['good_fraction'], **rep['text']['summary']}))


if __name__ == '__main__':
    main()
