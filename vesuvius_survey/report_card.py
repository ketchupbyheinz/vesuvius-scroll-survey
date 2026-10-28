"""report_card.py - one diagnostic card per traced patch.

  python report_card.py PATCH_DIR --ct CT_MID(.tif|.png) --voxel-um 8.64 --out card.png [--json card.json]

PATCH_DIR holds the ink maps seed42_s6[_reverse].tif and dnative_s6[_reverse].tif (forward/reverse =
the two faces of the sheet). The card answers, left to right:
  1. Is the trace on one sheet?   CT layer with the sheet-quality map (green = papyrus weave, red = cross-cut)
  2. What do the ink models see?  agreed ink of the better face over the CT, only inside green tiles
  3. Numbers                      sheet good fraction, face contrast ratio, model agreement, letter-chain score
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

try:
    from .inkcheck import _load_2d, _norm, crossness_map, CROSS_THRESHOLD, _top_iou, _contrast
    from .textchain import chain_score
except ImportError:  # run as a plain script
    sys.path.insert(0, str(Path(__file__).parent))
    from inkcheck import _load_2d, _norm, crossness_map, CROSS_THRESHOLD, _top_iou, _contrast  # noqa
    from textchain import chain_score  # noqa

SCALE = 0.5  # cards are drawn at half resolution
TILE = 768   # face-consistency tile, px (~7 mm at 9 um)


def _font(size):
    for f in ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def face_consistency(tops: dict, m: np.ndarray, tile: int = TILE) -> dict:
    """Real writing sits on one face of the sheet everywhere in a patch. Split the patch into tiles
    (half-overlapping, >= 50 % on-sheet) and count tiles where the two models' agreement (IoU of their top-3 %
    pixels, thresholds taken over the whole patch) is > 0.1 and more than 3x the other face. face_consistency = (strong tiles on the majority face - on the other face) / tiles."""
    h, w = m.shape
    n = {'forward': 0, 'reverse': 0}
    total = 0
    for y in range(0, h - tile + 1, tile // 2):
        for x in range(0, w - tile + 1, tile // 2):
            sl = (slice(y, y + tile), slice(x, x + tile))
            if m[sl].mean() < 0.5:
                continue
            total += 1
            iou = {s: tops[s][0][sl].sum() / max((tops[s][1][sl] & m[sl]).sum(), 1) for s in tops}
            for s, o in (('forward', 'reverse'), ('reverse', 'forward')):
                if iou[s] > 0.1 and iou[s] > 3 * iou[o]:
                    n[s] += 1
    major = max(n, key=n.get)
    minor = 'reverse' if major == 'forward' else 'forward'
    face = major if n[major] > n[minor] else 'tie'      # no face wins: no side can be named
    return {'face_consistency': round((n[major] - n[minor]) / max(total, 1), 3), 'consistent_face': face,
            'tiles': total, 'strong_tiles': n[major], 'strong_tiles_other_face': n[minor]}


def ink_on_dark(ct: np.ndarray, agreed: np.ndarray, m: np.ndarray) -> dict:
    """Crack check. Voids, cracks and gaps between layers are dark in the CT and can trigger ink models.
    Returns the share of the agreed-ink pixels (best face) that fall in the darkest 10 % of on-sheet CT (chance = 0.10)
    and their mean CT z-score. On the PHerc. 343 known text: 0.09 and 0.11, z +0.14 and +0.02."""
    if agreed.sum() == 0:
        return {'ink_on_dark': None, 'ink_ct_z': None}
    z = (ct - ct[m].mean()) / (ct[m].std() + 1e-6)
    return {'ink_on_dark': round(float(np.mean(z[agreed] < np.quantile(z[m], 0.1))), 3),
            'ink_ct_z': round(float(z[agreed].mean()), 3)}


def sheet_masks(ct: np.ndarray):
    """Sheet check of a flattened CT layer: (smoothed crossness, valid, good, m). ``good`` is the green part of the card;
    ``m`` is the mask the scores use (``good``, or all valid pixels when under 5 % is green)."""
    import cv2
    cross, _, valid = crossness_map(ct)
    cross = cv2.blur(np.where(valid, cross, 0).astype(np.float32), (129, 129)) / np.maximum(cv2.blur(valid.astype(np.float32), (129, 129)), 1e-3)
    good = (cross >= CROSS_THRESHOLD) & valid
    return cross, valid, good, (good if good.mean() > 0.05 else valid)


def agreed_top3(a: np.ndarray, b: np.ndarray, m: np.ndarray):
    """The card's agreed ink: both models in their top 3 % of ``m``. Returns (agreed, either, normalised a, normalised b)."""
    na, nb = _norm(a, m), _norm(b, m)
    ta, tb = na >= np.quantile(na[m], 0.97), nb >= np.quantile(nb[m], 0.97)
    return ta & tb & m, ta | tb, na, nb


def analyse(d: Path, ct: np.ndarray, um: float) -> dict:
    sides = {}
    for side, suf in (('forward', ''), ('reverse', '_reverse')):
        a = tifffile.imread(d / f'seed42_s6{suf}.tif').astype(np.float32)
        b = tifffile.imread(d / f'dnative_s6{suf}.tif').astype(np.float32)
        h, w = min(a.shape[0], ct.shape[0]), min(a.shape[1], ct.shape[1])
        sides[side] = (a[:h, :w], b[:h, :w])
    ct = ct[:h, :w]
    cross, valid, good, m = sheet_masks(ct)
    res = {'sheet_good_fraction': round(float(good.sum() / max(valid.sum(), 1)), 3)}
    stats = {}
    tops = {}
    for side, (a, b) in sides.items():
        agreed, either, na, nb = agreed_top3(a, b, m)
        tops[side] = (agreed, either)
        e = (na + nb) / 2
        stats[side] = {
            'agree_top3': round(_top_iou(na[m], nb[m]), 4),
            'contrast': round(_contrast(e[m]), 4),
            'chain': round(chain_score(np.where(m, e, 0), np.where(m, ct, 0), um)['chain'], 3),
            'map': e,
        }
    best = max(stats, key=lambda s: (stats[s]['agree_top3'], stats[s]['chain']))
    other = 'reverse' if best == 'forward' else 'forward'
    res.update({
        'face': best,
        'agree_top3': stats[best]['agree_top3'],
        'agree_top3_other_face': stats[other]['agree_top3'],
        'face_contrast_ratio': round(stats[best]['contrast'] / max(stats[other]['contrast'], 1e-6), 3),
        'chain': stats[best]['chain'],
        'chain_other_face': stats[other]['chain'],
        'face_asymmetry': round(stats[best]['agree_top3'] / max(stats[other]['agree_top3'], 1e-3), 3),
    })
    res.update(face_consistency(tops, m))
    res.update(ink_on_dark(ct, tops[best][0], m))
    return res, ct, cross, valid, good, stats[best]['map']


def draw(res, ct, cross, valid, good, ink, title, out):
    def small(x, mode='L'):
        im = Image.fromarray(x)
        return im.resize((max(1, int(im.width * SCALE)), max(1, int(im.height * SCALE))), Image.BILINEAR)
    c = _norm(ct, ct > 0)
    base = (np.stack([c] * 3, -1) * 255).astype(np.uint8)
    q = base.astype(np.float32).copy()
    tint = np.zeros_like(q)
    tint[good] = (40, 200, 70)
    tint[valid & ~good] = (220, 50, 50)
    q = np.where(valid[..., None], q * 0.65 + tint * 0.35, q * 0.3)
    p1 = small(q.clip(0, 255).astype(np.uint8))
    e = np.clip((_norm(ink, good if good.any() else valid) - 0.5) / 0.4, 0, 1) * good
    ov = (c[..., None] * 0.45) * (1 - e[..., None]) + np.array([1, .88, .35]) * e[..., None]
    p2 = small((ov * 255).clip(0, 255).astype(np.uint8))
    W, H = p1.width + p2.width + 420, max(p1.height, 300) + 60
    card = Image.new('RGB', (W, H), (18, 18, 20))
    card.paste(p1, (10, 50)); card.paste(p2, (20 + p1.width, 50))
    dr = ImageDraw.Draw(card)
    dr.text((10, 12), title, fill=(240, 240, 240), font=_font(22))
    dr.text((10, 34), 'sheet check (green = papyrus weave, red = cross-cut)', fill=(170, 170, 170), font=_font(13))
    dr.text((20 + p1.width, 34), f'agreed ink, {res["face"]} face, green tiles only', fill=(170, 170, 170), font=_font(13))
    x0, y = 30 + p1.width + p2.width, 60
    lines = [
        ('sheet good fraction', f'{res["sheet_good_fraction"]:.2f}'),
        ('face asymmetry (agreement best / other face)', f'{res["face_asymmetry"]:.2f}'),
        ('model agreement (top 3%)', f'{res["agree_top3"]:.3f}  (other face {res["agree_top3_other_face"]:.3f})'),
        ('face contrast ratio', f'{res["face_contrast_ratio"]:.2f}'),
        ('face consistency (tiles)', f'{res["face_consistency"]:.2f}  ({res["strong_tiles"]} vs {res["strong_tiles_other_face"]} of {res["tiles"]}, {res["consistent_face"]})'),
        ('ink on darkest 10% of CT (chance 0.10)', f'{res["ink_on_dark"]:.2f}' if res.get('ink_on_dark') is not None else '-'),
        ('letter-chain score', f'{res["chain"]:.2f}  (other face {res["chain_other_face"]:.2f})'),
    ]
    for k, v in lines:
        dr.text((x0, y), k, fill=(170, 170, 170), font=_font(15)); y += 20
        dr.text((x0, y), v, fill=(250, 250, 250), font=_font(19)); y += 36
    if 'verdict' in res:
        dr.text((x0, y + 10), res['verdict'], fill=(255, 220, 90), font=_font(18))
    card.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('patch', type=Path)
    ap.add_argument('--ct', required=True)
    ap.add_argument('--voxel-um', type=float, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--json')
    ap.add_argument('--title', default=None)
    a = ap.parse_args()
    ct = _load_2d(a.ct)
    res, ct, cross, valid, good, ink = analyse(a.patch, ct, a.voxel_um)
    draw(res, ct, cross, valid, good, ink, a.title or a.patch.name, a.out)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == '__main__':
    main()
