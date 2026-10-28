"""Sharp-scan check: does a patch's agreed ink sit on air gaps when the same place is seen in a sharper scan?

Ink models at ~9 um can fire on gaps between layers, cracks and voids. Some scrolls also have a 2.4 um scan. This
module lines the two scans up, carries a traced mesh across, and measures whether the agreed-ink pixels of the
report card fall on dark (gap) pixels of the sharper scan more often than shifted copies of the same ink mask do.

Steps (each is a CLI subcommand):

``register``  find where a voxel of the 9 um scan sits in the sharper scan. The scans of one scroll are close to aligned
              in physical units in x and y; the z offset is unknown. A coarse search matches one low-resolution slice
              of the sharp scan against every slice of the 9 um scan (all rotations), then a local search refines z and
              x/y at ~19 um around the seed.
``map``       write a copy of a tifxyz mesh in the sharp scan's coordinates (level L), ready for ``render_patch --level L``.
``gapcheck``  compare agreed ink (two models, top 3 %, on sheet-check tiles, the card's definition) with the sharp
              render's dark pixels (bottom quartile per layer), for each half of the layer stack, against 48 shifted
              copies of the ink mask.

Example (PHerc. 846A, numbers in the README)::

    B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0846A/volumes
    V9=$B/20250728152254-9.362um-1.2m-113keV-masked.zarr; V2=$B/20260319102732-2.403um-0.2m-77keV-masked.zarr
    python -m vesuvius_survey.sharpscan register --vol9 $V9 --vol2 $V2 --um9 9.362 --um2 2.403 --seed 2601 2784 3851
    # offset: from `register` (the README renders used 2.6 8.6 -7291.3; a re-run gives 7.8 10.7 -7293.9, 13 um away)
    python -m vesuvius_survey.sharpscan map MESH MESH_ON_SHARP_L1 --k 3.89596 --offset 2.6 8.6 -7291.3 --level 1
    python -m vesuvius_survey.render_patch MESH_ON_SHARP_L1 $V2 sharp.zarr --rows 0 4363 --cols 0 4363 --layers 32 --level 1
    python -m vesuvius_survey.sharpscan gapcheck PATCH_DIR sharp.zarr --face reverse
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


# ---------------------------------------------------------------- reading
def read_box(volume_url: str, level: int | str, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """A box [z0:z1, y0:y1, x0:x1] of one multiscale level. Raises if the box leaves the array (callers rely on the box
    starting at (z0, y0, x0)); ``whole_level`` reads a full level."""
    import zarr
    a = zarr.open_group(volume_url, mode="r")[str(level)]
    Z, Y, X = a.shape
    if min(z0, y0, x0) < 0 or z1 > Z or y1 > Y or x1 > X:
        raise ValueError(f"box z{z0}:{z1} y{y0}:{y1} x{x0}:{x1} leaves level {level} of shape {a.shape}")
    return np.asarray(a[z0:z1, y0:y1, x0:x1])


def whole_level(volume_url: str, level: int | str) -> np.ndarray:
    import zarr
    return np.asarray(zarr.open_group(volume_url, mode="r")[str(level)][:])


# ---------------------------------------------------------------- registration
def _ncc_best(image: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]]:
    import cv2
    r = cv2.matchTemplate(image.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    return float(mx), loc


def coarse_z(vol9: str, vol2: str, um9: float, um2: float, sharp_slice: int, step: int = 3, seed: int = 0) -> dict:
    """Which 9 um slice matches slice ``sharp_slice`` (level-0 index) of the sharp scan, and at what rotation.

    Works at ~150 um (9 um level 4, sharp level 5 resized). The 9 um slice is padded with low noise so that windows
    over the padding cannot score."""
    import cv2
    a9 = whole_level(vol9, 4).astype(np.float32)
    s5 = sharp_slice // 32
    q = whole_level(vol2, 5)[s5].astype(np.float32)
    f = (um2 * 32) / (um9 * 16)
    q = cv2.GaussianBlur(cv2.resize(q, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (0, 0), 1)
    rng = np.random.default_rng(seed)
    H, W = a9.shape[1:]
    best = (-1.0, 0, 0)
    frac = (a9 > 0).reshape(len(a9), -1).mean(1)
    for z in range(0, len(a9), step):
        if frac[z] < 0.1:
            continue
        P = rng.normal(0, 5, (H * 2, W * 2)).astype(np.float32)
        P[H // 2:H // 2 + H, W // 2:W // 2 + W] = cv2.GaussianBlur(a9[z], (0, 0), 1)
        for ang in range(0, 360, 4):
            R = cv2.warpAffine(q, cv2.getRotationMatrix2D((q.shape[1] / 2, q.shape[0] / 2), ang, 1), q.shape[::-1])
            mx, _ = _ncc_best(P, R)
            if mx > best[0]:
                best = (mx, z, ang)
    return {"score": round(best[0], 3), "z9": best[1] * 16 + 8, "sharp_z": s5 * 32 + 16, "rotation_deg": best[2]}


def local_register(vol9: str, vol2: str, um9: float, um2: float, seed_xyz, sharp_z_guess: float,
                   search_z: int = 80, search_xy: int = 100) -> dict:
    """Refine the sharp-scan position of a 9 um voxel (x, y, z) at ~19 um (9 um level 1 vs sharp level 3).

    Assumes no rotation and x/y aligned in physical units to within ``search_xy`` level-3 pixels (true for PHerc. 846A
    and 1203). Returns the level-0 sharp coordinates of the seed and the offset with p_sharp = p_9um * um9 / um2 + offset.
    Coordinates are voxel centres: voxel i of level L covers level-0 voxels i*2**L .. (i+1)*2**L - 1, centre
    (i + 0.5) * 2**L - 0.5."""
    import cv2
    X, Y, Z = (float(v) for v in seed_xyz)
    k = um9 / um2
    c1 = (int(Z / 2), int(Y / 2), int(X / 2))
    g3 = (int(sharp_z_guess / 8), int(Y * k / 8), int(X * k / 8))
    half = 200
    T = read_box(vol9, 1, c1[0] - 2, c1[0] + 3, c1[1] - half, c1[1] + half, c1[2] - half, c1[2] + half).mean(0)
    T = cv2.GaussianBlur(T.astype(np.float32), (0, 0), 1)
    span = half + search_xy
    B = read_box(vol2, 3, g3[0] - search_z, g3[0] + search_z, g3[1] - span, g3[1] + span, g3[2] - span, g3[2] + span)
    f = (um2 * 8) / (um9 * 2)
    res = []
    for z in range(B.shape[0]):
        I = cv2.GaussianBlur(cv2.resize(B[z].astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (0, 0), 1)
        if I.std() < 1:
            continue
        mx, loc = _ncc_best(I, T)
        res.append((mx, z, loc))
    res.sort(key=lambda r: -r[0])
    mx, z, loc = res[0]
    # the template centre (level-1 voxel c1) matched sharp level-3 position (x3, y3, z3); both as level-0 voxel centres
    x3 = (loc[0] + half) / f + g3[2] - span
    y3 = (loc[1] + half) / f + g3[1] - span
    z3 = z + g3[0] - search_z
    sharp_c = [(v + 0.5) * 8 - 0.5 for v in (x3, y3, z3)]
    nine_c = [(v + 0.5) * 2 - 0.5 for v in (c1[2], c1[1], c1[0])]
    offset = [sc - nc * k for sc, nc in zip(sharp_c, nine_c)]
    return {"score": round(mx, 3),
            "best_other_z_score": round(max([r[0] for r in res if abs(r[1] - z) > 6] or [0.0]), 3),
            "sharp_xyz": [round(v * k + o, 1) for v, o in zip((X, Y, Z), offset)],
            "k": k, "offset": [round(o, 1) for o in offset]}


# ---------------------------------------------------------------- mesh mapping
def map_mesh(src: Path, dst: Path, k: float, offset, level: int = 0) -> dict:
    """Copy a tifxyz mesh into sharp-scan coordinates of ``level``: p' = (p * k + offset) / 2**level.

    The mesh scale is divided by the same factor, so one render pixel is one voxel of that level."""
    import tifffile
    dst.mkdir(parents=True, exist_ok=True)
    f = k / 2 ** level
    for c, o in zip("xyz", offset):
        a = tifffile.imread(src / f"{c}.tif").astype(np.float32)
        b = a * f + np.float32(o / 2 ** level)
        b[a == -1] = -1
        tifffile.imwrite(dst / f"{c}.tif", b)
    meta = json.loads((src / "meta.json").read_text())
    meta["scale"] = [s / f for s in meta["scale"]]
    meta["sharpscan_mapping"] = {"k": k, "offset": list(offset), "level": level}
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


# ---------------------------------------------------------------- gap check
def gap_fraction(stack: np.ndarray, layers, q: float = 25) -> np.ndarray:
    """Per pixel, the share of ``layers`` where the sharp render is in its layer's darkest ``q`` %."""
    out = []
    for k in layers:
        a = stack[k]
        nz = a[a > 0]
        out.append(a < np.percentile(nz, q) if nz.size else np.zeros(a.shape, bool))
    return np.mean(out, 0)


def gap_ratio(gap: np.ndarray, ink: np.ndarray, valid: np.ndarray) -> float:
    """Mean gap fraction under the ink mask over that outside it (valid pixels only)."""
    inside, outside = gap[ink & valid], gap[~ink & valid]
    if not inside.size or not outside.size or outside.mean() == 0:
        return float("nan")
    return float(inside.mean() / outside.mean())


def shifted_ratios(gap: np.ndarray, ink: np.ndarray, valid: np.ndarray, n: int = 7, frac: float = 0.3) -> list[float]:
    """Gap ratios for the ink mask rolled by a grid of n x n shifts up to ``frac`` of the size (zero shift left out)."""
    H, W = ink.shape
    out = []
    for dy in np.linspace(-frac, frac, n) * H:
        for dx in np.linspace(-frac, frac, n) * W:
            if dy == 0 and dx == 0:
                continue
            r = gap_ratio(gap, np.roll(np.roll(ink, int(dy), 0), int(dx), 1), valid)
            if np.isfinite(r):
                out.append(r)
    return out


def agreed_ink(patch_dir: Path, face: str, models=("seed42", "dnative")):
    """The report card's agreed ink: both models' top 3 % on sheet-check pixels of ``face`` (forward/reverse)."""
    import tifffile
    from .report_card import agreed_top3, sheet_masks
    ct = tifffile.imread(patch_dir / "ct_mid.tif").astype(np.float32)
    suf = "" if face == "forward" else "_reverse"
    a, b = (tifffile.imread(patch_dir / f"{m}_s6{suf}.tif").astype(np.float32) for m in models)
    h, w = min(a.shape[0], b.shape[0], ct.shape[0]), min(a.shape[1], b.shape[1], ct.shape[1])
    _, _, _, m = sheet_masks(ct[:h, :w])
    agreed = agreed_top3(a[:h, :w], b[:h, :w], m)[0]
    return agreed, m


def gap_check(patch_dir: Path, sharp_render: str, face: str) -> dict:
    import cv2
    import zarr
    ink, good = agreed_ink(Path(patch_dir), face)
    A = np.asarray(zarr.open_array(sharp_render, mode="r")).astype(np.float32)
    L, H, W = A.shape
    sy, sx = H / ink.shape[0], W / ink.shape[1]
    if abs(sy / sx - 1) > 0.01:
        raise ValueError(f"sharp render {H}x{W} is not a uniform scaling of the {ink.shape} patch (x{sy:.3f} vs x{sx:.3f}); "
                         "render the full mapped mesh (rows/cols = patch size x k / 2**level)")
    inkU = cv2.resize(ink.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    goodU = cv2.resize(good.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) & (A[L // 2] > 0)
    out = {"face": face, "agreed_ink_px": int(ink.sum()), "scale": round(sx, 4), "halves": {}}
    # render_patch layer k samples at (k - (L-1)/2) * step along the mesh normal: the low half is the -normal side
    for name, ks in (("minus_normal_half", range(0, L // 2)), ("plus_normal_half", range(L // 2, L))):
        g = gap_fraction(A, ks)
        r = gap_ratio(g, inkU, goodU)
        sh = shifted_ratios(g, inkU, goodU)
        out["halves"][name] = {"ratio": round(r, 3), "shifted_median": round(float(np.median(sh)), 3),
                               "shifted_max": round(float(np.max(sh)), 3),
                               "shifted_below": int(sum(r > s for s in sh)), "n_shifted": len(sh)}
    return out


# ---------------------------------------------------------------- CLI
def _vol(scroll: str, name: str) -> str:
    return f"{BUCKET}/{scroll}/volumes/{name}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--vol9", required=True, help="URL of the ~9 um volume (OME-Zarr)")
    r.add_argument("--vol2", required=True, help="URL of the sharp volume (OME-Zarr)")
    r.add_argument("--um9", type=float, required=True)
    r.add_argument("--um2", type=float, required=True)
    r.add_argument("--seed", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"), help="9 um voxel")
    r.add_argument("--sharp-z-guess", type=float, help="skip the coarse search")
    r.add_argument("--coarse-slice", type=int, default=7680, help="sharp level-0 slice for the coarse search")
    m = sub.add_parser("map")
    m.add_argument("src", type=Path)
    m.add_argument("dst", type=Path)
    m.add_argument("--k", type=float, required=True)
    m.add_argument("--offset", type=float, nargs=3, required=True)
    m.add_argument("--level", type=int, default=0)
    g = sub.add_parser("gapcheck")
    g.add_argument("patch_dir", type=Path, help="survey patch folder (ct_mid.tif, seed42_s6*.tif, dnative_s6*.tif)")
    g.add_argument("sharp_render", help="render_patch output of the mapped mesh on the sharp scan")
    g.add_argument("--face", choices=["forward", "reverse"], required=True, help="the card's consistent face")
    a = ap.parse_args(argv)
    if a.cmd == "register":
        out = {}
        zg = a.sharp_z_guess
        if zg is None:
            c = coarse_z(a.vol9, a.vol2, a.um9, a.um2, a.coarse_slice)
            out["coarse"] = c
            zg = c["sharp_z"] + (a.seed[2] - c["z9"]) * a.um9 / a.um2
        out["local"] = local_register(a.vol9, a.vol2, a.um9, a.um2, a.seed, zg)
        print(json.dumps(out, indent=1))
    elif a.cmd == "map":
        map_mesh(a.src, a.dst, a.k, a.offset, a.level)
    else:
        print(json.dumps(gap_check(a.patch_dir, a.sharp_render, a.face), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
