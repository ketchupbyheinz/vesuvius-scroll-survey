"""Keep VC3D's patch-and-stitch pipeline on one winding.

VC3D builds large surfaces from many small patches: ``vc_grow_seg_from_seed`` in ``expansion`` mode grows new patches
from random points of the existing ones, and ``vc_grow_seg_from_segments`` stitches them by letting overlapping
patches vote. In a tightly rolled scroll the neighbouring windings sit only ~10-15 voxels apart, and expansion
patches drift onto them; once a patch is on a neighbouring winding, every patch seeded from it is too. On PHerc. 343
only 16-19 % of unsteered expansion patches stayed on the seed's winding, and the stitch stopped at ~3 cm².

Subcommands:

``same-winding``  start from the seed patch; a patch that overlaps the accepted ones with a median offset along their
                  normals below ``--thr`` voxels is accepted, one that overlaps them further away is rejected (it runs
                  parallel, a winding or a face away), the rest stay undecided. With ``--move`` rejected patches are
                  moved out of the folder, so expansion stops seeding from them and the stitcher never sees them.
``winding``       the same job with the ink vote's sheet grouping (inkvote.sheet_numbers): patches are cut into ~1 mm
                  tiles and smooth pieces; pieces lying on top of each other join one sheet, but two groups that are
                  clearly apart anywhere they overlap never join. A patch is accepted when >= ``--accept`` of its tiles
                  are on the seed's sheet, rejected when it is mostly on another shared sheet, undecided otherwise.
                  Acceptance cannot creep across touching windings the way a chain of small steps does (on PHerc. 1545
                  the chain filter let the patch set spread over many windings).
``gate``          render each patch's mid surface and tag patches that fail the report card's sheet check
                  ``"defective"`` in meta.json (the stitcher skips them).
``steer``         rounds of expansion patches (several single-thread VC3D processes at once, each killed if it hangs),
                  each followed by ``winding --move`` (``--method same-winding`` for the older filter); undecided
                  patches are parked between rounds so that new patches are only seeded from accepted ones.
``stitch``        run ``vc_grow_seg_from_segments`` on the folder with a time limit; it keeps its latest snapshot.

The stitcher needs a local volume only for its shape and voxel size: ``stub`` writes one (``0/.zarray`` + meta.json).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .render_patch import load_mesh, node_normals
    from . import inkvote
except ImportError:  # plain script (laptop tools folder)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_patch import load_mesh, node_normals  # noqa
    import inkvote  # noqa


# ---------------------------------------------------------------------------------------------- same winding
def _points_normals(patch: Path):
    pts, valid, _ = load_mesh(patch)
    nrm = node_normals(pts, valid)
    ok = valid & np.isfinite(nrm).all(-1)
    return pts[ok].astype(np.float64), nrm[ok].astype(np.float64)


def same_winding(paths: Path, seed: str, thr: float = 4.0, min_overlap: int = 40, lateral: float = 12.0,
                 previous: dict | None = None) -> dict:
    """Classify every auto_grown_* patch in ``paths`` as accepted / rejected / undecided relative to ``seed``."""
    from scipy.spatial import cKDTree
    names = sorted(p.name for p in paths.glob('auto_grown_*') if (p / 'meta.json').exists() and (p / 'z.tif').exists())
    state = {n: 'undecided' for n in names}
    for n, v in (previous or {}).items():
        if n in state and v == 'accepted':
            state[n] = 'accepted'
    state[seed] = 'accepted'
    cache: dict = {}

    def get(n):
        if n not in cache:
            cache[n] = _points_normals(paths / n)
        return cache[n]

    offsets: dict = {}
    changed = True
    while changed:
        changed = False
        acc = [n for n in names if state[n] == 'accepted']
        S = np.concatenate([get(n)[0] for n in acc])
        N = np.concatenate([get(n)[1] for n in acc])
        tree = cKDTree(S)
        for n in names:
            if state[n] != 'undecided':
                continue
            P = get(n)[0][::2]
            dist, idx = tree.query(P, distance_upper_bound=60)
            m = np.isfinite(dist)
            if m.sum() < min_overlap:
                continue
            v = P[m] - S[idx[m]]
            o = (v * N[idx[m]]).sum(1)
            lat = np.linalg.norm(v - o[:, None] * N[idx[m]], axis=1)
            k = lat < lateral
            if k.sum() < min_overlap:
                continue
            med = float(np.median(np.abs(o[k])))
            offsets[n] = round(med, 2)
            state[n] = 'accepted' if med < thr else 'rejected'
            changed = True
    counts = {s: sum(v == s for v in state.values()) for s in ('accepted', 'rejected', 'undecided')}
    return {'seed': seed, 'thr': thr, 'counts': counts, 'state': state, 'median_abs_offset': offsets}


def winding_filter(paths: Path, seed: str, accept: float = 0.8) -> dict:
    """Classify every auto_grown_* patch in ``paths`` by sheet relative to ``seed`` (see ``winding`` above)."""
    names = sorted(p.name for p in paths.glob('auto_grown_*') if (p / 'meta.json').exists() and (p / 'z.tif').exists())
    nodes = [inkvote.mesh_nodes(paths / n) for n in names]
    T, grids, o_p, sheet, counts = inkvote.sheet_numbers(nodes, names.index(seed))
    patches_in = {}
    for k in range(len(names)):
        for sh in np.unique(sheet[grids[k][grids[k] >= 0]]).tolist():
            patches_in[sh] = patches_in.get(sh, 0) + 1
    state, frac, main = {}, {}, {}
    for k, n in enumerate(names):
        t = sheet[grids[k][grids[k] >= 0]]
        if len(t) == 0:
            state[n] = 'undecided'
            continue
        # tiles on a sheet that other patches share, or that lies clearly apart from the seed's sheet
        shared = np.array([patches_in[x] > 1 or x in T['apart_from_0'] for x in t.tolist()])
        f0 = float((t == 0).mean())
        frac[n] = round(f0, 3)
        v, c = np.unique(t, return_counts=True)
        main[n] = int(v[np.argmax(c)])
        if n == seed or f0 >= accept:
            state[n] = 'accepted'
        elif shared.mean() >= 0.5:
            state[n] = 'rejected'
        else:
            state[n] = 'undecided'
    out = {s: sum(v == s for v in state.values()) for s in ('accepted', 'rejected', 'undecided')}
    return {'seed': seed, 'method': 'sheets', 'accept': accept, 'counts': out, 'state': state,
            'share_on_seed_sheet': frac, 'main_sheet': main, 'tiles_per_sheet': dict(list(counts.items())[:20]),
            'seed_sheet_area_cm2': round(counts.get(0, 0) * (inkvote.TILE_NODES * 20 * 9.362e-4) ** 2, 2)}


def move_rejected(paths: Path, report: dict, off: Path) -> int:
    off.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, s in report['state'].items():
        if s == 'rejected' and (paths / name).exists():
            shutil.move(str(paths / name), str(off / name))
            n += 1
    return n


# ---------------------------------------------------------------------------------------------- gate
def gate(paths: Path, ct_url: str, thr: float = 0.4, db: Path | None = None) -> dict:
    """One-layer render of every patch + the card's sheet check; tag patches below ``thr`` "defective"."""
    import tempfile
    import tifffile
    import zarr
    from .report_card import sheet_masks
    from . import render_patch
    db = db or paths.parent / 'gate.json'
    g = json.loads(db.read_text()) if db.exists() else {}
    for p in sorted(paths.glob('auto_grown_*')):
        if p.name in g or not (p / 'x.tif').exists():
            continue
        x = tifffile.imread(p / 'x.tif')
        R, C = x.shape[0] * 20, x.shape[1] * 20
        with tempfile.TemporaryDirectory() as tmp:
            render_patch.main([str(p.resolve()), ct_url, f'{tmp}/r.zarr', '--rows', '0', str(R), '--cols', '0', str(C),
                               '--layers', '1', '--tile', '512', '--workers', '3'])
            ct = np.asarray(zarr.open_array(f'{tmp}/r.zarr', mode='r')[0]).astype(np.float32)
        _, valid, good, _ = sheet_masks(ct)
        frac = float(good.sum() / max(valid.sum(), 1))
        meta = json.loads((p / 'meta.json').read_text())
        tags = meta.get('tags') or {}
        if isinstance(tags, list):
            tags = {t: {} for t in tags}
        if frac < thr:
            tags['defective'] = {'by': 'vesuvius_survey.patchset gate', 'sheet_good_fraction': round(frac, 3)}
        meta['tags'] = tags
        (p / 'meta.json').write_text(json.dumps(meta, indent=1))
        g[p.name] = {'sheet_good_fraction': round(frac, 3), 'defective': frac < thr}
        db.write_text(json.dumps(g, indent=1))
    return g


# ---------------------------------------------------------------------------------------------- VC3D runs
def expand(vc_seed: str, stitch: Path, pred_url: str, n_parallel: int, total: int, rng_base: int,
           max_hours: float = 12, hang_minutes: float = 12) -> int:
    """Run ``total`` expansion patches into stitch/paths, ``n_parallel`` at a time (params stitch/expand.json)."""
    logs = stitch / 'logs'
    logs.mkdir(exist_ok=True)
    t_end = time.time() + max_hours * 3600
    running, i, done = [], 0, 0
    with open(logs / 'progress.txt', 'a') as prog:
        while (i < total or running) and time.time() < t_end:
            while i < total and len(running) < n_parallel:
                s = rng_base + i
                lf = open(logs / f'exp_{s}.log', 'w')
                p = subprocess.Popen(vc_seed.split() + ['-v', pred_url, '-t', str(stitch / 'paths'), '-p', str(stitch / 'expand.json'),
                                                        '--rng-seed', str(s)], stdout=lf, stderr=subprocess.STDOUT)
                running.append((p, s, lf, time.time()))
                i += 1
            time.sleep(5)
            for r in running[:]:
                if r[0].poll() is None and time.time() - r[3] > hang_minutes * 60:
                    r[0].kill()
                    r[0].wait()
                    prog.write(f"{time.strftime('%H:%M:%S')} seed {r[1]} killed after {hang_minutes:g} min\n")
                if r[0].poll() is not None:
                    r[2].close()
                    running.remove(r)
                    done += 1
                    txt = (logs / f'exp_{r[1]}.log').read_text(errors='ignore')
                    sv = [ln for ln in txt.splitlines() if 'saving' in ln]
                    prog.write(f"{time.strftime('%H:%M:%S')} seed {r[1]} rc {r[0].returncode} {sv[-1] if sv else 'no save'}\n")
                    prog.flush()
        for p, *_ in running:
            p.kill()
    return done


def steer(vc_seed: str, stitch: Path, pred_url: str, seed: str, n_parallel: int, per_round: int, hours: float,
          rng_base: int, thr: float = 4.0, method: str = 'winding') -> None:
    t_end = time.time() + hours * 3600
    rep_path = stitch / 'same_report.json'
    with open(stitch / 'logs' / 'steer.txt', 'a') as log:
        r = 0
        park = stitch / 'parked'
        while True:
            # parked (undecided) patches come back for every test, but expansion only ever sees accepted ones:
            # undecided patches that later turn out to be a winding away would otherwise seed whole off-winding families
            if park.exists():
                for q in park.iterdir():
                    shutil.move(str(q), str(stitch / 'paths' / q.name))
            prev = json.loads(rep_path.read_text())['state'] if rep_path.exists() else None
            rep = (winding_filter(stitch / 'paths', seed) if method == 'winding'
                   else same_winding(stitch / 'paths', seed, thr, previous=prev))
            moved = move_rejected(stitch / 'paths', rep, stitch / 'off')
            park.mkdir(exist_ok=True)
            for name, st in rep['state'].items():
                if st == 'undecided' and (stitch / 'paths' / name).exists():
                    shutil.move(str(stitch / 'paths' / name), str(park / name))
            rep_path.write_text(json.dumps(rep, indent=1))
            log.write(f"{time.strftime('%H:%M:%S')} round {r} {json.dumps(rep['counts'])} moved {moved}\n")
            log.flush()
            if time.time() >= t_end:
                for q in park.iterdir():          # leave every non-rejected patch in paths/ for the stitcher
                    shutil.move(str(q), str(stitch / 'paths' / q.name))
                break
            expand(vc_seed, stitch, pred_url, n_parallel, per_round, rng_base + r * per_round, (t_end - time.time()) / 3600)
            r += 1


def stitch_run(vc_segments: str, stitch: Path, seed: str, minutes: float = 60) -> str:
    (stitch / 'traces').mkdir(exist_ok=True)
    with open(stitch / 'logs' / 'trace.log', 'w') as lf:
        p = subprocess.Popen(vc_segments.split() + ['-v', str(stitch / 'stub'), '-s', str(stitch / 'paths'),
                                                    '-t', str(stitch / 'traces'), '-p', str(stitch / 'trace.json'),
                                                    '--src-segment', str(stitch / 'paths' / seed)],
                             stdout=lf, stderr=subprocess.STDOUT)
        try:
            return f'rc {p.wait(timeout=minutes * 60)}'
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            return f'stopped after {minutes:g} min'


def write_stub(stitch: Path, shape, voxelsize: float) -> None:
    (stitch / 'stub' / '0').mkdir(parents=True, exist_ok=True)
    (stitch / 'stub' / '0' / '.zarray').write_text(json.dumps({
        'chunks': [128, 128, 128], 'compressor': None, 'dtype': '|u1', 'fill_value': 0, 'filters': None, 'order': 'C',
        'shape': list(shape), 'zarr_format': 2, 'dimension_separator': '/'}))
    (stitch / 'stub' / '.zgroup').write_text('{"zarr_format":2}')
    for f in ('meta.json', 'metadata.json'):
        (stitch / 'stub' / f).write_text(json.dumps({'voxelsize': voxelsize}))


DEFAULT_TRACE = {"src_step": 20, "step": 10, "steps": 5000, "same_surface_th": 4, "local_cost_inl_th": 0.2,
                 "inlier_base_threshold": 20, "consensus_default_th": 10, "consensus_limit_th": 4, "straight_min_count": 1,
                 "flip_x": 0, "use_cuda": False, "growth_directions": ["all"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('same-winding')
    s.add_argument('paths', type=Path); s.add_argument('seed')
    s.add_argument('--thr', type=float, default=4.0); s.add_argument('--move', type=Path); s.add_argument('--out', type=Path)
    w = sub.add_parser('winding')
    w.add_argument('paths', type=Path); w.add_argument('seed')
    w.add_argument('--accept', type=float, default=0.8); w.add_argument('--move', type=Path); w.add_argument('--out', type=Path)
    g = sub.add_parser('gate')
    g.add_argument('paths', type=Path); g.add_argument('ct_url'); g.add_argument('--thr', type=float, default=0.4)
    st = sub.add_parser('steer')
    st.add_argument('stitch', type=Path); st.add_argument('pred_url'); st.add_argument('seed')
    st.add_argument('--vc-seed', default='vc_grow_seg_from_seed', help='command for vc_grow_seg_from_seed')
    st.add_argument('--parallel', type=int, default=8); st.add_argument('--per-round', type=int, default=16)
    st.add_argument('--hours', type=float, default=4); st.add_argument('--rng-base', type=int, default=100000)
    st.add_argument('--method', choices=('winding', 'same-winding'), default='winding')
    sc = sub.add_parser('stitch')
    sc.add_argument('stitch', type=Path); sc.add_argument('seed')
    sc.add_argument('--vc-segments', default='vc_grow_seg_from_segments'); sc.add_argument('--minutes', type=float, default=60)
    sb = sub.add_parser('stub')
    sb.add_argument('stitch', type=Path); sb.add_argument('--shape', type=int, nargs=3, required=True, metavar=('Z', 'Y', 'X'))
    sb.add_argument('--voxelsize', type=float, required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'same-winding':
        prev = json.loads(a.out.read_text())['state'] if a.out and a.out.exists() else None
        rep = same_winding(a.paths, a.seed, a.thr, previous=prev)
        if a.move:
            rep['moved'] = move_rejected(a.paths, rep, a.move)
        if a.out:
            a.out.write_text(json.dumps(rep, indent=1))
        print(json.dumps({'counts': rep['counts'], 'moved': rep.get('moved', 0)}))
    elif a.cmd == 'winding':
        rep = winding_filter(a.paths, a.seed, a.accept)
        if a.move:
            rep['moved'] = move_rejected(a.paths, rep, a.move)
        if a.out:
            a.out.write_text(json.dumps(rep, indent=1))
        print(json.dumps({'counts': rep['counts'], 'moved': rep.get('moved', 0), 'seed_sheet_area_cm2': rep['seed_sheet_area_cm2'],
                          'tiles_per_sheet': rep['tiles_per_sheet']}))
    elif a.cmd == 'gate':
        g = gate(a.paths, a.ct_url, a.thr)
        print(json.dumps({'gated': len(g), 'defective': sum(v['defective'] for v in g.values())}))
    elif a.cmd == 'steer':
        steer(a.vc_seed, a.stitch, a.pred_url, a.seed, a.parallel, a.per_round, a.hours, a.rng_base, method=a.method)
    elif a.cmd == 'stitch':
        if not (a.stitch / 'trace.json').exists():
            (a.stitch / 'trace.json').write_text(json.dumps(DEFAULT_TRACE))
        print(stitch_run(a.vc_segments, a.stitch, a.seed, a.minutes))
    else:
        write_stub(a.stitch, a.shape, a.voxelsize)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
