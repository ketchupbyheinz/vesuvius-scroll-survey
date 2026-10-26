"""worker_watch.py - queue worker used for the survey (run from the repository root: env\\Scripts\\python.exe windows\\worker_watch.py).

Loops forever over two queues in work/:
  deep/<tag>/job.json   depth sweep on a finished survey patch: models x depths, both directions
                        (job: {"models": [...], "depths": [...]}); maps go to deep/<tag>/maps/
  survey/<tag>/         new traced patch (size.txt present, finished.txt absent): render 28 layers,
                        seed42 + dense_native at d=0 both directions, ct_mid.tif, finished.txt
Deep jobs go first, one (model, depth) at a time, so a new survey patch never waits more than a few minutes:
after every deep step, pending survey patches are processed.
"""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # repository root
(ROOT / 'logs').mkdir(exist_ok=True)
PY = sys.executable
CK = ROOT / 'ckpt'
LOG = ROOT / 'logs' / 'watch.log'
# Hecate: put hecate.py and hecate_9.6um.pth from huggingface.co/scrollprize/hecate into ckpt\\
HEC = {'ckpt': CK / 'hecate_9.6um.pth', 'py': CK / 'hecate.py', 'prep': ROOT / 'windows' / 'hecate_prep.py'}
CKPT = {'seed42': CK / 'seed42_step-075000.pth', 'seed43': CK / 'seed43_step-075000.pth',
        'dnative': CK / 'dense_native-016000.rebuilt.pth'}
BATCH = {'seed42': 4, 'seed43': 4, 'dnative': 2}


def log(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')


def run(cmd, logfile):
    with open(logfile, 'a') as f:
        return subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT).returncode


def infer(zarr, model, out_tif, start, logfile):
    return run([PY, '-m', 'vesuvius.ink_detection.inference.infer', zarr, CKPT[model], out_tif, '--overlap', '0.5',
                '--blend-mode', 'hann', '--batch-size', BATCH[model], '--direction', 'both',
                '--layer-start', start, '--layer-end', start + 17, '--no-compile', '--num-workers', '0'], logfile)


def hecate(zarr, survey_dir, dep, n, out_png, logfile):
    import re
    um = float(re.search(r'-([0-9.]+)um-', (survey_dir / 'ct.txt').read_text()).group(1))
    prep = out_png.parent / f'_prep_d{dep:+d}.zarr'
    rc = run([PY, HEC['prep'], '--src', zarr, '--out', prep, '--native-um', um, '--native-um-z', um, '--target-um', '9.6',
              '--z-center', f'{(n - 1) / 2 + dep:.2f}', '--nz', '16'], logfile)
    if rc:
        return rc
    for rev, suf in (([], ''), (['--reverse'], '_reverse')):
        rc = run([PY, HEC['py'], '--checkpoint', HEC['ckpt'], '--input', prep, '--spacing-um', '9.6',
                  '--output', out_png.with_name(out_png.stem + suf + '.png'), '--device', 'cuda', '--batch-size', '2', *rev],
                 logfile) or rc
    import shutil
    shutil.rmtree(prep, ignore_errors=True)
    return rc


def run_cmds() -> bool:
    """work/cmds/<name>.json = {"cmd": [...]} : run once (in ROOT), log to logs/cmd_<name>.txt, then mark .done"""
    base = ROOT / 'work' / 'cmds'
    if not base.exists():
        return False
    for j in sorted(base.glob('*.json')):
        done = j.with_suffix('.done')
        if done.exists():
            continue
        job = json.loads(j.read_text())
        cmd = [PY if c == '{PY}' else c for c in job['cmd']]
        log('cmd', j.stem)
        rc = subprocess.run(cmd, cwd=ROOT, stdout=open(ROOT / 'logs' / f'cmd_{j.stem}.txt', 'a'), stderr=subprocess.STDOUT).returncode
        done.write_text(str(rc))
        log('cmd done', j.stem, rc)
        return True
    return False


def survey_one(d: Path) -> None:
    lf = ROOT / 'logs' / f'survey_{d.name}.txt'
    log('survey', d.name)
    ct = (d / 'ct.txt').read_text().split()[0]
    R, C = (d / 'size.txt').read_text().split()[:2]
    z = d / 'render.zarr'
    if run([PY, '-m', 'vesuvius_survey.render_patch', d / 'mesh', ct, z, '--rows', '0', R, '--cols', '0', C,
            '--layers', '28', '--tile', '512', '--workers', '8', '--resume'], lf):
        log('render failed', d.name); return
    for m in ('seed42', 'dnative'):
        if not (d / f'{m}_s6_reverse.tif').exists():
            infer(z, m, d / f'{m}_s6.tif', 6, lf)
    run([PY, '-c', f"import zarr,numpy as np,tifffile;a=zarr.open_array(r'{z}',mode='r');"
                   f"tifffile.imwrite(r'{d / 'ct_mid.tif'}',np.asarray(a[a.shape[0]//2]),compression='zlib')"], lf)
    if (d / 'dnative_s6_reverse.tif').exists() and (d / 'seed42_s6_reverse.tif').exists():
        (d / 'finished.txt').write_text('ok')
        log('survey done', d.name)


def pending_survey():
    return [d for d in sorted((ROOT / 'work' / 'survey').iterdir())
            if d.is_dir() and (d / 'size.txt').exists() and not (d / 'finished.txt').exists()]


def deep_steps():
    out = []
    base = ROOT / 'work' / 'deep'
    if not base.exists():
        return out
    for j in sorted(base.glob('*/job*.json')):
        d = j.parent
        donef = d / ('deep_done.txt' if j.stem == 'job' else f'deep_done_{j.stem}.txt')
        if donef.exists():
            continue
        job = json.loads(j.read_text())
        z = ROOT / 'work' / 'survey' / d.name / 'render.zarr'
        n = json.loads((z / '.zarray').read_text())['shape'][0]
        steps = [(m, dep) for dep in job['depths'] for m in job['models']]
        def have(m, dep):
            ext = 'png' if m == 'hecate' else 'tif'
            return (d / 'maps' / f'{m}_d{dep:+d}_reverse.{ext}').exists()
        todo = [(m, dep) for m, dep in steps if not have(m, dep) and 0 <= n // 2 + dep - 8 and n // 2 + dep + 9 <= n]
        if not todo:
            donef.write_text('ok'); log('deep done', d.name, j.stem); continue
        out.append((d, z, n, todo[0]))
    return out


def main():
    log('watch.py started')
    while True:
        flag = ROOT / 'work' / 'restart.flag'
        if flag.exists():
            flag.unlink(); log('restarting with new watch.py')
            sys.exit(subprocess.call([PY, str(Path(__file__).resolve())]))
        did = run_cmds()
        steps = deep_steps()
        if steps:
            d, z, n, (m, dep) = steps[0]
            (d / 'maps').mkdir(exist_ok=True)
            log('deep', d.name, m, f'd={dep:+d}')
            lf = ROOT / 'logs' / f'deep_{d.name}.txt'
            if m == 'hecate':
                if hecate(z, ROOT / 'work' / 'survey' / d.name, dep, n, d / 'maps' / f'hecate_d{dep:+d}.png', lf):
                    (d / 'maps' / f'hecate_d{dep:+d}_reverse.png').write_bytes(b'')   # mark failed, do not loop
                    log('hecate failed', d.name)
            else:
                infer(z, m, d / 'maps' / f'{m}_d{dep:+d}.tif', n // 2 + dep - 8, lf)
            did = True
        for s in pending_survey():
            survey_one(s); did = True
            if deep_steps():
                break
        if not did:
            time.sleep(60)


if __name__ == '__main__':
    main()
