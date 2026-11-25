"""Run the whole analysis, skipping steps whose outputs already exist.

    python run_pipeline.py              # everything that's missing
    python run_pipeline.py --from 6     # re-run step 6 onwards
    python run_pipeline.py --only 7     # just step 7
    python run_pipeline.py --list       # show steps and whether they're done
    python run_pipeline.py --fieldmap   # also run the optional field-map check

Steps 1 and 3 are slow (~20 and ~10 minutes with 4 cores); the rest take a few minutes each.
"""
import argparse, importlib, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STEPS = ['00_convert', '01_motion', '02_register', '03_to_mni', '04_denoise', '05_qc', '06_networks', '07_drowsiness', '08_interactive', '09_trial_onsets']
OPTIONAL = ['x1_fieldmap_check']


def load(name):
    return importlib.import_module(f'pipeline.{name}')


def done(mod):
    outs = mod.outputs()
    return bool(outs) and all(o.exists() for o in outs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from', dest='start', type=int, help='force re-run from this step number')
    ap.add_argument('--only', type=int, help='force re-run of just this step')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--fieldmap', action='store_true', help='also run the optional field-map check')
    a = ap.parse_args()
    steps = STEPS + (OPTIONAL if a.fieldmap else [])
    for name in steps:
        mod = load(name); num = int(name[:2]) if name[:2].isdigit() else None
        forced = (a.only is not None and num == a.only) or (a.start is not None and num is not None and num >= a.start)
        if a.list:
            print(f"{name:20s} {'done' if done(mod) else 'missing'}   {(mod.__doc__ or '').strip().splitlines()[0]}"); continue
        if a.only is not None and num != a.only: continue
        if done(mod) and not forced:
            print(f'[skip] {name} (outputs exist)'); continue
        print(f'[run ] {name}', flush=True); t0 = time.time()
        mod.main()
        print(f'[done] {name} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
