"""Where everything lives, and the facts about the scan session."""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'Brain'                 # original DICOMs from the scanner
NIFTI = ROOT / 'nifti'               # dcm2niix conversion of RAW (committed, via git LFS)
# Large intermediate files (motion-corrected runs, MNI-space data, ...) are regenerated, not committed.
DERIV = Path(os.environ.get('MYBRAIN_DERIV', ROOT / 'derivatives'))
FIGURES = ROOT / 'figures'
INTERACTIVE = FIGURES / 'interactive'
RESULTS = ROOT / 'results'

RUNS = [1, 2, 3, 4, 5]               # fMRI_1 .. fMRI_5
TASK_RUNS = [2, 3, 4, 5]             # identical 11.3-min protocols; run 1 is shorter (36 slices)
TR = 1.155                           # seconds per volume
DUMMY = 5                            # first volumes dropped (signal not yet at steady state)
FD_THRESH = 0.5                      # mm; volumes above this are scrubbed
SMOOTH_FWHM = 6.0                    # mm

for _p in (DERIV, FIGURES, INTERACTIVE, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)


def _one(pattern):
    hits = sorted(NIFTI.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f'expected exactly one {pattern} in {NIFTI}, found {len(hits)}')
    return hits[0]


def bold(run):
    """Raw 4D fMRI NIfTI for a run."""
    return _one(f'Brain_fMRI_{run}_*.nii')


def bold_meta(run):
    return json.loads(_one(f'Brain_fMRI_{run}_*.json').read_text())


def t1():
    return _one('Brain_t1memprage_*.nii')


def fieldmap_files():
    return dict(mag=_one('Brain_gre_field_mapping_*_e1.nii'), phase=_one('Brain_gre_field_mapping_*_ph.nii'),
                mag_meta=json.loads(_one('Brain_gre_field_mapping_*_e1.json').read_text()),
                mag2_meta=json.loads(_one('Brain_gre_field_mapping_*_e2.json').read_text()))


def deriv(name):
    return DERIV / name
