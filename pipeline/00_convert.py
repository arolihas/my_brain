"""DICOM -> NIfTI with dcm2niix (how nifti/ was made). Skipped when nifti/ already has the scans."""
import shutil, subprocess
from mybrain.config import RAW, NIFTI


def outputs():
    return sorted(NIFTI.glob('Brain_fMRI_*.nii'))


def main():
    if not shutil.which('dcm2niix'):
        raise SystemExit('dcm2niix not found: pip install dcm2niix')
    NIFTI.mkdir(exist_ok=True)
    subprocess.run(['dcm2niix', '-f', 'Brain_%p_%t_%s', '-o', str(NIFTI), str(RAW)], check=True)


if __name__ == '__main__':
    main()
