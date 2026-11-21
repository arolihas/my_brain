"""Per run, in MNI space: drop dummy volumes, smooth, regress out noise, and save compact arrays.

Noise model: Friston-24 motion terms + 5 white-matter principal components (aCompCor),
linear detrending and a 0.008 Hz high-pass. Saves, per run:
  clean_run{r}.npy  denoised grey-matter data (volumes x voxels)
  gs_run{r}.npy     raw global grey-matter signal (for the drowsiness markers)
  eyes_run{r}.npy   raw signal in the eyeballs (eye-movement proxy)
  tsnr_run{r}.nii.gz temporal SNR map
"""
import sys
import numpy as np, nibabel as nib
from scipy.ndimage import gaussian_filter
from nilearn import signal
from mybrain.config import RUNS, TR, DUMMY, SMOOTH_FWHM, deriv
from mybrain import motion, space

EYES_MNI = [(33, 52, -36), (-33, 52, -36)]


def outputs():
    return [deriv(f'{k}_run{r}.npy') for r in RUNS for k in ('clean', 'gs', 'eyes')] + [deriv(f'tsnr_run{r}.nii.gz') for r in RUNS]


def denoise_run(run):
    M = space.brain_mask().get_fdata() > 0; W = space.wm_mask().get_fdata() > 0; aff = space.mni().affine
    vol = nib.load(deriv(f'run{run}_mni.nii.gz')).get_fdata(dtype=np.float32)[..., DUMMY:]
    raw = vol[M].T; wm = vol[W].T
    ijk = np.indices(vol.shape[:3]).reshape(3, -1).T; xyz = (ijk @ aff[:3, :3].T + aff[:3, 3]).reshape(vol.shape[:3] + (3,))
    eyes = np.any([np.linalg.norm(xyz - c, axis=-1) < 9 for c in EYES_MNI], axis=0)
    np.save(deriv(f'eyes_run{run}.npy'), vol[eyes].T)
    np.save(deriv(f'gs_run{run}.npy'), raw.mean(1))
    tsnr = raw.mean(0) / (raw.std(0) + 1e-6)
    nib.save(space.unmask(tsnr), deriv(f'tsnr_run{run}.nii.gz'))
    sigma = SMOOTH_FWHM / 2.3548 / 2.0  # in 2 mm voxels
    X = np.stack([gaussian_filter(vol[..., t], sigma)[M] for t in range(vol.shape[-1])])
    del vol, raw
    compcor = np.linalg.svd(signal.clean(wm, detrend=True, standardize='zscore_sample'), full_matrices=False)[0][:, :5]
    conf = np.hstack([motion.confounds24(motion.params(run))[DUMMY:], compcor])
    clean = signal.clean(X, confounds=conf, detrend=True, standardize='zscore_sample', high_pass=0.008, t_r=TR)
    np.save(deriv(f'clean_run{run}.npy'), clean.astype(np.float32))
    print(f'run {run}: denoised {X.shape[0]} volumes x {X.shape[1]} voxels', flush=True)


def main(runs=RUNS):
    for r in runs:      # one at a time: each run is ~2 GB in memory
        denoise_run(r)


if __name__ == '__main__':
    main([int(a) for a in sys.argv[1:]] or RUNS)
