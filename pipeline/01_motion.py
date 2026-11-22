"""Rigid-body motion correction of every fMRI volume to its run's reference (mean of the first 10 volumes)."""
import os, sys, time
from multiprocessing import Pool
import numpy as np, nibabel as nib
from mybrain.config import RUNS, bold, deriv


def outputs():
    return [deriv(f'run{r}_moco.nii.gz') for r in RUNS] + [deriv(f'run{r}_motion.txt') for r in RUNS]


def rotation_angles(R):
    return [np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])]


def correct_run(run):
    from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
    from dipy.align.transforms import RigidTransform3D
    im = nib.load(bold(run)); data = im.get_fdata(dtype=np.float32); A = im.affine
    ref = data[..., :10].mean(-1)
    reg = AffineRegistration(metric=MutualInformationMetric(nbins=32), level_iters=[50, 20], sigmas=[1.0, 0.0], factors=[2, 1], verbosity=0)
    T = data.shape[-1]; params = np.zeros((T, 6)); out = np.empty_like(data); prev = np.eye(4); t0 = time.time()
    for t in range(T):
        m = reg.optimize(ref, data[..., t], RigidTransform3D(), None, static_grid2world=A, moving_grid2world=A, starting_affine=prev)
        prev = m.affine; out[..., t] = m.transform(data[..., t])
        params[t, :3] = m.affine[:3, 3]; params[t, 3:] = rotation_angles(m.affine[:3, :3])
    nib.save(nib.Nifti1Image(out, A, im.header), deriv(f'run{run}_moco.nii.gz'))
    np.savetxt(deriv(f'run{run}_motion.txt'), params, fmt='%.6f', header='tx ty tz (mm) rx ry rz (rad)')
    return f'run {run}: {T} volumes in {time.time() - t0:.0f}s'


def main(runs=RUNS):
    os.environ['OMP_NUM_THREADS'] = '1'
    with Pool(min(4, len(runs), os.cpu_count())) as pool:
        for msg in pool.imap_unordered(correct_run, sorted(runs, key=lambda r: r == 1)):  # long runs first
            print(msg, flush=True)


if __name__ == '__main__':
    main([int(a) for a in sys.argv[1:]] or RUNS)
