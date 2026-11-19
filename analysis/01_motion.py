"""Rigid-body motion correction of each fMRI run (dipy), plus framewise displacement."""
import sys, time, numpy as np, nibabel as nib
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
from dipy.align.transforms import RigidTransform3D
from dipy.align import imaffine
import glob, os
out = 'analysis/derivatives'; os.makedirs(out, exist_ok=True)
runs = sorted(glob.glob('nifti/Brain_fMRI_*.nii'), key=lambda f: int(f.split('_')[2]))
which = [int(a) for a in sys.argv[1:]] or range(1, 6)
reg = AffineRegistration(metric=MutualInformationMetric(nbins=32), level_iters=[50, 20], sigmas=[1.0, 0.0], factors=[2, 1], verbosity=0)
for r in which:
    f = runs[r-1]; im = nib.load(f); d = im.get_fdata(dtype=np.float32); A = im.affine
    ref = d[..., :10].mean(-1)  # run-specific reference
    T = d.shape[-1]; params = np.zeros((T, 6)); moco = np.empty_like(d); prev = np.eye(4)
    t0 = time.time()
    for t in range(T):
        m = reg.optimize(ref, d[..., t], RigidTransform3D(), None, static_grid2world=A, moving_grid2world=A, starting_affine=prev)
        prev = m.affine
        moco[..., t] = m.transform(d[..., t])
        params[t] = m.get_parameters() if hasattr(m, 'get_parameters') else 0
        # store translation + rotation from matrix
        M = m.affine; params[t, :3] = M[:3, 3]
        R = M[:3, :3]
        params[t, 3:] = [np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])]
    nib.save(nib.Nifti1Image(moco, A, im.header), f'{out}/run{r}_moco.nii.gz')
    np.savetxt(f'{out}/run{r}_motion.txt', params, fmt='%.6f', header='tx ty tz (mm) rx ry rz (rad)')
    print(f'run{r}: {T} vols in {time.time()-t0:.0f}s', flush=True)
