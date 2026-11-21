"""Warp each motion-corrected run into 2 mm MNI space: run reference -> run-2 reference -> T1 -> MNI."""
import os, sys, pickle
from multiprocessing import Pool
import numpy as np, nibabel as nib
from mybrain.config import RUNS, deriv


def outputs():
    return [deriv(f'run{r}_mni.nii.gz') for r in RUNS]


def warp_run(run):
    from dipy.align.imaffine import AffineRegistration, MutualInformationMetric, AffineMap
    from dipy.align.transforms import RigidTransform3D
    from mybrain.space import mni
    T1 = nib.load(deriv('t1_2mm.nii.gz'))
    syn = pickle.load(open(deriv('t1_to_mni.pkl'), 'rb'))['syn']
    e2t1 = pickle.load(open(deriv('epi_to_t1.pkl'), 'rb'))['epi_to_t1']
    r2 = nib.load(deriv('run2_moco.nii.gz')); ref2 = np.asarray(r2.dataobj[..., :10]).mean(-1)
    im = nib.load(deriv(f'run{run}_moco.nii.gz')); A = im.affine; data = im.get_fdata(dtype=np.float32)
    reg = AffineRegistration(metric=MutualInformationMetric(nbins=32), level_iters=[200, 50], sigmas=[1, 0], factors=[2, 1], verbosity=0)
    to2 = reg.optimize(ref2, data[..., :10].mean(-1), RigidTransform3D(), None, static_grid2world=r2.affine, moving_grid2world=A)
    # e2t1.affine maps T1-world points to run-2 world; to2.affine maps run-2 world to this run's world
    amap = AffineMap(e2t1.affine @ to2.affine, domain_grid_shape=T1.shape, domain_grid2world=T1.affine,
                     codomain_grid_shape=im.shape[:3], codomain_grid2world=A)
    M = mni(); out = np.zeros(M.shape + (data.shape[-1],), np.float32)
    for t in range(data.shape[-1]):
        out[..., t] = syn.transform(amap.transform(data[..., t]))
    nib.save(nib.Nifti1Image(out, M.affine), deriv(f'run{run}_mni.nii.gz'))
    return f'run {run}: head offset from run 2 (mm) {np.round(to2.affine[:3, 3], 2)}'


def main(runs=RUNS):
    os.environ['OMP_NUM_THREADS'] = '1'
    with Pool(min(4, len(runs), os.cpu_count())) as pool:
        for msg in pool.imap_unordered(warp_run, runs):
            print(msg, flush=True)


if __name__ == '__main__':
    main([int(a) for a in sys.argv[1:]] or RUNS)
