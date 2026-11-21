"""Anatomy: T1 -> MNI (affine + SyN), a T1 brain mask, and the fMRI reference -> T1 (rigid)."""
import pickle
import numpy as np, nibabel as nib
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric, transform_centers_of_mass
from dipy.align.transforms import TranslationTransform3D, RigidTransform3D, AffineTransform3D
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.segment.mask import median_otsu
from nilearn import datasets, image
from mybrain.config import t1, bold, deriv
from mybrain.space import mni


def outputs():
    return [deriv(n) for n in ('t1_2mm.nii.gz', 't1_to_mni.pkl', 't1_in_mni.nii.gz', 't1_brainmask_2mm.nii.gz', 'epi_to_t1.pkl')]


def affine_reg():
    return AffineRegistration(metric=MutualInformationMetric(nbins=32), level_iters=[1000, 200, 50], sigmas=[3, 1, 0], factors=[4, 2, 1], verbosity=0)


def main():
    T1 = image.resample_img(nib.load(t1()), target_affine=np.diag([2, 2, 2]), interpolation='continuous')
    t1d = np.clip(T1.get_fdata(), 0, None)
    nib.save(nib.Nifti1Image(t1d.astype(np.float32), T1.affine), deriv('t1_2mm.nii.gz'))
    head, _ = median_otsu(t1d, median_radius=2, numpass=2, dilate=1)   # removes background, not the skull

    # T1 -> MNI: centre of mass, translation, rigid, affine, then nonlinear (SyN)
    M = mni(); md = M.get_fdata(); kw = dict(static_grid2world=M.affine, moving_grid2world=T1.affine)
    start = transform_centers_of_mass(md, M.affine, head, T1.affine).affine
    for tf in (TranslationTransform3D(), RigidTransform3D(), AffineTransform3D()):
        x = affine_reg().optimize(md, head, tf, None, starting_affine=start, **kw); start = x.affine
    syn = SymmetricDiffeomorphicRegistration(CCMetric(3), level_iters=[50, 25, 10]).optimize(md, head, M.affine, T1.affine, start)
    pickle.dump({'affine': x, 'syn': syn}, open(deriv('t1_to_mni.pkl'), 'wb'))
    nib.save(nib.Nifti1Image(syn.transform(head).astype(np.float32), M.affine), deriv('t1_in_mni.nii.gz'))

    # brain mask in T1 space: the MNI brain mask pulled back through the warp
    bm = image.resample_to_img(datasets.load_mni152_brain_mask(resolution=2), M, interpolation='nearest').get_fdata()
    mask = syn.transform_inverse(bm.astype(float), interpolation='linear') > 0.5
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), T1.affine), deriv('t1_brainmask_2mm.nii.gz'))
    print('T1 -> MNI done', flush=True)

    # fMRI reference (run 2) -> T1 brain, rigid
    e = nib.load(bold(2)); ref = np.asarray(e.dataobj[..., :10]).mean(-1)
    r = affine_reg().optimize(t1d * mask, ref, RigidTransform3D(), None, starting_affine=np.eye(4),
                              static_grid2world=T1.affine, moving_grid2world=e.affine)
    pickle.dump({'epi_to_t1': r}, open(deriv('epi_to_t1.pkl'), 'wb'))
    nib.save(nib.Nifti1Image(r.transform(ref).astype(np.float32), T1.affine), deriv('epiref_in_t1.nii.gz'))
    print('fMRI -> T1 done', flush=True)


if __name__ == '__main__':
    main()
