"""T1 -> MNI (affine + SyN) and mean EPI -> T1 (rigid), all with dipy."""
import numpy as np, nibabel as nib, os
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric, transform_centers_of_mass
from dipy.align.transforms import TranslationTransform3D, RigidTransform3D, AffineTransform3D
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.segment.mask import median_otsu
from nilearn import datasets, image
import pickle
out = 'analysis/derivatives'
t1 = image.resample_img(nib.load('nifti/Brain_t1memprage_20190920132538_5.nii'), target_affine=np.diag([2, 2, 2]), interpolation='continuous')
t1d = t1.get_fdata(); t1d = np.clip(t1d, 0, None)
brain, mask = median_otsu(t1d, median_radius=2, numpass=2, dilate=1)
nib.save(nib.Nifti1Image(brain.astype(np.float32), t1.affine), f'{out}/t1_brain_2mm.nii.gz')
mni = datasets.load_mni152_template(resolution=2); md = mni.get_fdata()
reg = AffineRegistration(metric=MutualInformationMetric(nbins=32), level_iters=[1000, 200, 50], sigmas=[3, 1, 0], factors=[4, 2, 1], verbosity=0)
kw = dict(static_grid2world=mni.affine, moving_grid2world=t1.affine)
c = transform_centers_of_mass(md, mni.affine, brain, t1.affine)
x = reg.optimize(md, brain, TranslationTransform3D(), None, starting_affine=c.affine, **kw)
x = reg.optimize(md, brain, RigidTransform3D(), None, starting_affine=x.affine, **kw)
aff = reg.optimize(md, brain, AffineTransform3D(), None, starting_affine=x.affine, **kw)
sdr = SymmetricDiffeomorphicRegistration(CCMetric(3), level_iters=[50, 25, 10])
syn = sdr.optimize(md, brain, mni.affine, t1.affine, aff.affine)
nib.save(nib.Nifti1Image(syn.transform(brain).astype(np.float32), mni.affine), f'{out}/t1_in_mni.nii.gz')
pickle.dump({'affine': aff, 'syn': syn}, open(f'{out}/t1_to_mni.pkl', 'wb'))
print('t1->mni done', flush=True)
# EPI reference (first 10 vols of each run share a session; use run 2 ref) -> T1 brain (rigid)
import glob
f = sorted(glob.glob('nifti/Brain_fMRI_2_*.nii'))[0]; e = nib.load(f)
ref = np.asarray(e.dataobj[..., :10]).mean(-1)
reg2 = AffineRegistration(metric=MutualInformationMetric(nbins=32), level_iters=[1000, 200, 50], sigmas=[3, 1, 0], factors=[4, 2, 1], verbosity=0)
kw = dict(static_grid2world=t1.affine, moving_grid2world=e.affine)
r = reg2.optimize(brain, ref, RigidTransform3D(), None, starting_affine=np.eye(4), **kw)
pickle.dump({'epi_to_t1': r}, open(f'{out}/epi_to_t1.pkl', 'wb'))
nib.save(nib.Nifti1Image(r.transform(ref).astype(np.float32), t1.affine), f'{out}/epiref_in_t1.nii.gz')
print('epi->t1 done')
