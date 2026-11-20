"""Score field-map correction direction against the skull-stripped T1 (brain-outline Dice, whole + low/frontal)."""
import numpy as np, nibabel as nib, json, pickle
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
from dipy.align.transforms import RigidTransform3D
import importlib; fm = importlib.import_module('06_fieldmap')
d = 'derivatives/'
t1 = nib.load(d + 't1_stripped_2mm.nii.gz'); t1d = t1.get_fdata(); tm = nib.load(d + 't1_brainmask_2mm.nii.gz').get_fdata() > 0
im = nib.load(d + 'run2_moco.nii.gz'); A = im.affine; ref = np.asarray(im.dataobj[..., :10]).mean(-1)
hz = np.load(d + 'fmap_hz_run2.npy')
# region where distortion matters: voxels in T1 space whose world z < 10 mm-ish lower third & front half
ijk = np.indices(tm.shape).reshape(3, -1).T; xyz = (ijk @ t1.affine[:3, :3].T + t1.affine[:3, 3]).reshape(tm.shape + (3,))
zc = xyz[..., 2][tm].mean(); yc = xyz[..., 1][tm].mean()
low_front = (xyz[..., 2] < zc) & (xyz[..., 1] > yc)
def dice(a, b): return 2 * (a & b).sum() / (a.sum() + b.sum())
out = {}
for s in (0, -1, +1):
    u = fm.unwarp(ref, hz, s) if s else ref
    reg = AffineRegistration(metric=MutualInformationMetric(nbins=32), level_iters=[1000, 200, 50], sigmas=[3, 1, 0], factors=[4, 2, 1], verbosity=0)
    x = reg.optimize(t1d, u, RigidTransform3D(), None, static_grid2world=t1.affine, moving_grid2world=A)
    w = x.transform(u); em = binary_fill_holes(w > threshold_otsu(w[w > 0]))
    out[s] = dict(dice_whole=round(dice(em, tm), 4), dice_low_front=round(dice(em & low_front, tm & low_front), 4),
                  corr_in_brain=round(float(np.corrcoef(w[tm], t1d[tm])[0, 1]), 4))
    pickle.dump({'epi_to_t1': x}, open(d + f'epi_to_t1_sign{s:+d}.pkl', 'wb'))
    nib.save(nib.Nifti1Image(w.astype(np.float32), t1.affine), d + f'epiref_in_t1_sign{s:+d}.nii.gz')
    print(s, out[s], flush=True)
json.dump({str(k): v for k, v in out.items()}, open(d + 'fmap_sign_check.json', 'w'), indent=1)
