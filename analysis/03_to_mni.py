"""Put each motion-corrected run into MNI space (run ref -> run2 ref -> T1 -> MNI)."""
import sys, pickle, numpy as np, nibabel as nib
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
from dipy.align.transforms import RigidTransform3D
from dipy.align.imaffine import AffineMap
from nilearn import datasets
d = 'derivatives/'
mni = datasets.load_mni152_template(resolution=2)
t1 = nib.load(d + 't1_brain_2mm.nii.gz')
T = pickle.load(open(d + 't1_to_mni.pkl', 'rb')); syn = T['syn']
e2t1 = pickle.load(open(d + 'epi_to_t1.pkl', 'rb'))['epi_to_t1']
ref2_img = nib.load(d + 'run2_moco.nii.gz')
ref2 = np.asarray(ref2_img.dataobj[..., :10]).mean(-1)
reg = AffineRegistration(metric=MutualInformationMetric(nbins=32), level_iters=[200, 50], sigmas=[1, 0], factors=[2, 1], verbosity=0)
for r in [int(a) for a in sys.argv[1:]]:
    im = nib.load(d + f'run{r}_moco.nii.gz'); A = im.affine; data = im.get_fdata(dtype=np.float32)
    ref = data[..., :10].mean(-1)
    to2 = reg.optimize(ref2, ref, RigidTransform3D(), None, static_grid2world=ref2_img.affine, moving_grid2world=A)
    # compose: moving voxel->world(run r) -> run2 world -> T1 world ; e2t1.affine maps T1-world -> run2-world points
    M = e2t1.affine @ to2.affine   # static(T1) world -> run r world
    amap = AffineMap(M, domain_grid_shape=t1.shape, domain_grid2world=t1.affine, codomain_grid_shape=im.shape[:3], codomain_grid2world=A)
    out = np.zeros(mni.shape + (data.shape[-1],), np.float32)
    for t in range(data.shape[-1]):
        out[..., t] = syn.transform(amap.transform(data[..., t]))
    nib.save(nib.Nifti1Image(out, mni.affine), d + f'run{r}_mni.nii.gz')
    print('run', r, 'to-run2 shift (mm):', np.round(to2.affine[:3, 3], 2), flush=True)
