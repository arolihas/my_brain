"""MNI template, analysis masks, atlases and coordinate helpers (all in 2 mm MNI space)."""
from functools import lru_cache
import numpy as np, nibabel as nib
from nilearn import datasets, image, masking
from .config import deriv, RUNS

# Seed regions for seed-based connectivity maps: key -> (title, MNI mm)
SEEDS = {'default': ('Default mode (posterior cingulate)', (0, -52, 26)),
         'motor': ('Motor (left hand area)', (-38, -22, 56)),
         'visual': ('Visual (primary)', (0, -88, 6)),
         'salience': ('Salience (anterior insula)', (38, 22, -4)),
         'amygdala': ('Amygdala (left)', (-22, -4, -18))}

YEO7 = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
YEO7_NAMES = dict(Vis='Visual', SomMot='Somatomotor', DorsAttn='Dorsal attention', SalVentAttn='Salience / ventral attention',
                  Limbic='Limbic', Cont='Frontoparietal control', Default='Default mode')


@lru_cache
def mni():
    return datasets.load_mni152_template(resolution=2)


def _to_mni(img, nearest=True):
    return image.resample_to_img(img, mni(), interpolation='nearest' if nearest else 'continuous')


@lru_cache
def brain_mask():
    """Grey matter (template prob > 0.2) that is inside the fMRI field of view."""
    gm = image.math_img('img > 0.2', img=_to_mni(datasets.load_mni152_gm_mask()))
    fov = image.math_img('img > 0', img=nib.load(deriv('run2_mni.nii.gz')).slicer[..., 0])
    return image.math_img('a * b', a=gm, b=fov)


@lru_cache
def wm_mask():
    return image.math_img('img > 0.9', img=_to_mni(datasets.load_mni152_wm_mask()))


def unmask(vec):
    return masking.unmask(np.asarray(vec, dtype=np.float32), brain_mask())


@lru_cache
def voxel_mm():
    """MNI coordinates (mm) of every analysis voxel, in mask order."""
    ijk = np.array(np.nonzero(brain_mask().get_fdata() > 0)).T
    return ijk @ mni().affine[:3, :3].T + mni().affine[:3, 3]


def sphere(center, radius=6):
    return np.linalg.norm(voxel_mm() - np.asarray(center), axis=1) <= radius


@lru_cache
def schaefer100():
    """Returns (labels, network-per-label, parcel index per analysis voxel; 0 = none)."""
    a = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2, verbose=0)
    labels = [l.decode() if isinstance(l, bytes) else l for l in a.labels]
    labels = [l for l in labels if l != 'Background']
    nets = [l.split('_')[2] for l in labels]
    per_voxel = masking.apply_mask(_to_mni(a.maps), brain_mask()).astype(int)
    return labels, nets, per_voxel


@lru_cache
def thalamus():
    ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm', verbose=0)
    lab = masking.apply_mask(_to_mni(ho.maps), brain_mask()).astype(int)
    return np.isin(lab, [i for i, n in enumerate(ho.labels) if 'Thalamus' in n])


@lru_cache
def _ho_cortical():
    ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', verbose=0)
    return ho.labels, masking.apply_mask(_to_mni(ho.maps), brain_mask()).astype(int)


def cortical_region(*name_parts):
    """Analysis voxels in Harvard-Oxford cortical regions whose names contain any of name_parts."""
    labels, per_voxel = _ho_cortical()
    return np.isin(per_voxel, [i for i, n in enumerate(labels) if any(k in n for k in name_parts)])


def parcellate(X, per_voxel, n):
    """Mean time series in each parcel 1..n of an (time x voxel) array."""
    return np.stack([X[:, per_voxel == k + 1].mean(1) for k in range(n)], 1)


def t1_in_mni():
    return str(deriv('t1_in_mni.nii.gz'))
