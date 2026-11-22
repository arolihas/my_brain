"""OPTIONAL / NOT USED by the main pipeline: field-map distortion correction, and the test that rejected it.

What it does (a hand-rolled equivalent of FSL fugue, since FSL isn't assumed to be installed):
 1. phase difference -> unwrapped field map in Hz (masked, demeaned, extrapolated, 3 mm smoothing)
 2. rigidly align the field-map magnitude image to the run-2 fMRI reference
 3. shift along the phase-encode axis = Hz x total readout time, undone by resampling (+ Jacobian)
 4. compare no correction / correction (-1, the physics convention for 'j-') / correction (+1)
    against the T1: brain-outline overlap (whole brain and lower-front), and edge agreement where
    the field map predicts the most distortion, all with one fixed EPI->T1 transform.

Result for this scan (results/fieldmap_check.json): the field map is sensible (in-brain shifts mostly
< 0.5 voxel, ~1.5 voxels in orbitofrontal cortex / temporal poles) and -1 beats +1, but neither beats
no correction at 3.4 mm resolution, so the pipeline leaves the data uncorrected.
For a validated correction, run fMRIPrep (it fits the field map with proper regularisation).
"""
import json, pickle
import numpy as np, nibabel as nib
from scipy.ndimage import map_coordinates, gaussian_filter, binary_dilation, binary_erosion, binary_fill_holes, sobel
from skimage.restoration import unwrap_phase
from skimage.filters import threshold_otsu
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
from dipy.align.transforms import RigidTransform3D
from dipy.segment.mask import median_otsu
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from nilearn import plotting
from mybrain.config import bold_meta, fieldmap_files, deriv, FIGURES, RESULTS


def outputs():
    return [RESULTS / 'fieldmap_check.json', FIGURES / 'fieldmap_shift.png']


def field_map_hz():
    f = fieldmap_files(); dTE = f['mag2_meta']['EchoTime'] - f['mag_meta']['EchoTime']
    ph, mag = nib.load(f['phase']), nib.load(f['mag'])
    rad = ph.get_fdata() / 4096 * np.pi                        # Siemens phase difference: +-4096 = +-pi
    _, mask = median_otsu(mag.get_fdata(), median_radius=2, numpass=2)
    mask = binary_fill_holes(binary_erosion(mask))
    hz = unwrap_phase(np.ma.array(rad, mask=~mask)).filled(0) / (2 * np.pi * dTE)
    hz[mask] -= np.median(hz[mask])
    known = mask.copy(); filled = hz * mask
    for _ in range(8):                                           # extrapolate a few voxels outward before smoothing
        grow = binary_dilation(known) & ~known
        filled[grow] = gaussian_filter(filled, 1)[grow] / np.maximum(gaussian_filter(known.astype(float), 1)[grow], 1e-6); known |= grow
    hz = gaussian_filter(filled, 3 / 2.3548 / np.array(ph.header.get_zooms()[:3]))
    return mag, hz, mask


def unwarp(vol, hz, sign, readout):
    shift = sign * hz * readout
    jac = np.clip(1 + np.gradient(shift, axis=1), 0.2, 3)
    I, J, K = np.meshgrid(*[np.arange(s) for s in hz.shape], indexing='ij')
    return map_coordinates(vol, [I, J + shift, K], order=3, mode='nearest') * jac


def main():
    meta = bold_meta(2); readout = meta['TotalReadoutTime']; assert meta['PhaseEncodingDirection'] == 'j-'
    mag, hz, mask = field_map_hz()
    plotting.plot_stat_map(nib.Nifti1Image((hz * mask * readout * 3.4375).astype(np.float32), mag.affine), bg_img=mag, cmap='RdBu_r', vmax=12,
                           threshold=2, display_mode='x', cut_coords=[-40, -25, 0, 25, 40], black_bg=True,
                           title='Predicted fMRI distortion (mm, front-back)').savefig(FIGURES / 'fieldmap_shift.png', dpi=110)
    im = nib.load(deriv('run2_moco.nii.gz')); A = im.affine; ref = np.asarray(im.dataobj[..., :10]).mean(-1)
    reg = lambda: AffineRegistration(metric=MutualInformationMetric(nbins=32), level_iters=[500, 100, 30], sigmas=[2, 1, 0], factors=[4, 2, 1], verbosity=0)
    hz_epi = reg().optimize(ref, mag.get_fdata(), RigidTransform3D(), None, static_grid2world=A, moving_grid2world=mag.affine).transform(hz)
    T1 = nib.load(deriv('t1_2mm.nii.gz')); tm = nib.load(deriv('t1_brainmask_2mm.nii.gz')).get_fdata() > 0; t1d = T1.get_fdata() * tm
    x = pickle.load(open(deriv('epi_to_t1.pkl'), 'rb'))['epi_to_t1']          # one fixed transform for every candidate
    ijk = np.indices(tm.shape).reshape(3, -1).T; xyz = (ijk @ T1.affine[:3, :3].T + T1.affine[:3, 3]).reshape(tm.shape + (3,))
    low_front = (xyz[..., 2] < xyz[..., 2][tm].mean()) & (xyz[..., 1] > xyz[..., 1][tm].mean())
    distorted = binary_dilation(tm, iterations=2) & (np.abs(x.transform(hz_epi)) * readout > 0.5)
    grad = lambda v: np.sqrt(sum(sobel(gaussian_filter(v, 1), a) ** 2 for a in range(3)))
    dice = lambda a, b: float(2 * (a & b).sum() / (a.sum() + b.sum()))
    shift_vox = (hz_epi * readout)[ref > np.percentile(ref, 65)]
    out = {'readout_s': readout, 'in_brain_shift_voxels_pct5_50_95': np.round(np.percentile(shift_vox, [5, 50, 95]), 2).tolist(), 'scores': {}}
    for name, s in [('no correction', 0), ('correction -1 (physics convention)', -1), ('correction +1', +1)]:
        w = x.transform(unwarp(ref, hz_epi, s, readout) if s else ref)
        em = binary_fill_holes(w > threshold_otsu(w[w > 0]))
        out['scores'][name] = dict(outline_dice_whole=round(dice(em, tm), 4), outline_dice_lower_front=round(dice(em & low_front, tm & low_front), 4),
                                   edge_agreement_where_distorted=round(float(np.corrcoef(grad(w)[distorted], grad(t1d)[distorted])[0, 1]), 4))
        print(name, out['scores'][name], flush=True)
    best = max(out['scores'], key=lambda k: out['scores'][k]['outline_dice_whole'])
    out['decision'] = 'leave uncorrected' if best == 'no correction' else f'consider: {best}'
    json.dump(out, open(RESULTS / 'fieldmap_check.json', 'w'), indent=1); print(out['decision'])


if __name__ == '__main__':
    main()
