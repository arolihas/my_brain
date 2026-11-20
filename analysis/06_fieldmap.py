"""Field-map distortion correction (a hand-rolled FSL-fugue equivalent).

1. phase difference -> unwrapped field map in Hz (masked, demeaned, smoothed, extrapolated)
2. align field-map magnitude to each run's EPI reference (rigid)
3. voxel shift along the phase-encode axis = Hz * total readout time
4. resample each motion-corrected volume along that axis (+ Jacobian intensity correction)
STATUS (see 06b_check_sign.py): NOT used in the main pipeline. The field map looks physically
sensible (in-brain shifts mostly < 0.5 voxel, tails to ~1.5 voxels in orbitofrontal / temporal
pole), and the physics-convention direction (-1 for 'j-') beats the opposite direction, but neither
correction measurably improved the EPI-to-T1 fit over no correction at this 3.4 mm resolution.
"""
import sys, json, numpy as np, nibabel as nib
from scipy.ndimage import map_coordinates, gaussian_filter, binary_dilation, binary_erosion, binary_fill_holes
from skimage.restoration import unwrap_phase
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric, AffineMap
from dipy.align.transforms import RigidTransform3D
from dipy.segment.mask import median_otsu
d = 'derivatives/'; N = '../nifti/'
meta = json.load(open(N + 'Brain_fMRI_2_20190920132538_7.json'))
TRT = meta['TotalReadoutTime']; assert meta['PhaseEncodingDirection'] == 'j-'
dTE = 0.00738 - 0.00492

def build_fieldmap():
    ph = nib.load(N + 'Brain_gre_field_mapping_20190920132538_3_ph.nii'); mag = nib.load(N + 'Brain_gre_field_mapping_20190920132538_2_e1.nii')
    rad = ph.get_fdata() / 4096 * np.pi
    m = mag.get_fdata(); _, mask = median_otsu(m, median_radius=2, numpass=2)
    mask = binary_fill_holes(binary_erosion(mask, iterations=1))
    un = unwrap_phase(np.ma.array(rad, mask=~mask)).filled(0)
    hz = un / (2 * np.pi * dTE); hz[mask] -= np.median(hz[mask])
    wrapped_frac = float(np.mean(np.abs(un[mask] - rad[mask]) > 1))
    # extrapolate outward from the mask, then smooth (like fugue --smooth3)
    filled, known = hz * mask, mask.copy()
    for _ in range(8):
        grow = binary_dilation(known) & ~known
        num = gaussian_filter(filled, 1); den = gaussian_filter(known.astype(float), 1)
        filled[grow] = num[grow] / np.maximum(den[grow], 1e-6); known |= grow
    hz_s = gaussian_filter(filled, 3 / 2.3548 / np.array(ph.header.get_zooms()[:3]))
    nib.save(nib.Nifti1Image(hz_s.astype(np.float32), ph.affine), d + 'fmap_hz_smooth.nii.gz')
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), ph.affine), d + 'fmap_mask.nii.gz')
    return mag, hz_s, wrapped_frac

def fmap_in_epi(mag, hz, ref, A):
    reg = AffineRegistration(metric=MutualInformationMetric(nbins=32), level_iters=[500, 100, 30], sigmas=[2, 1, 0], factors=[4, 2, 1], verbosity=0)
    r = reg.optimize(ref, mag.get_fdata(), RigidTransform3D(), None, static_grid2world=A, moving_grid2world=mag.affine)
    return r.transform(hz, interpolation='linear'), r

def unwarp(vol, hz_epi, sign):
    """vol: 3D or 4D EPI in its own grid; shift along axis 1 (j)."""
    shift = sign * hz_epi * TRT
    jac = np.clip(1 + np.gradient(shift, axis=1), 0.2, 3)  # avoid folding / negative intensities
    I, J, K = np.meshgrid(*[np.arange(s) for s in hz_epi.shape], indexing='ij')
    coords = np.array([I, J + shift, K])
    if vol.ndim == 3: return map_coordinates(vol, coords, order=3, mode='nearest') * jac
    out = np.empty_like(vol)
    for t in range(vol.shape[-1]): out[..., t] = map_coordinates(vol[..., t], coords, order=3, mode='nearest') * jac
    return out

if __name__ == '__main__':
    mag, hz, wf = build_fieldmap()
    print(f'field map: {wf*100:.1f}% of brain voxels needed phase unwrapping; readout {TRT*1000:.1f} ms', flush=True)
    t1 = nib.load(d + 't1_brain_2mm.nii.gz'); t1d = t1.get_fdata()
    runs = [int(a) for a in sys.argv[1:]] or [2]
    sign = None
    for r in runs:
        im = nib.load(d + f'run{r}_moco.nii.gz'); A = im.affine
        data = im.get_fdata(dtype=np.float32); ref = data[..., :10].mean(-1)
        hz_epi, _ = fmap_in_epi(mag, hz, ref, A)
        np.save(d + f'fmap_hz_run{r}.npy', hz_epi)
        if sign is None:
            # pick the sign that best matches anatomy: register each candidate to T1 and score with MI
            scores = {}
            for s in (+1, -1, 0):
                u = unwarp(ref, hz_epi, s) if s else ref
                reg = AffineRegistration(metric=MutualInformationMetric(nbins=32), level_iters=[1000, 200, 50], sigmas=[3, 1, 0], factors=[4, 2, 1], verbosity=0)
                x = reg.optimize(t1d, u, RigidTransform3D(), None, static_grid2world=t1.affine, moving_grid2world=A)
                metric = MutualInformationMetric(nbins=32); metric.setup(RigidTransform3D(), t1d, u, t1.affine, A)
                metric._update_mutual_information(x.get_parameters() if hasattr(x, 'get_parameters') else np.zeros(6), update_gradient=False) if False else None
                # score: correlation of T1 with the warped EPI inside the T1 brain (frontal/temporal-weighted below z<10mm)
                w = x.transform(u); m = t1d > np.percentile(t1d[t1d > 0], 20)
                scores[s] = float(np.corrcoef(w[m], t1d[m])[0, 1])
                nib.save(nib.Nifti1Image(w.astype(np.float32), t1.affine), d + f'epiref_in_t1_sign{s:+d}.nii.gz')
                pk = __import__('pickle'); pk.dump({'epi_to_t1': x}, open(d + f'epi_to_t1_sign{s:+d}.pkl', 'wb'))
            print('EPI-T1 match (corr): no correction %.4f | sign +1 %.4f | sign -1 %.4f' % (scores[0], scores[1], scores[-1]), flush=True)
            sign = max((+1, -1), key=lambda s: scores[s]); json.dump({'sign': sign, 'scores': {str(k): v for k, v in scores.items()}}, open(d + 'fmap_sign.json', 'w'))
            if len(runs) == 1 and runs == [2] and '--all' not in sys.argv: pass
        out = unwarp(data, hz_epi, sign)
        nib.save(nib.Nifti1Image(out.astype(np.float32), A, im.header), d + f'run{r}_moco_sdc.nii.gz')
        print(f'run {r} unwarped (sign {sign:+d})', flush=True)
