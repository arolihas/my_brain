"""Quality control: head motion, temporal SNR, and registration checks."""
import json
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from nilearn import plotting, image
from mybrain.config import RUNS, TR, DUMMY, FD_THRESH, FIGURES, RESULTS, deriv
from mybrain import motion, space


def outputs():
    return [RESULTS / 'qc.json', FIGURES / 'qc_motion.png', FIGURES / 'qc_tsnr.png', FIGURES / 'qc_registration.png']


def main():
    qc = {}
    fig, ax = plt.subplots(2, 1, figsize=(11, 5))
    for r in RUNS:
        p = motion.params(r); p = p - p[0]; fd = motion.framewise_displacement(p)
        tsnr = nib.load(deriv(f'tsnr_run{r}.nii.gz')).get_fdata()[space.brain_mask().get_fdata() > 0]
        qc[f'run{r}'] = dict(volumes=len(fd), minutes=round(len(fd) * TR / 60, 1), mean_FD_mm=round(float(fd.mean()), 3),
                             pct_volumes_FD_over_0p5=round(float((fd > FD_THRESH).mean() * 100), 1),
                             max_displacement_mm=round(float(np.abs(p[:, :3]).max()), 2),
                             max_rotation_deg=round(float(np.degrees(np.abs(p[:, 3:]).max())), 2),
                             volumes_kept=int((fd[DUMMY:] <= FD_THRESH).sum()), median_tSNR=round(float(np.median(tsnr)), 1))
        ax[0].plot(np.arange(len(fd)) * TR / 60, fd, lw=.7, label=f'run {r}')
    ax[0].axhline(FD_THRESH, c='k', ls=':', lw=.8); ax[0].set(ylabel='FD (mm)', xlabel='minutes into run'); ax[0].legend(ncol=5, fontsize=8)
    ax[1].bar([f'run {r}' for r in RUNS], [qc[f'run{r}']['mean_FD_mm'] for r in RUNS]); ax[1].set_ylabel('mean FD (mm)')
    ax[0].set_title('Head motion (framewise displacement)'); plt.tight_layout(); plt.savefig(FIGURES / 'qc_motion.png', dpi=110); plt.close()

    mean_tsnr = image.mean_img([deriv(f'tsnr_run{r}.nii.gz') for r in RUNS], copy_header=True)
    plotting.plot_img(mean_tsnr, cmap='inferno', colorbar=True, display_mode='z', cut_coords=8, vmin=0, vmax=80, black_bg=True,
                      title='temporal SNR (mean over runs)').savefig(FIGURES / 'qc_tsnr.png', dpi=110)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    p = plotting.plot_anat(space.t1_in_mni(), axes=ax[0], cut_coords=(0, -20, 10), title='T1 in MNI (red: template edges)')
    p.add_edges(space.mni())
    p = plotting.plot_anat(str(deriv('t1_2mm.nii.gz')), axes=ax[1], cut_coords=(0, 0, 0), title='fMRI on T1 (red: fMRI edges)')
    p.add_edges(str(deriv('epiref_in_t1.nii.gz')))
    plt.savefig(FIGURES / 'qc_registration.png', dpi=110); plt.close()
    json.dump(qc, open(RESULTS / 'qc.json', 'w'), indent=1)
    for k, v in qc.items(): print(k, v)


if __name__ == '__main__':
    main()
