"""Interactive 3D viewers you can open in a browser (figures/interactive/*.html).

anatomy_3d.html and mean_bold.html replace the original 3d_brain.html / 4d_brain.html, now built from
the motion-corrected, MNI-aligned data.
"""
from nilearn import plotting, image
from mybrain.config import INTERACTIVE, deriv
from mybrain import space
from mybrain.space import SEEDS


def outputs():
    return [INTERACTIVE / 'anatomy_3d.html', INTERACTIVE / 'mean_bold.html'] + [INTERACTIVE / f'seed_{k}.html' for k in SEEDS]


def main():
    plotting.view_img(space.t1_in_mni(), bg_img=False, cmap='gray', symmetric_cmap=False, black_bg=True, threshold=None,
                      title='My T1 anatomy (MNI space)').save_as_html(INTERACTIVE / 'anatomy_3d.html')
    mean_bold = image.mean_img(str(deriv('run2_mni.nii.gz')), copy_header=True)
    plotting.view_img(mean_bold, bg_img=False, cmap='gray', symmetric_cmap=False, black_bg=True, threshold=None,
                      title='Mean fMRI signal, run 2').save_as_html(INTERACTIVE / 'mean_bold.html')
    for key, (title, xyz) in SEEDS.items():
        plotting.view_img(str(deriv(f'seed_{key}.nii.gz')), bg_img=space.t1_in_mni(), threshold=0.25, vmax=1, cut_coords=xyz,
                          title=title, black_bg=True).save_as_html(INTERACTIVE / f'seed_{key}.html')
    print('wrote', len(outputs()), 'viewers to', INTERACTIVE)


if __name__ == '__main__':
    main()
