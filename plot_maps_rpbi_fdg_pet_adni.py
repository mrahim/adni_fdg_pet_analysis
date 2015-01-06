"""
    Routines for plotting rpbi's p-maps
    Requires:
    - Pmap
    - MNI T-1 template
"""
import os
import numpy as np
import nibabel as nib
from nilearn.plotting import  plot_img, cm
import matplotlib.cm as cmap
import matplotlib.pyplot as plt

def plot_pmaps():
    for gr in groups:
        filename = '_'.join(['pmap', 'rpbi', 'voxel', 'norm',
                             '_'.join(gr), 'baseline',
                             'adni' ]) + '.nii'
        nii_img = os.path.join( NII_DIR, filename)
        for ext in ['.png', '.pdf', '.svg']:
            try:
                print np.max(nib.load(nii_img).get_data())
                vm = 3
                s = plot_img(nii_img, bg_img=MNI_TEMPLATE,
                             colorbar=True, cmap=cmap.hot,
                             cut_coords=(0, -45, 32),
                             #output_file=os.path.join('figures', 'release',
                             #                         filename.split('.')[0])+ext,
                             black_bg=True, threshold=1, vmin=0, vmax=vm,
                             title='/'.join(gr))
                s.draw_cross(color='b')
                plt.draw()
                plt.savefig(os.path.join('figures',
                                         'release',
                                         filename.split('.')[0])+ext)
            except ValueError:
                plot_img(nii_img, bg_img=MNI_TEMPLATE,
                         colorbar=True, cmap=cmap.hot,
                         #cut_coords=(0, -45, 32),
                         #output_file=os.path.join('figures', 'release',
                         #                         filename.split('.')[0])+ext,
                         black_bg=True, threshold='auto', vmin=0, vmax=vm,
                         title='/'.join(gr))


BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet'
NII_DIR = os.path.join('figures', 'nii')

MNI_TEMPLATE = os.path.join(BASE_DIR, 'wMNI152_T1_2mm_brain.nii')

groups = [['AD', 'Normal'], ['AD', 'EMCI'], ['AD', 'LMCI'],
          ['LMCI', 'Normal'], ['EMCI', 'LMCI'], ['EMCI', 'Normal']]

plot_pmaps()