"""
    CanICA on ADNI rs-fmri
"""
import os, glob
import numpy as np
import pandas as pd
from nilearn.plotting import plot_img
from nilearn.decomposition.canica import CanICA
from sklearn.decomposition import FastICA
from nilearn.input_data import MultiNiftiMasker, NiftiMasker
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn._utils import concat_niimgs

BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet'
CACHE_DIR = '/disk4t/mehdi/data/tmp'

data = pd.read_csv(os.path.join(BASE_DIR, 'description_file.csv'))
pet_files = []
for idx, row in data.iterrows():
    pet_file = glob.glob(os.path.join(BASE_DIR,
                                      'I' + str(row.Image_ID), 'wI*.nii'))
    if len(pet_file) > 0:
        pet_files.append(pet_file[0])

nb_sample = 250

pet4d = concat_niimgs(np.random.permutation(pet_files)[:nb_sample])


masker = NiftiMasker(mask_strategy='epi',
                     memory=CACHE_DIR,
                     memory_level=2)

pet4d_masked = masker.fit_transform(pet4d)

n_components = 20

fica = FastICA(n_components)
fica.fit(pet4d_masked)
components_img = masker.inverse_transform(fica.components_)
components_img.to_filename(os.path.join(CACHE_DIR, 'fica_tep.nii.gz'))


for i in range(n_components):
    plot_stat_map(nib.Nifti1Image(components_img.get_data()[..., i],
                                      components_img.get_affine()),
                  display_mode="z", title="IC %d"%i, cut_coords=1,
                  colorbar=False, threshold='auto')
plt.show()


"""
multi_masker = MultiNiftiMasker(mask_strategy='epi',
                                memory=CACHE_DIR,
                                n_jobs=1,
                                memory_level=2)
#multi_masker.fit(pet_files_sample)
pet_files_sample_masked = np.array(multi_masker.fit_transform(pet4d))
plot_img(multi_masker.mask_img_)


n_components = 20

fica = FastICA(n_components)
fica.fit(pet_files_sample_masked[...,0])

# Retrieve the independent components in brain space
components_img = multi_masker.inverse_transform(fica.components_)
# components_img is a Nifti Image object, and can be saved to a file with
# the following line:
components_img.to_filename(os.path.join(CACHE_DIR, 'fica_tep.nii.gz'))

### Visualize the results #####################################################
# Show some interesting components

for i in range(n_components):
    plot_stat_map(nib.Nifti1Image(components_img.get_data()[..., i],
                                      components_img.get_affine()),
                  display_mode="z", title="IC %d"%i, cut_coords=1,
                  colorbar=False, threshold='auto')
plt.show()
"""