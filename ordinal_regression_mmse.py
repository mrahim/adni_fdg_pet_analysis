# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 16:29:32 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
from fetch_data import fetch_adni_petmr, fetch_adni_masks, set_features_base_dir
from mord import OrdinalLogistic
from sklearn.cross_validation import ShuffleSplit
from nilearn.masking import apply_mask

FEAT_DIR = set_features_base_dir()

dataset = fetch_adni_petmr()
mask = fetch_adni_masks()
pet_files = dataset['pet']
Y = dataset['mmscores']
x = apply_mask(pet_files, mask['mask_petmr'])
y = np.array(Y - np.min(Y), dtype=np.int)


olr = OrdinalLogistic(verbose=2)
ss = ShuffleSplit(len(y), n_iter=1, test_size=.3)
for train, test in ss:
    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]
    
    olr.fit(x_train, y_train)
    olr.score(x_test, y_test)