#!/bin/usr/env python

import numpy as np
import nibabel as nib
import csv
import glob

atlas = '/projects/jdv/data/imob/working/assets/anat_MNI_shen_268-resamp.nii.gz'
contrast = 19
output = 'glm-outputs.csv'

# get subject files
hc_im = glob.glob('DTI_CMH_H*_im_glm_1stlvl.nii'); hc_im.sort()
hc_ob = glob.glob('DTI_CMH_H*_ob_glm_1stlvl.nii'); hc_ob.sort()
sz_im = glob.glob('DTI_CMH_S*_im_glm_1stlvl.nii'); sz_im.sort()
sz_ob = glob.glob('DTI_CMH_S*_ob_glm_1stlvl.nii'); sz_ob.sort()

# init output csv
with open(output, 'wb') as f:
    c = csv.writer(f)
    c.writerow(['id','group','condition','roi','beta'])

# load atlas, get data
a = nib.load(atlas).get_data()
rois = np.unique(a[a>0])

subjid = 1

group = 1 # healthy
for i in range(len(hc_im)):

    # get betas from imitate, observe data
    im = nib.load(hc_im[i]).get_data()
    im = im[:, :, :, 0, contrast]
    ob = nib.load(hc_ob[i]).get_data()
    ob = ob[:, :, :, 0, contrast]

    # collect average beta per ROI
    for roi in rois:
        idx = np.where(a == roi)

        # imitate
        b = float(np.mean(im[idx]))
        with open(output, 'a') as f:
            c = csv.writer(f)
            c.writerow([subjid, group, 1, int(roi), b])

        # observe
        b = float(np.mean(ob[idx]))
        with open(output, 'a') as f:
            c = csv.writer(f)
            c.writerow([subjid, group, 2, int(roi), b])

    subjid = subjid + 1 # iterate subject counter

group = 2 # schizophrenia
for i in range(len(sz_im)):

    # get betas from imitate, observe data
    im = nib.load(sz_im[i]).get_data()
    im = im[:, :, :, 0, contrast]
    ob = nib.load(sz_ob[i]).get_data()
    ob = ob[:, :, :, 0, contrast]

    # collect average beta per ROI
    for roi in rois:
        idx = np.where(a == roi)

        # imitate
        b = float(np.mean(im[idx]))
        with open(output, 'a') as f:
            c = csv.writer(f)
            c.writerow([subjid, group, 1, int(roi), b])

        # observe
        b = float(np.mean(ob[idx]))
        with open(output, 'a') as f:
            c = csv.writer(f)
            c.writerow([subjid, group, 2, int(roi), b])

    subjid = subjid + 1 # iterate subject counter


