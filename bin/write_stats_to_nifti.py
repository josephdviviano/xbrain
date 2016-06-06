#!/usr/bin/env python

import numpy as np
import nibabel as nib

def write_nifti(name, mask, dims, aff, hdr):

    # import stats from R
    data = np.genfromtxt('{}.csv'.format(name), skip_header=1)

    # init output matrix
    output = np.zeros((dims[0]*dims[1]*dims[2], 5))

    # load in stats ROI by ROI
    for i, roi in enumerate(data[:,5]):
        idx = np.where(mask == roi)[0]
        output[idx, 0] = data[i, 2] # x-y
        output[idx, 1] = 1-data[i, 0] # p (inverse for ez thresholding)
        output[idx, 2] = data[i, 1] # t
        output[idx, 3] = data[i, 3] # x
        output[idx, 4] = data[i, 4] # y

    # write out data
    output = output.reshape(dims[0], dims[1], dims[2], 5)
    output = nib.nifti1.Nifti1Image(output, aff, header=hdr)
    output.to_filename('{}.nii.gz'.format(name))

def main():

    # import mask
    mask = nib.load('anat_MNI_rois-shen.nii.gz')
    aff = mask.get_affine() # use this to write image out later
    hdr = mask.get_header() #
    dims = mask.shape
    mask = mask.get_data()
    mask = mask.reshape(dims[0]*dims[1]*dims[2], 1)

    # write out nifti files
    write_nifti('betas-condition', mask, dims, aff, hdr)
    write_nifti('correlations-condition', mask, dims, aff, hdr)
    write_nifti('correlations-group', mask, dims, aff, hdr)
    write_nifti('correlations-int-im', mask, dims, aff, hdr)
    write_nifti('correlations-int-ob', mask, dims, aff, hdr)

if __name__ == '__main__':
    main()

