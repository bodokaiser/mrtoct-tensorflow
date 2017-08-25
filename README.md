# MRtoCT

Conversion from MRI to CT using GANs.

## Dataset

Download [RIRE][RIRE] dataset and unpack to `data/raw`.

### Registration

Download and install [SPM12][SPM12] to `matlab/spm12` then run
`matlab/register.m` which will convert the custom format of the dataset
first to dicom then to NIfTI-1 which is used by SPM12 to coregister both
volumes. Final results are written to `data/nii`.

### Conversion

To use `pix2pix` we need image pairs we can create them from `data/nii` with
`matlab/convert.m` in `data/png`. Manually remove slices with reduced MRI.

To use the present tensorflow implementation you need to convert data from
`data/nii` to `data/tfr` in tfrecord format with `python convert.py`.

[RIRE]: http://www.insight-journal.org/rire
[SPM12]: http://www.fil.ion.ucl.ac.uk/spm/software/spm12/
