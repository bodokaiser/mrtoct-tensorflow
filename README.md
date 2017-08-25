# MRtoCT

Conversion from MRI to CT using GANs.

## Dataset

Download `ct.tar.gz` and `mr_T1.tar.gz` for every patient from
[RIRE][RIREdownload] and unpack archives inside to a `data` directory.
Rename `patient_x0y` to `px0y`.

### Registration

Download [SPM12][SPM12] and install according to the [wiki][SPM12install] the
files to `matlab/spm12` then run `matlab/process.m`.

This will convert the custom format of the dataset first to dicom then to
NIfTI-1 which is used by SPM12, from there on we coregister both volumes.

### Conversion

If we want to use `pix2pix` we need image pairs wheras our own network uses
the tfrecord format.

Conversion scripts will be provided.

[RIRE]: http://www.insight-journal.org/rire
[RIREdownload]: http://www.insight-journal.org/rire/download.php

[SPM12]: http://www.fil.ion.ucl.ac.uk/spm/software/spm12/
[SPM12install]: https://en.wikibooks.org/wiki/SPM/Installation_on_64bit_Mac_OS_(Intel)
