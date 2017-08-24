# MRtoCT

Conversion from MRI to CT using GANs.

## Dataset

Download `ct.tar.gz` and `mr_T1.tar.gz` for every patient from
[RIRE][RIREdownload] and unpack archives inside. Note that there will be some
patients which do not possess a CT or MRI. Finally rename `patient_x0y` to
`px0y`.

### Conversion

We are now going to convert the custom data format from [RIRE][RIRE] to dicom
using `...`.

### Registration

Download [SPM12][SPM12] and install according to the [wiki][SPM12install] the
files to `matlab/spm12`.

[RIRE]: http://www.insight-journal.org/rire
[RIREdownload]: http://www.insight-journal.org/rire/download.php

[SPM12]: http://www.fil.ion.ucl.ac.uk/spm/software/spm12/
[SPM12install]: https://en.wikibooks.org/wiki/SPM/Installation_on_64bit_Mac_OS_(Intel)
