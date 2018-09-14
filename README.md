# MRtoCT

Tensorflow model for MRI to CT synthesis.

## Dataset

See [1][1] to download, extract, convert and coregister data.

Then use `python convert.py` to convert `.nii` to `.tfrecord` format.

## Install

Clone this repository, install python `3.6.6` and then the required packages
via `pip install -r requirements.txt`, this also ensures that tensorflow will
be installed as `1.5.1`. Newer tensorflow versions contain api changes, thus
will give an error. Feel free to send a PR to make the code compatible with
newer versions of tensorflow.

## Training

    python train_unet.py \
      --inputs-path data/tfrecord/training/mr.tfrecord \
      --targets-path data/tfrecord/training/ct.tfrecord \
      --checkpoint-path results/unet

[1]: https://github.com/bodokaiser/mrtoct-scripts
