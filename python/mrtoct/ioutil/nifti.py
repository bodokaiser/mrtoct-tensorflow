import nibabel as nib

def read_nifti(filename):
    return nib.load(filename).get_data()
