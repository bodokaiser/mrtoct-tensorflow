import nibabel

def read(filename):
    return nibabel.load(filename).get_data()