# helper methods for generating intensity images
import tifffile as tif
import numpy as np
import os

from model.util import PolarityType

def get_intensity_image (filename, channels_last=False):
    """
    Get an intensity image the specified tif file
    @param filename     : the filename of the raw SAR image in tif format
    @param channels_last: indicate if the image channels are stored last (defaul False)
    @returns the 2-channel intensity image (VH and VV intensity channels)
    """
    image = tif.imread(filename)
    
    if channels_last:
        image = np.rollaxis(image, 2)
    
    bands = image.shape[0]

    if bands != 4:
        raise ValueError("In %s expected 4 bands but found %i" % (filename, bands))
    else:
        intensity = np.zeros((image.shape[1], image.shape[2], 2), dtype=np.float32)
        for band, pol in enumerate(PolarityType):
            i = pol.value*2
            j = i+1       # indices of our desired bands
            intensity[:, :, band] = np.sqrt(image[i]*image[i] + image[j]*image[j])

        return intensity.astype(np.uint8)   # convert to 8-bit before returning

def intensity_images_from_dir (load_dir, save_dir):
    """
    Convert a directory of raw SAR images into 2-channel intensity images and save them
    @param load_dir: the directory from which to load the raw images
    @param save_dir: the directory in which to save the intensity images
    """
    print('BEGIN directory ', load_dir)
    files = os.listdir(load_dir)
    additional_dirs = []
    for f in files:
        if str(f)[-3:] != 'tif':
            path = os.path.join(load_dir, f + '/')
            print("\tSKIPPING:", f, "expected filetype TIFF")
        else:
            try:
                intensity = get_intensity_image(os.path.join(load_dir, f))
            except:
                print("\tERROR converting", f)
            if intensity is not None:
                try:
                    tif.imsave(data=intensity, file=os.path.join(save_dir, f))
                except:
                    print("\tERROR saving ", f, ". Was the output directory created?")
    print("END directory", load_dir)
