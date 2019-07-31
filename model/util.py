import numpy as np
from enum import Enum

LIN_MAX = 255.0
LOG_MAX = np.log10(LIN_MAX + 1)

class MeanShiftType(int, Enum):
    Smart  = 0  # apply meanshift whenever filtering is done in logspace
    Always = 1
    Never  = -1

class PolarityType(int, Enum):
    vh = 0
    vv = 1

polarity_types = ['vh', 'vv']

def lin2log(image, max_val=0.5, min_val=-0.5):
    """
    Convert pixel values from linear space to logarithmic space
    @param image  : an image as np.array
    @max_val      : the maximum possible value of a pixel
                     defaults to 0.5 on the assumption image is normalized
    @param min_val: the minimum possible value of a pixel
                    defaults to -0.5 on the assumption image is normalized
    @returns      : the image converted to logspace as a numpy array
    """
    image, dtype = _get_float32(image)
    img_min = np.min(image)
    img_max = np.max(image)
    stretch_max = LIN_MAX * (img_max - min_val) / (max_val - min_val) 
    stretch_min = LIN_MAX * (img_min - min_val) / (max_val - min_val) 
    image = stretch(image, max_val=stretch_max, min_val=stretch_min)
    image = np.log10(image + 1)
    image = image / LOG_MAX - 0.5
    image = _revert_dtype(image, dtype)
    return image

def log2lin(image, max_val=0.5, min_val=-0.5):
    """
    Convert pixel values from logarithmic space to linear space
    @param image   : an image as np.array
    @max_val       : the maximum possible value of a pixel
                     defaults to 0.5 on the assumption image is normalized
    @param min_val : the minimum possible value of a pixel
                     defaults to -0.5 on the assumption image is normalized
    @returns       : the image converted to linspace as a numpy array
    """
    image, dtype = _get_float32(image)
    img_min = np.min(image)
    img_max = np.max(image)
    stretch_max = LOG_MAX * (img_max - min_val) / (max_val - min_val)
    stretch_min = LOG_MAX * (img_min - min_val) / (max_val - min_val) 
    image = stretch(image, max_val=stretch_max, min_val=stretch_min)
    image = np.power(10, image) - 1
    image = image / LIN_MAX - 0.5
    image = _revert_dtype(image, dtype)
    return image

def stretch(image, max_val=255, min_val=0):
    """
    Stretch the pixel values of an image to the specified range
    @param image : an image as np.array
    @max_val     : the maximum possible value of a pixel
                   defaults to 255
    @min_val     : the minimum possible value of a pixel
                   defaults to 0
    """
    image, dtype = _get_float32(image)
    img_min = np.min(image)
    img_max = np.max(image)
    image_stretched = min_val + (max_val - min_val) * (image - img_min) / (img_max - img_min)
    image_stretched = _revert_dtype(image_stretched, dtype)
    return image_stretched

def batch_stretch(images, max_val=255, min_val=0):
    """
    Stretch the pixel values of a set of images to the specified range
    @param images : an set of images as a list of np.arrays
    @max_val      : the maximum possible value of a pixel
                    defaults to 255
    @min_val      : the minimum possible value of a pixel
                    defaults to 0
    """
    image, dtype = _get_float32(image)
    set_min = np.Inf
    set_max = -np.Inf
    for img in images:
        if np.min(img) < set_min: set_min = np.min(img)
        if np.max(img) < set_max: set_max = np.max(img)
    stretched = []
    for img in images:
        img_stretched = min_val + (max_val - min_val) * (img - set_min) / (set_max - set_min)
    image = _revert_dtype(image, dtype)
    return images

def normalize(image, min_val=0, max_val=255):
    """
    Normalize pixel intensity into the range [-0.5, 0.5]
    @param image  : the image to normalize as a numpy array
    @param min_val: the minimum possible pixel value of the provided image
    @param max_val: the maximum possible pixel value of the provided image
    @returns      : the normalized image as a numpy array
    """
    image, dtype = _get_float32(image)
    normalized = (image - min_val) / (max_val - min_val) - 0.5
    normalized = _revert_dtype(normalized, dtype)
    return normalized

def denormalize(image, min_val=-0.5, max_val=0.5):
    """
    Revert a normalized image so that pixel values are in the range [0, 255]
    @param image  : the image to denormalize as a numpy array
    @param min_val: the minimum possible pixel value of the provided image
    @param max_val: the maximum possible pixel value of the provided image
    @returns      : the denormalized image as a numpy array
    """
    image, dtype = _get_float32(image)
    denormalized = (image - min_val) / (max_val - min_val) * 255.0
    denormalized = _revert_dtype(denormalized, dtype)
    return denormalized

def mean_shift(image, target_mean, min_val=-0.5, max_val=0.5):
    """
    Forces the mean to match target_mean
    @param image: the image for which to adjust the mean as a numpy array
    @target_mean: the mean value the image should have
    @max_val    : the maximum possible value of a pixel
                  defaults to 0.5
    @min_val    : the minimum possible value of a pixel
                  defaults to -0.5
    @returns    : the mean-adjusted image as a numpy array
    """
    
    image, dtype = _get_float32(image)
    
    img_mean = np.mean(image)
    mean_difference = img_mean - target_mean
    
    image = image - mean_difference
    
    # symmetrically scale to keep in bounds
    max_range = max_val - min_val
    overflow_top = max([0, np.max(image) - max_val])
    overflow_bottom = min([0, np.min(image) - min_val])
    max_overflow = max([overflow_top, abs(overflow_bottom)])
    
    image /= max_range / (max_range - max_overflow)
    
    image = _revert_dtype(image, dtype)
    return image

def print_image_stats(img):
    print(
        'Min',    np.min(img),
        'Max',    np.max(img),
        'Mean',   np.mean(img),
        'Median', np.median(img),
        'Std.',   np.std(img)
    )
    

def _get_float32(image):
    """
    Convert pixel values to float32
    @param image  : the image for which to convert pixel values as a numpy array
    @returns image: the image as a numpy array of float32 values
    @returns dtype: the dtype of the array that was passed to the function
    """
    dtype = image.dtype
    if dtype != np.float32:
        image = image.astype(np.float32)
    return image, dtype

def _revert_dtype(image, dtype):
    """
    Converts pixel values to a specified dtype
    @param image: the image for which to convert pixel values as a numpy array
    @param dtype: the dtype to convert the image to
    @returns    : the image as a numpy array of float32 values
    
    """
    current_dtype = image.dtype
    if current_dtype != dtype:
        image = image.astype(dtype)
    return image
