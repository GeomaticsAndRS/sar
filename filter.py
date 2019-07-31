#!/usr/bin/env python3

import numpy as np
import argparse
import os
from math import ceil, floor
from tifffile import imwrite
from keras import backend as K

from model.util import (
    denormalize,
    mean_shift,
    MeanShiftType, 
    normalize, 
    PolarityType,
    print_image_stats,
    stretch
)
from predictor import predictor
from tif2intensity import get_intensity_image

def filter_image(
    image, 
    model, 
    logspace_model=False, 
    stride=128, 
    center_weight=True,
    mean_correction=MeanShiftType.Smart):
    """
    Despeckles an input image
    @param image          : The image to be filtered
    @param model          : The model to filter the image (instance of predictor.predictor)
    @param logspace_model : Boolean indicating if model requires input in logspace
    @param stride         : Stride between image patches (must be <= patch size, i.e. 256)
    @param center_weight  : Whether to apply center-weighted averaging to output patches to ameliorate hard edges
    @param mean_correction: util.MeanShiftType indicating when to mean-shift results, defaults to Smart
    @returns              : The filtered image in linear space as np.array of dtype uint8
    """
    CHANNELS = 2
    PATCH_SIZE = 256
    assert stride <= PATCH_SIZE, "Stride cannot exceed the patch dimensions (%i)" % PATCH_SIZE
    
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    rows, columns, channels = image.shape
    
    assert channels == CHANNELS, "Unexpected number of image channels. Expected %i but received %i" % (CHANNELS, channels)
    
    # array for the filtered image
    output = np.zeros((rows, columns, CHANNELS), dtype=np.float32)
    # array for tracking how many times each pixel has been processed, for averaging
    output_count = np.zeros((rows, columns, CHANNELS), dtype=np.float32)
    
    # create mask for center-weighting
    mask = np.ones((PATCH_SIZE, PATCH_SIZE, CHANNELS), dtype=np.float32)
    if center_weight:
        max_array_dist = np.sqrt(2) * PATCH_SIZE // 2
        for r in range(PATCH_SIZE):
            for c in range(PATCH_SIZE):
                mask[r,c] = (max_array_dist - np.sqrt((r - PATCH_SIZE // 2)**2 + (c - PATCH_SIZE // 2)**2) + 1)**2
        mask /= max_array_dist**2
        mask = stretch(mask, 5.0, 1.0) # for numerical stability
    
    image = normalize(image)
    
    # iteratively predict despeckled 256x256 patches
    r_iterations = ceil(rows / stride)
    c_iterations = ceil(columns / stride)
    iterations = r_iterations * c_iterations
    
    for r in range(r_iterations):
        r0 = r * stride
        r1 = r0 + PATCH_SIZE
        if r1 > rows:
            r0 = rows - PATCH_SIZE
            r1 = rows
        
        for c in range(c_iterations):
            c0 = c * stride
            c1 = c0 + PATCH_SIZE
            if c1 > columns:
                c0 = columns - PATCH_SIZE
                c1 = columns
            
            patch = image[r0:r1, c0:c1]
            
            output_patch = model.predict(patch[np.newaxis,...], False)[0,...]
            
            if center_weight:
                output_patch *= mask
            
            output[r0:r1, c0:c1] += output_patch
            output_count[r0:r1, c0:c1] += mask
            
            completed_iterations = c + r * c_iterations + 1
            percent_iterations = completed_iterations / iterations
            progress = floor(20 * percent_iterations)
            bar = ""
            while len(bar) < progress: bar += "#"
            while len(bar) < 20: bar += " "
            print("\r[%s] %i%%" % (bar, ceil(100 * percent_iterations)), end="")
    
    print("") # trigger line break
    
    # average pixel values by the number of times they were processed
    output /= output_count
    
    # correct the image-mean
    if (
        mean_correction == MeanShiftType.Always or 
        (logspace_model and mean_correction == MeanShiftType.Smart)
    ):
         output = mean_shift(output, np.mean(image))
    
    output = denormalize(output)
    return output.astype(np.uint8)

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("input", type=str,
                        help="tif image or directory of tif images")
    
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output file or directory")
    
    parser.add_argument("--channels_last", action="store_true",
                        help="indicate if the image channels are stored last")
    
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="which model to use for processing [lin|log]")
    
    parser.add_argument("--mean_correction", type=str, default="smart",
                        help="when to apply mean-correction to images [smart|always|never]")
    
    parser.add_argument("--no_weighting", action="store_true",
                        help="do not center-weight image patches")
    
    parser.add_argument("-r", action="store_true",
                        help="recursively process subdirectories")
    
    parser.add_argument("-s", "--stride", type=int, default=192,
                        help="stride when processing images (default 192)")
    
    parser.add_argument("--single_channel_output", action="store_true",
                        help="create a separate file for each channel (polarity)")
    
    args = parser.parse_args()
    validate_args(args)
    return args

def validate_args(args):
    assert args.model in ["lin", "log"], "Unexpected --model. Must be [lin|log]."
    assert os.path.exists(args.input), "Input path does not exist"
    assert (
        os.path.isdir(args.input) or _is_tiffile(args.input)
    ), "Unexpected input. Must be a directory or tif file."
    assert args.stride > 0 and args.stride <= 256, "Stride must be in range [1,256]."
    assert args.mean_correction in ["smart", "always", "never"], "Unexpected --mean_correction. Must be [smart|always|never]."
    
def _is_tiffile(file):
    if len(file) > 3 and (file[-4:] == '.tif' or file[-5:] == '.tiff'):
        return True
    return False

def _get_weights(logspace_model):
    if logspace_model:
        return "./resources/log.hdf5"
    return "./resources/lin.hdf5"

def _get_mean_correction(mean_correction):
    if (mean_correction == "always"):
        return MeanShiftType.Always
    elif (mean_correction == "never"):
        return MeanShiftType.Never
    return MeanShiftType.Smart

def main():
    # Gather arguments
    args = get_args()
    infile = args.input
    input_isdir = os.path.isdir(infile)
    outfile = args.output
    output_isfile = _is_tiffile(outfile)
    assert(input_isdir != output_isfile), "Output type must match input type [directory or tif file]"
    logspace_model = args.model == "log"
    center_weight = not args.no_weighting
    mean_correction = _get_mean_correction(args.mean_correction)
    channels_last = args.channels_last
    single_channel_output = args.single_channel_output    
    stride = args.stride
    
    # Prepare our predictor
    weights = _get_weights(logspace_model)
    model = predictor(weights, logspace=logspace_model)
    
    # Depeckel and save all files
    if (input_isdir):
        files = os.listdir(infile)
        while len(files) > 0:
            imgfile = files.pop(0)
            if _is_tiffile(imgfile):
                print("BEGIN", imgfile)
                intensity_image = get_intensity_image(os.path.join(infile, imgfile), channels_last)
                filtered_image = filter_image(
                                    intensity_image, 
                                    model, 
                                    logspace_model=logspace_model,
                                    stride=stride,
                                    center_weight=center_weight,
                                    mean_correction=mean_correction
                                )
                if single_channel_output:
                    for polarity in PolarityType:
                        imwrite(os.path.join(outfile, imgfile[:-4] + "_" + polarity.name + ".tif"), filtered_image[:, :, polarity.value])
                else:
                    imwrite(os.path.join(outfile, imgfile), filtered_image)
                print("END", imgfile)
            elif os.path.isdir(os.path.join(infile, imgfile)):
                if (args.r):
                    if not os.path.exists(os.path.join(outfile, imgfile)):
                        os.makedirs(os.path.join(outfile, imgfile))
                    for _imgfile in os.listdir(os.path.join(infile, imgfile)):
                        files.append(os.path.join(imgfile, _imgfile))
            else:
                print("Skipping unexpected file type %s" % imgfile)
    else:
        intensity_image = get_intensity_image(infile, channels_last)
        filtered_image = filter_image(
                            intensity_image, 
                            model, 
                            logspace_model=logspace_model,
                            stride=stride,
                            center_weight=center_weight,
                            mean_correction=mean_correction
                        )
        # Write output to file
        if single_channel_output:
            for polarity in PolarityType:
                imwrite(outfile[:-4] + "_" + polarity.name + ".tif", filtered_image[:, :, polarity.value])
        else:
            imwrite(outfile, filtered_image)
    
    K.clear_session()
    
    print(" ----- JOB COMPLETE ----- ")

if __name__ == '__main__':
    main()
