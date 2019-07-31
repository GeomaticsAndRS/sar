import os
import random
import numpy as np
import tifffile as tif
from pandas import DataFrame, concat
from datetime import date
from keras.utils import Sequence
from util import *

class imgloader(Sequence):

    # indices for accessing the list of files
    iset =   1
    idate =  4
    ifile =  0

    def __init__(self, load_dir, batch_size=4, min_date_separation=6, logspace=False, verbose=False):
        """
        @param load_dir: the directory where the prepared image patches are stored
        @param batch_size: the number of image pairs to be returned.
        @param min_date_separation: the minimum timespan between image acquisitions in days. default 0.
        """
        self.load_dir = load_dir
        self.batch_size = batch_size
        self.min_date_separation = min_date_separation
        self.logspace = logspace
        self.verbose = verbose
        sub_dirs = os.listdir(load_dir)
        if logspace: print("Imgloader is converting inputs to logspace")
        print("Loaded directory. %i sets found." % len(sub_dirs))
        self.sets = DataFrame([], columns=['file', 'set', 'set_int', 'datestr', 'date'])
        for sub_dir in sub_dirs:
            path = os.path.join(load_dir, sub_dir)
            files = os.listdir(path)
            imgs = DataFrame(os.listdir(path), columns={'file'})
            imgs = imgs[imgs.file.str[-7:] != 'avg.tif'] # filter out our avg images
            imgs['set'] = sub_dir
            imgs['set_int'] = int(sub_dir)

            # extract date information from filename
            datestr_func = lambda f: f.split('_')[2][:8]
            imgs['datestr'] = imgs['file'].apply(lambda f: f.split('_')[1][:8])
            date_func = lambda d: date(int(d[:4]), int(d[4:6]), int(d[6:8]))
            imgs['date'] = imgs['datestr'].apply(date_func)

            # add to our sets and print overview
            self.sets = concat([self.sets, imgs])
            img_count = imgs.file.nunique()
            print("Set %s contains %i images"
                  % (sub_dir, img_count))
        self.imgcount = self.sets.shape[0]

    def __len__(self):
        return self.imgcount // self.batch_size

    def __len__(self):
        return self.imgcount

    def __getitem__(self, idx):
        return self.next_batch()

    def next_batch(self):
        """
        @return x,y: matching image patches. all pixels are in range [-0.5,0.5]
        """
        filelist = self.sets[self.sets['set_int'] > 0].values.tolist()
        assert(self.batch_size < len(filelist)/2)
        random.shuffle(filelist)
        x = np.zeros((self.batch_size, 256, 256, 2), dtype=np.float32)
        y = np.zeros((self.batch_size, 256, 256, 2), dtype=np.float32)
        pair_count = 0
        while pair_count < self.batch_size and len(filelist) > 1:      # more breakoff conditions?
            xfile = filelist.pop()
            # check if there is another picture in the batch that meets the date separation
            for yfile in filelist:                # manually iterate so we can complete in O(n)
                if xfile[self.iset] == yfile[self.iset]:
                    datediff = xfile[self.idate] - yfile[self.idate]
                    if abs(datediff.days) >= self.min_date_separation:
                        pathx = os.path.join(self.load_dir, xfile[self.iset], xfile[self.ifile])
                        pathy = os.path.join(self.load_dir, yfile[self.iset], yfile[self.ifile])

                        if self.verbose:
                            print('loading pair %s, %s' % (os.path.join(xfile[self.iset], xfile[self.ifile]), os.path.join(yfile[self.iset], yfile[self.ifile])))

                        imgx, imgy = self.get_random_patch(pathx, pathy)
                        imgx = normalize(imgx)
                        imgy = normalize(imgy)
                        if self.logspace:
                            imgx = lin2log(imgx)
                            imgy = lin2log(imgy)

                        x[pair_count] = imgx
                        y[pair_count] = imgy
                        pair_count += 1
                        break
        assert(pair_count == self.batch_size) # in case we ran out of images
        return (x,y)

    def get_validation_patch(self):
        """
        @return x,y: matching image patches from our validation set, normalized in range [-0.5,0.5]
        """
        filelist = self.sets[self.sets['set_int'] < 0].values.tolist()
        random.shuffle(filelist)
        xfile = filelist[0]
        pathx = os.path.join(self.load_dir, str(xfile[self.iset]), str(xfile[self.ifile]))
        pathy = os.path.join(self.load_dir, str(xfile[self.iset]), 'avg.tif')

        if self.verbose:
            print('Loading validation image %s' % os.path.join(str(xfile[self.iset]), str(xfile[self.ifile])), end=' ')

        x,y = self.get_random_patch(pathx, pathy)
        x = normalize(x)
        y = normalize(y)
        if self.logspace:
            x = lin2log(x)
            y = lin2log(y)
        return x, y

    def get_random_patch(self, pathx, pathy):
        """
        @param pathx: the path to file x
        @param pathy: the path to file y
        @return imgx, imgy: matching patches from random x,y coordinates
        """
        PATCH_SIZE = 256

        _imgx = tif.imread(pathx).astype(np.float32)
        _imgy = tif.imread(pathy).astype(np.float32)
        
        w, h, _ = _imgx.shape
        w2, h2, _ = _imgy.shape
        if w2 < w: w = w2
        if h2 < h: h = h2

        x0 = random.randint(0, w-PATCH_SIZE)
        y0 = random.randint(0, h-PATCH_SIZE)

        if self.verbose:
            print('Using random patch at [%i, %i]' % (x0, y0))

        imgx = _imgx[x0:x0+PATCH_SIZE, y0:y0+PATCH_SIZE, :]
        imgy = _imgy[x0:x0+PATCH_SIZE, y0:y0+PATCH_SIZE, :]

        return imgx[np.newaxis,...], imgy[np.newaxis,...]
