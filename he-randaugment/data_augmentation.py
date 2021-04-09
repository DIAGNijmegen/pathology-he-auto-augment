''' Define data augmentation '''


import matplotlib as mpl
mpl.use('Agg')  # plot figures when no screen available
from augmenters.passthroughaugmenter import PassThroughAugmenter
from augmenters.spatial.flipaugmenter import FlipAugmenter
from augmenters.spatial.rotate90augmenter import Rotate90Augmenter

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import imread
from scipy.misc import imsave
from glob import glob
from os.path import join, basename, dirname, exists
import os
from tqdm import tqdm
import skimage.color
from PIL import Image, ImageEnhance
import shutil
import time

class DataAugmenter(object):

    def __init__(self, augmentation_tag):
        #get list of augs corresponding to the tag
        self.augmenters = define_augmenters(augmentation_tag)
        #get num of augs corresponding to the tag
        self.n_augmenters = len(self.augmenters)

    def augment(self, patch):
        #switch channel to the first dim
        
        patch = patch.transpose((2, 0, 1))
        #loop through a list of augs
        for k in range(self.n_augmenters):
            #select an aug and randomize its sigma
            self.augmenters[k][1].randomize()
            # t = time.time()
            patch = self.augmenters[k][1].transform(patch)
            # print('{t} took {s} logs.'.format(t=self.augmenters[k][0], s=np.log(time.time() - t)), flush=True)  # TODO

        patch = patch.transpose((1, 2, 0))

        return patch

def rgb_to_gray(batch):

    new_batch = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], 1))
    for i in range(batch.shape[0]):
        new_batch[i, :, :, 0] = skimage.color.rgb2grey(batch[i, ...])

    new_batch = (new_batch * 255.0).astype('uint8')
    return new_batch

def define_augmenters(augmentation_tag):
    #Note that this is not the baseline from paper
    if augmentation_tag == 'baseline':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
         ]



    elif augmentation_tag == 'none':

        augmenters = [
            ('none', PassThroughAugmenter())
        ]

    else:
        raise Exception('Unknown augmentation tag: {tag}'.format(tag=augmentation_tag))

    return augmenters



#----------------------------------------------------------------------------------------------------
