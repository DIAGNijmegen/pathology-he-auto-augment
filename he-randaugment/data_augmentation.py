''' Define data augmentation '''


import matplotlib as mpl
mpl.use('Agg')  # plot figures when no screen available

from augmenters.noise.gaussianbluraugmenter import GaussianBlurAugmenter
from augmenters.color.contrastaugmenter import ContrastAugmenter
from augmenters.color.hedcoloraugmenter import HedColorAugmenter
from augmenters.color.hsbcoloraugmenter import HsbColorAugmenter
from augmenters.noise.additiveguassiannoiseaugmenter import AdditiveGaussianNoiseAugmenter
from augmenters.passthroughaugmenter import PassThroughAugmenter
from augmenters.spatial.elasticagumenter import ElasticAugmenter
from augmenters.spatial.flipaugmenter import FlipAugmenter
from augmenters.spatial.rotate90augmenter import Rotate90Augmenter
from augmenters.spatial.scalingaugmenter import ScalingAugmenter
from augmenters.color import coloraugmenterbase

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

    if augmentation_tag == 'baseline':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
         ]



    elif augmentation_tag == 'hed-light':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50, interpolation_order=1)),
            ('hed_stain', HedColorAugmenter(haematoxylin_sigma_range=(-0.05, 0.05), haematoxylin_bias_range=(-0.05, 0.05),
                                            eosin_sigma_range=(-0.05, 0.05), eosin_bias_range=(-0.05, 0.05),
                                            dab_sigma_range=(-0.05, 0.05), dab_bias_range=(-0.05, 0.05),
                                            cutoff_range=(0.15, 0.85))),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5])),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1)))
         ]

    elif augmentation_tag == 'none':

        augmenters = [
            ('none', PassThroughAugmenter())
        ]

    else:
        raise Exception('Unknown augmentation tag: {tag}'.format(tag=augmentation_tag))

    return augmenters



#----------------------------------------------------------------------------------------------------

class BrightnessEnhAugmenter(coloraugmenterbase.ColorAugmenterBase):

    def __init__(self, sigma_range):

        # Initialize base class.
        #
        super().__init__(keyword='brightness')

        # Initialize members.
        #
        self.__sigma_range = None
        self.__sigma = None

        # Save configuration.
        #
        self.__setsigmaranges(sigma_range=sigma_range)

    def __setsigmaranges(self, sigma_range):

        # Store the setting.
        #
        self.__sigma_range = sigma_range

    def transform(self, patch):

        # Prepare
        rgb_space = True if patch.shape[0] == 3 else False
        patch_image = np.transpose(a=patch, axes=(1, 2, 0))
        image = Image.fromarray(np.uint8(patch_image if rgb_space else patch_image[:, :, 0]))

        # Change brightness
        enhanced_image = ImageEnhance.Brightness(image)
        enhanced_image = enhanced_image.enhance(self.__sigma)

        # Convert back
        patch_enhanced = np.asarray(enhanced_image)
        patch_enhanced = patch_enhanced if rgb_space else patch_enhanced[:, :, np.newaxis]
        patch_enhanced = patch_enhanced.astype(dtype=np.uint8)
        patch_enhanced = np.transpose(a=patch_enhanced, axes=(2, 0, 1))

        return patch_enhanced

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize sigma for each channel.
        #
        self.__sigma = np.random.uniform(low=self.__sigma_range[0], high=self.__sigma_range[1], size=None)


#----------------------------------------------------------------------------------------------------

class ContrastEnhAugmenter(coloraugmenterbase.ColorAugmenterBase):

    def __init__(self, sigma_range):

        # Initialize base class.
        #
        super().__init__(keyword='contrast_enh')

        # Initialize members.
        #
        self.__sigma_range = None
        self.__sigma = None

        # Save configuration.
        #
        self.__setsigmaranges(sigma_range=sigma_range)

    def __setsigmaranges(self, sigma_range):

        # Store the setting.
        #
        self.__sigma_range = sigma_range

    def transform(self, patch):

        # Prepare
        rgb_space = True if patch.shape[0] == 3 else False
        patch_image = np.transpose(a=patch, axes=(1, 2, 0))
        image = Image.fromarray(np.uint8(patch_image if rgb_space else patch_image[:, :, 0]))

        # Change brightness
        enhanced_image = ImageEnhance.Contrast(image)
        enhanced_image = enhanced_image.enhance(self.__sigma)

        # Convert back
        patch_enhanced = np.asarray(enhanced_image)
        patch_enhanced = patch_enhanced if rgb_space else patch_enhanced[:, :, np.newaxis]
        patch_enhanced = patch_enhanced.astype(dtype=np.uint8)
        patch_enhanced = np.transpose(a=patch_enhanced, axes=(2, 0, 1))

        return patch_enhanced

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize sigma for each channel.
        #
        self.__sigma = np.random.uniform(low=self.__sigma_range[0], high=self.__sigma_range[1], size=None)


#----------------------------------------------------------------------------------------------------

class ColorEnhAugmenter(coloraugmenterbase.ColorAugmenterBase):

    def __init__(self, sigma_range):

        # Initialize base class.
        #
        super().__init__(keyword='color_enh')

        # Initialize members.
        #
        self.__sigma_range = None
        self.__sigma = None

        # Save configuration.
        #
        self.__setsigmaranges(sigma_range=sigma_range)

    def __setsigmaranges(self, sigma_range):

        # Store the setting.
        #
        self.__sigma_range = sigma_range

    def transform(self, patch):

        # Prepare
        rgb_space = True if patch.shape[0] == 3 else False
        patch_image = np.transpose(a=patch, axes=(1, 2, 0))
        image = Image.fromarray(np.uint8(patch_image if rgb_space else patch_image[:, :, 0]))

        # Change brightness
        enhanced_image = ImageEnhance.Color(image)
        enhanced_image = enhanced_image.enhance(self.__sigma)

        # Convert back
        patch_enhanced = np.asarray(enhanced_image)
        patch_enhanced = patch_enhanced if rgb_space else patch_enhanced[:, :, np.newaxis]
        patch_enhanced = patch_enhanced.astype(dtype=np.uint8)
        patch_enhanced = np.transpose(a=patch_enhanced, axes=(2, 0, 1))

        return patch_enhanced

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize sigma for each channel.
        #
        self.__sigma = np.random.uniform(low=self.__sigma_range[0], high=self.__sigma_range[1], size=None)


