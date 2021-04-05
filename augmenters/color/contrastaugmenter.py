"""
This file contains a class for augmenting patches from whole slide images with contrast changes.
"""

from . import coloraugmenterbase as dptcoloraugmenterbase

from ...errors import augmentationerrors as dptaugmentationerrors

import skimage.exposure
import skimage.color
import numpy as np

#----------------------------------------------------------------------------------------------------

class ContrastAugmenter(dptcoloraugmenterbase.ColorAugmenterBase):
    """Apply contrast enhancements on the patch."""

    def __init__(self, sigma_range):
        """
        Initialize the object.

        Args:
            sigma_range (tuple): Range for contrast adjustment from the [-1.0, 1.0] range. For example: (-0.4, 0.4).

        Raises:
            InvalidContrastSigmaRangeError: The contrast adjustment range is not valid.
        """

        # Initialize base class.
        #
        super().__init__(keyword='contrast')

        # Initialize members.
        #
        self.__sigma_range = None  # Configured sigma range for contrast enhancement.
        self.__sigma = None        # Randomized sigma.

        # Save configuration.
        #
        self.__setsigmarange(sigma_range=sigma_range)

    def __setsigmarange(self, sigma_range):
        """
        Set the interval.

        Args:
            sigma_range (tuple): Range for contrast adjustment.

        Raises:
            InvalidContrastSigmaRangeError: The contrast adjustment range is not valid.
        """

        # Check the interval.
        #
        if len(sigma_range) != 2 or sigma_range[1] < sigma_range[0] or sigma_range[0] < -1.0 or 1.0 < sigma_range[1]:
            raise dptaugmentationerrors.InvalidContrastSigmaRangeError(sigma_range)

        # Store the settings.
        #
        self.__sigma_range = list(sigma_range)
        self.__sigma = sigma_range[0]

    def transform(self, patch):
        """
        Apply contrast deformation on the patch.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.
        """

        # Reorder the patch to channel last format.
        #
        patch_image = np.transpose(a=patch, axes=(1, 2, 0))

        # Augment the contrast.
        #
        patch_center = skimage.color.rgb2gray(rgb=patch_image).mean() * 255.0
        patch_range = (self.__sigma * patch_center, 255.0 - self.__sigma * (255.0 - patch_center))
        patch_contrast = skimage.exposure.rescale_intensity(image=patch_image, in_range=patch_range, out_range='dtype')

        # Order back to channels first order.
        #
        patch_transformed = np.transpose(a=patch_contrast, axes=(2, 0, 1))

        return patch_transformed

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize sigma.
        #
        self.__sigma = np.random.uniform(low=self.__sigma_range[0], high=self.__sigma_range[1], size=None)
