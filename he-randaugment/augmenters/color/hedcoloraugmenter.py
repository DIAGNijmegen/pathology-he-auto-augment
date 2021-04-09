"""
This file contains a class for augmenting patches from whole slide images by applying color correction in HED color space.
"""
from utils.custom_hed_transform import rgb2hed, hed2rgb
from . import coloraugmenterbase as dptcoloraugmenterbase

import numpy as np

#----------------------------------------------------------------------------------------------------

class HedColorAugmenter(dptcoloraugmenterbase.ColorAugmenterBase):
    """Apply color correction in HED color space on the RGB patch."""

    def __init__(self, haematoxylin_sigma_range, haematoxylin_bias_range, eosin_sigma_range, eosin_bias_range, dab_sigma_range, dab_bias_range, cutoff_range):
        """
        Initialize the object. For each channel the augmented value is calculated as value = value * sigma + bias

        Args:
            haematoxylin_sigma_range (tuple, None): Adjustment range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.1, 0.1).
            haematoxylin_bias_range (tuple, None): Bias range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.2, 0.2).
            eosin_sigma_range (tuple, None): Adjustment range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
            eosin_bias_range (tuple, None) Bias range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
            dab_sigma_range (tuple, None): Adjustment range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
            dab_bias_range (tuple, None): Bias range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
            cutoff_range (tuple, None): Patches with mean value outside the cutoff interval will not be augmented. Values from the [0.0, 1.0] range. The RGB channel values are from the same range.

        Raises:
            InvalidHaematoxylinSigmaRangeError: The sigma range for Haematoxylin channel adjustment is not valid.
            InvalidHaematoxylinBiasRangeError: The bias range for Haematoxylin channel adjustment is not valid.
            InvalidEosinSigmaRangeError: The sigma range for Eosin channel adjustment is not valid.
            InvalidEosinBiasRangeError: The bias range for Eosin channel adjustment is not valid.
            InvalidDabSigmaRangeError: The sigma range for DAB channel adjustment is not valid.
            InvalidDabBiasRangeError: The bias range for DAB channel adjustment is not valid.
            InvalidCutoffRangeError: The cutoff range is not valid.
        """

        # Initialize base class.
        #
        super().__init__(keyword='hed_color')

        # Initialize members.
        #
        self.__sigma_ranges = None  # Configured sigma ranges for H, E, and D channels.
        self.__bias_ranges = None   # Configured bias ranges for H, E, and D channels.
        self.__cutoff_range = None  # Cutoff interval.
        self.__sigmas = None        # Randomized sigmas for H, E, and D channels.
        self.__biases = None        # Randomized biases for H, E, and D channels.

        # Save configuration.
        #
        self.__setsigmaranges(haematoxylin_sigma_range=haematoxylin_sigma_range, eosin_sigma_range=eosin_sigma_range, dab_sigma_range=dab_sigma_range)
        self.__setbiasranges(haematoxylin_bias_range=haematoxylin_bias_range, eosin_bias_range=eosin_bias_range, dab_bias_range=dab_bias_range)
        self.__setcutoffrange(cutoff_range=cutoff_range)

    def __setsigmaranges(self, haematoxylin_sigma_range, eosin_sigma_range, dab_sigma_range):
        """
        Set the sigma intervals.

        Args:
            haematoxylin_sigma_range (tuple, None): Adjustment range for the Haematoxylin channel.
            eosin_sigma_range (tuple, None): Adjustment range for the Eosin channel.
            dab_sigma_range (tuple, None): Adjustment range for the DAB channel.

        Raises:
            InvalidHaematoxylinSigmaRangeError: The sigma range for Haematoxylin channel adjustment is not valid.
            InvalidEosinSigmaRangeError: The sigma range for Eosin channel adjustment is not valid.
            InvalidDabSigmaRangeError: The sigma range for DAB channel adjustment is not valid.
        """

        # Check the intervals.
        #
        if haematoxylin_sigma_range is not None:
            if len(haematoxylin_sigma_range) != 2 or haematoxylin_sigma_range[1] < haematoxylin_sigma_range[0] or haematoxylin_sigma_range[0] < -1.0 or 1.0 < haematoxylin_sigma_range[1]:
                raise Exception("InvalidHaematoxylinSigmaRangeError(haematoxylin_sigma_range)")

        if eosin_sigma_range is not None:
            if len(eosin_sigma_range) != 2 or eosin_sigma_range[1] < eosin_sigma_range[0] or eosin_sigma_range[0] < -1.0 or 1.0 < eosin_sigma_range[1]:
                raise Exception("InvalidEosinSigmaRangeError(eosin_sigma_range)")

        if dab_sigma_range is not None:
            if len(dab_sigma_range) != 2 or dab_sigma_range[1] < dab_sigma_range[0] or dab_sigma_range[0] < -1.0 or 1.0 < dab_sigma_range[1]:
                raise Exception("InvalidDabSigmaRangeError(dab_sigma_range)")

        # Store the settings.
        #
        self.__sigma_ranges = [haematoxylin_sigma_range, eosin_sigma_range, dab_sigma_range]

        self.__sigmas = [haematoxylin_sigma_range[0] if haematoxylin_sigma_range is not None else 0.0,
                         eosin_sigma_range[0] if eosin_sigma_range is not None else 0.0,
                         dab_sigma_range[0] if dab_sigma_range is not None else 0.0]

    def __setbiasranges(self, haematoxylin_bias_range, eosin_bias_range, dab_bias_range):
        """
        Set the bias intervals.

        Args:
            haematoxylin_bias_range (tuple, None): Bias range for the Haematoxylin channel.
            eosin_bias_range (tuple, None) Bias range for the Eosin channel.
            dab_bias_range (tuple, None): Bias range for the DAB channel.

        Raises:
            InvalidHaematoxylinBiasRangeError: The bias range for Haematoxylin channel adjustment is not valid.
            InvalidEosinBiasRangeError: The bias range for Eosin channel adjustment is not valid.
            InvalidDabBiasRangeError: The bias range for DAB channel adjustment is not valid.
        """

        # Check the intervals.
        #
        if haematoxylin_bias_range is not None:
            if len(haematoxylin_bias_range) != 2 or haematoxylin_bias_range[1] < haematoxylin_bias_range[0] or haematoxylin_bias_range[0] < -1.0 or 1.0 < haematoxylin_bias_range[1]:
                raise Exception("InvalidHaematoxylinBiasRangeError(haematoxylin_bias_range)")

        if eosin_bias_range is not None:
            if len(eosin_bias_range) != 2 or eosin_bias_range[1] < eosin_bias_range[0] or eosin_bias_range[0] < -1.0 or 1.0 < eosin_bias_range[1]:
                raise Exception("InvalidEosinBiasRangeError(eosin_bias_range)")

        if dab_bias_range is not None:
            if len(dab_bias_range) != 2 or dab_bias_range[1] < dab_bias_range[0] or dab_bias_range[0] < -1.0 or 1.0 < dab_bias_range[1]:
                raise Exception("InvalidDabBiasRangeError(dab_bias_range)")

        # Store the settings.
        #
        self.__bias_ranges = [haematoxylin_bias_range, eosin_bias_range, dab_bias_range]

        self.__biases = [haematoxylin_bias_range[0] if haematoxylin_bias_range is not None else 0.0,
                         eosin_bias_range[0] if eosin_bias_range is not None else 0.0,
                         dab_bias_range[0] if dab_bias_range is not None else 0.0]

    def __setcutoffrange(self, cutoff_range):
        """
        Set the cutoff value. Patches with mean value outside the cutoff interval will not be augmented.

        Args:
            cutoff_range (tuple, None): Patches with mean value outside the cutoff interval will not be augmented.

        Raises:
            InvalidCutoffRangeError: The cutoff range is not valid.
        """

        # Check the interval.
        #
        if cutoff_range is not None:
            if len(cutoff_range) != 2 or cutoff_range[1] < cutoff_range[0] or cutoff_range[0] < 0.0 or 1.0 < cutoff_range[1]:
                raise Exception("InvalidCutoffRangeError(cutoff_range)")

        # Store the setting.
        #
        self.__cutoff_range = cutoff_range if cutoff_range is not None else [0.0, 1.0]

    def transform(self, patch):
        """
        Apply color deformation on the patch.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.
        """

        # Check if the patch is inside the cutoff values.
        #
        patch_mean = np.mean(a=patch) / 255.0
        if self.__cutoff_range[0] <= patch_mean <= self.__cutoff_range[1]:
            # Reorder the patch to channel last format and convert the image patch to HED color coding.
            #
            patch_image = np.transpose(a=patch, axes=(1, 2, 0))
            patch_hed = rgb2hed(rgb=patch_image)

            # Augment the Haematoxylin channel.
            #
            if self.__sigmas[0] != 0.0:
                patch_hed[:, :, 0] *= (1.0 + self.__sigmas[0])

            if self.__biases[0] != 0.0:
                patch_hed[:, :, 0] += self.__biases[0]

            # Augment the Eosin channel.
            #
            if self.__sigmas[1] != 0.0:
                patch_hed[:, :, 1] *= (1.0 + self.__sigmas[1])

            if self.__biases[1] != 0.0:
                patch_hed[:, :, 1] += self.__biases[1]

            # Augment the DAB channel.
            #
            if self.__sigmas[2] != 0.0:
                patch_hed[:, :, 2] *= (1.0 + self.__sigmas[2])

            if self.__biases[2] != 0.0:
                patch_hed[:, :, 2] += self.__biases[2]

            # Convert back to RGB color coding and order back to channels first order.
            #
            patch_rgb = hed2rgb(hed=patch_hed)
            patch_rgb = np.clip(a=patch_rgb, a_min=0.0, a_max=1.0)
            patch_rgb *= 255.0
            patch_rgb = patch_rgb.astype(dtype=np.uint8)

            patch_transformed = np.transpose(a=patch_rgb, axes=(2, 0, 1))

            return patch_transformed

        else:
            # The image patch is outside the cutoff interval.
            #
            return patch



    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize sigma and bias for each channel.
        #
        self.__sigmas = [np.random.uniform(low=sigma_range[0], high=sigma_range[1], size=None) if sigma_range is not None else 1.0 for sigma_range in self.__sigma_ranges]
        self.__biases = [np.random.uniform(low=bias_range[0], high=bias_range[1], size=None) if bias_range is not None else 0.0 for bias_range in self.__bias_ranges]

