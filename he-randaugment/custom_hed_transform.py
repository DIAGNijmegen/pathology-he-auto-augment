
import numpy as np
from scipy import linalg
from skimage.util import dtype, dtype_limits
from skimage.exposure import rescale_intensity
import time

rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]]).astype('float32')
hed_from_rgb = linalg.inv(rgb_from_hed).astype('float32')


def rgb2hed(rgb):

    return separate_stains(rgb, hed_from_rgb)

def hed2rgb(hed):

    return combine_stains(hed, rgb_from_hed)

def separate_stains(rgb, conv_matrix):

    # # t = time.time()
    # rgb = dtype.img_as_float(rgb, force_copy=True).astype('float32')
    # # print('{f} took {s} s'.format(f='separate img_as_float', s=(time.time() - t)), flush=True)
    # # print('rgb type is {r}, matrix type is {m}'.format(r=rgb.dtype, m=conv_matrix.dtype), flush=True)
    #
    # # t = time.time()
    # rgb = -np.log(rgb)
    # # print('{f} took {s} s'.format(f='separate np.log', s=(time.time() - t)), flush=True)
    #
    # # t = time.time()
    # rgb += 2
    # rgb = np.reshape(rgb, (-1, 3))
    # # print('{f} took {s} s'.format(f='separate add reshape', s=(time.time() - t)), flush=True)
    #
    # # print('x shape is {s}, conv_matrix shape is {c}'.format(s=x.shape, c=conv_matrix.shape))
    #
    # # t = time.time()
    # stains = np.dot(rgb, conv_matrix)
    # # print('{f} took {s} s'.format(f='separate np.dot', s=(time.time() - t)), flush=True)
    #
    # return np.reshape(stains, rgb.shape)

    rgb = dtype.img_as_float(rgb, force_copy=True).astype('float32')
    rgb += 2
    stains = np.dot(np.reshape(-np.log(rgb), (-1, 3)), conv_matrix)
    return np.reshape(stains, rgb.shape)


def combine_stains(stains, conv_matrix):

    # # t = time.time()
    # stains = dtype.img_as_float(stains).astype('float32')
    # # stains = stains.astype('float32')
    # # conv_matrix = conv_matrix.astype('float32')
    # # print('{f} took {s} s'.format(f='separate img_as_float', s=(time.time() - t)), flush=True)
    # # print('stains type is {r}, matrix type is {m}'.format(r=stains.dtype, m=conv_matrix.dtype), flush=True)
    #
    # stains = -np.reshape(stains, (-1, 3))
    # # print('x shape is {s}, conv_matrix shape is {c}'.format(s=x.shape, c=conv_matrix.shape))
    #
    # # t = time.time()
    # logrgb2 = np.dot(stains, conv_matrix)
    # # print('{f} took {s} s'.format(f='combine np.dot', s=(time.time() - t)), flush=True)
    #
    # # t = time.time()
    # rgb2 = np.exp(logrgb2)
    # # print('{f} took {s} s'.format(f='combine np.exp', s=(time.time() - t)), flush=True)
    #
    # t = time.time()
    # x = rescale_intensity(np.reshape(rgb2 - 2, stains.shape),
    #                          in_range=(-1, 1))
    # print('{f} took {s} s'.format(f='combine rescale_intensity', s=(time.time() - t)), flush=True)
    #
    # return x

    stains = dtype.img_as_float(stains.astype('float64')).astype('float32')  # stains are out of range [-1, 1] so dtype.img_as_float complains if not float64
    logrgb2 = np.dot(-np.reshape(stains, (-1, 3)), conv_matrix)
    rgb2 = np.exp(logrgb2)
    return rescale_intensity(np.reshape(rgb2 - 2, stains.shape),
                             in_range=(-1, 1))