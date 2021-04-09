"""
Keras data generators for supervised classification tasks.
"""
from data_augmentation import rgb_to_gray
import dl
from os.path import join, exists, basename, dirname, splitext
import numpy as np
import argparse
import os
import gc
import shutil
from glob import glob
import time
import randaugment as ra
#import tensorflow.compat.v1 as tf

from keras.utils import Sequence

#----------------------------------------------------------------------------------------------------

class SupervisedGenerator(object):
    """
    Class to randomly provide batches of supervised images loaded from numpy arrays on disk.
    """

    def __init__(self, x_path, y_path, batch_size, augmenter, one_hot=True, compare_augmentation=False, color_space='rgb',randaugment=True,rand_m=5, rand_n=1,ra_type=None):
        """
        Class to randomly provide batches of images loaded from numpy arrays on disk.

        Args:
            x_path (str): path to array of images in numpy uint8 format with channels in last dimension.
            y_path (str): path to array of labels.
            batch_size (int): number of samples per batch.
            augment (bool): True to apply rotation and flipping augmentation.
        """

        # Params
        self.batch_size = batch_size
        self.x_path = x_path
        self.y_path = y_path
        self.augmenter = augmenter
        self.one_hot = one_hot
        self.compare_augmentation = compare_augmentation
        self.color_space = color_space
        self.randaugment = randaugment
        self.rand_m = rand_m
        self.rand_n = rand_n
        self.ra_type = ra_type

        # Read data
        self.x = np.load(x_path)  # need to read here due to keras multiprocessing
        #print('read x')
        # self.x = None
        self.y = np.load(y_path)
        #print('read y')
        # Drop 255 class
        self.idx = np.where(self.y != 255)[0]
        #print('got idxx')
        self.classes = np.unique(self.y[self.idx])
        #print('got  clases')
        self.n_classes = len(self.classes)
        #print('got num clases')
        self.n_samples = len(self.idx)
        #print('got num samples')
        self.n_batches = self.n_samples // self.batch_size
        #print('got num batches')

        # Indexes for classes
        self.class_idx = []
        for i in self.classes:
            self.class_idx.append(
                np.where(self.y == i)[0]
            )

    def get_n_classes(self):
        return self.n_classes

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        #print('In next')
        return self.next()

    def __len__(self):
        """
        Provide length in number of batches
        Returns (int): number of batches available in the entire dataset.
        """
        #print('In len')
        #print('self.n_batches',self.n_batches)
        
        return self.n_batches
    def randaugment_batch(self, x):
        """
        Applies randaugmentation.

        Args:
            x: batch of images with shape [batch, x, y, c].

        Returns: batch of augmented images.

        """
        if self.randaugment:
            
            x_augmented = np.zeros_like(x)
            batch_list=[]
            for i in range(x.shape[0]):
                
                x_augmented[i, ...] =  ra.distort_image_with_randaugment(x[i, ...], self.rand_n, self.rand_m,self.ra_type)
       
             
    
        else:
            x_augmented = x

        if self.compare_augmentation:
            x_augmented = np.stack([x, x_augmented])
        return x_augmented
        
        
    def augment_batch(self, x):
        """
        Applies augmentation.

        Args:
            x: batch of images with shape [batch, x, y, c].

        Returns: batch of augmented images.

        """

        if self.augmenter is not None:
            x_augmented = np.zeros_like(x)
            for i in range(x.shape[0]):
                x_augmented[i, ...] = self.augmenter.augment(x[i, ...])
        else:
            x_augmented = x

        if self.compare_augmentation:
            x_augmented = np.dstack([x, x_augmented])

        return x_augmented

    def get_batch(self):
        """
        Draws a random set of samples from the training set and assembles pairs of images and labels.

        Returns: batch of images with shape [batch, x, y, c].
        """
        #print('In get batch')
        # Get samples
        idxs = []
        for i in range(self.n_classes):
            idxs.append(
                np.random.choice(self.class_idx[i], self.batch_size // self.n_classes, replace=True)
            )

        # Merge
        idxs = np.concatenate(idxs)

        # Randomize
        np.random.shuffle(idxs)

        # Build batch
        if self.x is None:
            
            self.x = np.load(self.x_path)
        x = self.x[idxs, ...]
        y = self.y[idxs]

        # Color space
        if self.color_space == 'grayscale':
            x = rgb_to_gray(x)
        #print('In get batch: got the ids')
        # Augment
        if self.augmenter is not None:
            # t = time.time()
            x = self.augment_batch(x)
            # print('{f} took {s} s'.format(f='augment_batch', s=(time.time() - t)), flush=True)
        # RandAugment
        if self.randaugment:
            #print('About to rand')
            # t = time.time()
            x = self.randaugment_batch(x)
            # print('{f} took {s} s'.format(f='augment_batch', s=(time.time() - t)), flush=True)

        # Range (float [-1, +1])
        x = (x / 255.0 * 2) - 1

        # One-hot encoding
        if self.one_hot:
            y = np.eye(self.n_classes)[y]
        #if self.randaugment:
        #y=tf.constant(y, dtype=tf.float32)

        #return 
        return x, y

    def next(self):
        # t = time.time()
        #print('In next')
        batch = self.get_batch()
        # print('{f} took {s} s'.format(f='next', s=(time.time() - t)), flush=True)
        batch = self.transform(batch)
        return batch

    def transform(self, batch):
        """
        Implement this function to alter the returned batch in any way. Useful if you inherit this class to transform
        the batch data.

        Args:
            batch: batch of images with shape [batch, x, y, c].

        Returns: batch of images with shape [batch, x, y, c].
        """
        return batch

#----------------------------------------------------------------------------------------------------

class SupervisedSequence(dl.utils.Sequence):
    """
    Class to sequentially provide batches of supervised images loaded from numpy arrays on disk.
    """

    def __init__(self, x_path, y_path, batch_size, one_hot=True, color_space='rgb', include_255=False, augmenter=None, compare_augmentation=False):
        """
        Class to sequentially provide batches of supervised images loaded from numpy arrays on disk.

        Args:
            x_path (str): path to array of images in numpy uint8 format with channels in last dimension.
            y_path (str): path to array of labels.
            batch_size (int): number of samples per batch.
        """

        # Params
        print('Entered SupervisedSequence')
        self.batch_size = batch_size
        self.x_path = x_path
        self.y_path = y_path
        self.one_hot = one_hot
        self.color_space = color_space
        self.include_255 = include_255
        self.augmenter = augmenter
        self.compare_augmentation = compare_augmentation

        # Read data
        
        
        self.x =  np.concatenate((np.load(x_path[0]) , np.load(x_path[1]) , np.load(x_path[2]) ))# need to read here due to keras multiprocessing
        
        
        self.y = np.concatenate((np.load(y_path[0]) , np.load(y_path[1]) , np.load(y_path[2]) ))# need to read here due to keras multiprocessing
   
        print('Concatenated everything')
        # Drop 255 class
        if self.include_255:
            self.idx = np.arange(0, self.y.shape[0])
        else:
            self.idx = np.where(self.y != 255)[0]

        self.n_samples = len(self.idx)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.classes = np.unique(self.y[self.idx])
        self.n_classes = len(self.classes)

    def augment_batch(self, x):
        """
        Applies augmentation.

        Args:
            x: batch of images with shape [batch, x, y, c].

        Returns: batch of augmented images.

        """

        if self.augmenter is not None:
            x_augmented = np.zeros_like(x)
            for i in range(x.shape[0]):
                x_augmented[i, ...] = self.augmenter.augment(x[i, ...])
        else:
            x_augmented = x

        if self.compare_augmentation:
            x_augmented = np.dstack([x, x_augmented])

        return x_augmented

    def get_n_classes(self):
        return self.n_classes

    def __len__(self):
        """
        Provide length in number of batches
        Returns (int): number of batches available in the entire dataset.
        """
        return self.n_batches

    def get_batch(self, idx):
        """
        Draws a set of samples from the dataset based on the index and assembles pairs of images and labels. Index refers
        to batches (not samples).

        Returns: batch of images with shape [batch, x, y, c].
        """

        # Get samples
        idx_batch = idx * self.batch_size
        if idx_batch + self.batch_size >= self.n_samples:
            idxs = np.arange(idx_batch, self.n_samples)
        else:
            idxs = np.arange(idx_batch, idx_batch + self.batch_size)

        # Build batch
        if self.x is None:
            self.x = np.load(self.x_path)
        x = self.x[self.idx[idxs], ...]
        y = self.y[self.idx[idxs]]

        # Color space
        if self.color_space == 'grayscale':
            x = rgb_to_gray(x)

        # Augment
        if self.augmenter is not None:
            x = self.augment_batch(x)

        # Format
        x = (x / 255.0 * 2) - 1

        # One-hot encoding
        if self.one_hot:
            y = np.eye(self.n_classes)[y]
    
        return x, y

    def __getitem__(self, idx):
        batch = self.get_batch(idx)
        batch = self.transform(batch)
        return batch

    def transform(self, batch):
        """
        Implement this function to alter the returned batch in any way. Useful if you inherit this class to transform
        the batch data.

        Args:
            batch: batch of images with shape [batch, x, y, c].

        Returns: batch of images with shape [batch, x, y, c].
        """
        return batch

    def get_all_labels(self, one_hot=True):

        y = self.y[self.idx]
        if one_hot:
            y = np.eye(self.n_classes)[y]

        return y
class SupervisedSequenceSingle(dl.utils.Sequence):
    """
    Class to sequentially provide batches of supervised images loaded from numpy arrays on disk.
    """

    def __init__(self, x_path, y_path, batch_size, one_hot=True, color_space='rgb', include_255=False, augmenter=None, compare_augmentation=False):
        """
        Class to sequentially provide batches of supervised images loaded from numpy arrays on disk.

        Args:
            x_path (str): path to array of images in numpy uint8 format with channels in last dimension.
            y_path (str): path to array of labels.
            batch_size (int): number of samples per batch.
        """

        # Params
        self.batch_size = batch_size
        self.x_path = x_path
        self.y_path = y_path
        self.one_hot = one_hot
        self.color_space = color_space
        self.include_255 = include_255
        self.augmenter = augmenter
        self.compare_augmentation = compare_augmentation

        # Read data
        self.x = np.load(x_path)  # need to read here due to keras multiprocessing
        self.y = np.load(y_path)

        # Drop 255 class
        if self.include_255:
            self.idx = np.arange(0, self.y.shape[0])
        else:
            self.idx = np.where(self.y != 255)[0]

        self.n_samples = len(self.idx)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.classes = np.unique(self.y[self.idx])
        self.n_classes = len(self.classes)

    def augment_batch(self, x):
        """
        Applies augmentation.

        Args:
            x: batch of images with shape [batch, x, y, c].

        Returns: batch of augmented images.

        """

        if self.augmenter is not None:
            x_augmented = np.zeros_like(x)
            for i in range(x.shape[0]):
                x_augmented[i, ...] = self.augmenter.augment(x[i, ...])
        else:
            x_augmented = x

        if self.compare_augmentation:
            x_augmented = np.dstack([x, x_augmented])

        return x_augmented

    def get_n_classes(self):
        return self.n_classes

    def __len__(self):
        """
        Provide length in number of batches
        Returns (int): number of batches available in the entire dataset.
        """
        return self.n_batches

    def get_batch(self, idx):
        """
        Draws a set of samples from the dataset based on the index and assembles pairs of images and labels. Index refers
        to batches (not samples).

        Returns: batch of images with shape [batch, x, y, c].
        """

        # Get samples
        idx_batch = idx * self.batch_size
        if idx_batch + self.batch_size >= self.n_samples:
            idxs = np.arange(idx_batch, self.n_samples)
        else:
            idxs = np.arange(idx_batch, idx_batch + self.batch_size)

        # Build batch
        if self.x is None:
            self.x = np.load(self.x_path)
        x = self.x[self.idx[idxs], ...]
        y = self.y[self.idx[idxs]]

        # Color space
        if self.color_space == 'grayscale':
            x = rgb_to_gray(x)

        # Augment
        if self.augmenter is not None:
            x = self.augment_batch(x)

        # Format
        x = (x / 255.0 * 2) - 1

        # One-hot encoding
        if self.one_hot:
            y = np.eye(self.n_classes)[y]

        return x, y

    def __getitem__(self, idx):
        batch = self.get_batch(idx)
        batch = self.transform(batch)
        return batch

    def transform(self, batch):
        """
        Implement this function to alter the returned batch in any way. Useful if you inherit this class to transform
        the batch data.

        Args:
            batch: batch of images with shape [batch, x, y, c].

        Returns: batch of images with shape [batch, x, y, c].
        """
        return batch

    def get_all_labels(self, one_hot=True):

        y = self.y[self.idx]
        if one_hot:
            y = np.eye(self.n_classes)[y]

        return y

#----------------------------------------------------------------------------------------------------

