"""Functions for downloading and reading time series data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import random_seed


def slide_window(a, window):
    """ Extract examples from time series"""
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    examples = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    inp = examples[:-1]
    out = examples[1:]
    return inp, out

        
class DataSet(object):

  def __init__(self,
               data,
               num_steps,
               seed=None):
    """Construct a DataSet.
    Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
   
    inps, outs = slide_window(data, num_steps)

    assert inps.shape[0] == outs.shape[0], (
        'inps.shape: %s outs.shape: %s' % (inps.shape, outs.shape))


    self._num_examples = inps.shape[0]
    self._inps = inps
    self._outs = outs
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def inps(self):
    return self._inps

  @property
  def outs(self):
    return self._outs

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._inps = self.inps[perm0]
      self._outs = self.outs[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      inps_rest_part = self._inps[start:self._num_examples]
      outs_rest_part = self._outs[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._inps = self.inps[perm]
        self._outs = self.outs[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      inps_new_part = self._inps[start:end]
      outs_new_part = self._outs[start:end]
      return np.concatenate((inps_rest_part, inps_new_part), axis=0) , np.concatenate((outs_rest_part, outs_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._inps[start:end], self._outs[start:end]

def read_data_sets(data_path, 
                  n_steps = 10,
                  n_test_steps = 10,
                  val_size = 0.1, 
                  test_size = 0.1, 
                  seed=None):
    print("loading time series ...")
    data = np.load(data_path)
    # Expand the dimension if univariate time series
    if (np.ndim(data)==1):
        data = np.expand_dims(data, axis=1)
    print("input type ",type( data), np.shape(data))

    # """normalize the data"""
    # print("normalize to (0-1)")
    # data = normalize_columns(data)
    ntest = int(round(len(data) * (1.0 - test_size)))
    nval = int(round(len(data[:ntest]) * (1.0 - val_size)))

    train_data, valid_data, test_data = data[:nval, ], data[nval:ntest, ], data[ntest:,]

    train_options = dict(num_steps=n_steps, seed=seed)
    test_options = dict(num_steps=n_test_steps, seed=seed)
    train = DataSet(train_data, **train_options)
    valid = DataSet(valid_data, **train_options)
    test = DataSet(test_data, **test_options)
  
    return base.Datasets(train=train, validation=valid, test=test)

