import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cPickle as pickle
def seq_raw_data(data_path="../../../chaotic_ts_mat.pkl", val_size = 0.1, test_size = 0.1):
    print("loading sequence data ...")
    data = pickle.load(open(data_path))
    if (np.ndim(data)==1):
        data = np.expand_dims(data, axis=1)
    print("input type ",type( data), np.shape(data))

    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data[:ntest]) * (1 - val_size)))
    train_data, valid_data, test_data = data[:nval,:], data[nval:ntest,:], data[ntest:,:]
    return train_data, valid_data, test_data


def ptb_producer_rnd(raw_data, is_training, batch_size, num_steps, horizon, name):
    """The mini-batch generator: randomly select across different sources
    Args:
        num_steps: length of back-propogation in time
        horizon: forecast horizon"""
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        (data_len,data_dim,num_sources) = np.shape(raw_data)
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.float32)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len,:],
                          [batch_size, batch_len, data_dim, -1])
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
              epoch_size = tf.identity(epoch_size, name="epoch_size")

        if is_training:
            i = tf.train.range_input_producer(batch_len-num_steps-horizon, shuffle=True).dequeue()
            j = tf.train.range_input_producer(i%num_sources, shuffle=False).dequeue()
            x = tf.squeeze(tf.slice(data, [0, i , 0 ,j], [batch_size, num_steps, data_dim, 1]),[3])
            y = tf.squeeze(tf.slice(data, [0, i + horizon, 0, j], [batch_size, num_steps, data_dim , 1]),[3])
        else: 
            i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
            j = tf.train.range_input_producer(i%num_sources, shuffle=False).dequeue()
            
            x = tf.squeeze(tf.slice(data, [0, i*num_steps, 0, j], [batch_size, num_steps, data_dim, 1]), [3])
            y = tf.squeeze(tf.slice(data, [0, i*num_steps + horizon, 0, j], [batch_size, num_steps, data_dim, 1]), [3])

        return x, y


def ptb_producer(raw_data, is_training, batch_size, num_steps, horizon, name):
    """The mini-batch generator
    Args:
        num_steps: length of back-propogation in time
        horizon: forecast horizon"""
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        (data_len,data_dim) = np.shape(raw_data)
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.float32)
      
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len,:],
                          [batch_size, batch_len, -1])
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
              epoch_size = tf.identity(epoch_size, name="epoch_size")

        if is_training:
            i = tf.train.range_input_producer(batch_len-num_steps-horizon, shuffle=True).dequeue()
            x = tf.slice(data, [0, i , 0], [batch_size, num_steps, data_dim])
            y = tf.slice(data, [0, i + horizon, 0], [batch_size, num_steps, data_dim])
        else: 
            i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
            x = tf.slice(data, [0, i*num_steps, 0], [batch_size, num_steps, data_dim])
            y = tf.slice(data, [0, i*num_steps + horizon, 0], [batch_size, num_steps, data_dim])

        return x, y

class PTBInput(object):
    """The input data."""

    def __init__(self, is_training, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.horizon = horizon = config.horizon
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_size = np.shape(data)[1]
        if np.ndim(data)==2:
            self.input_data, self.targets = ptb_producer(
                data, is_training, batch_size, num_steps, horizon, name=name)
        else:
            self.input_data, self.targets = ptb_producer_rnd(
                data, is_training, batch_size, num_steps, horizon, name=name)

