
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from trnn import *

def LSTM(inputs, is_training, config):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    inputs = tf.unstack(inputs, config.num_steps, 1)
    # Define a lstm cell with tensorflow
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(config.hidden_size,forget_bias=1.0)

    cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(config.num_layers)])

    # Get lstm cell output
    feed_prev = not is_training if config.use_error_prop else False

    outputs, state  = rnn_with_feed_prev(cell, inputs, feed_prev, config)

    # Tanh activation
    prediction = tf.nn.tanh(outputs)
    return prediction

def TRNN(inputs, is_training, config):

    inputs = tf.unstack(inputs, config.num_steps, 1)
    def trnn_cell():
        return EinsumTensorRNNCell(config.hidden_size,config.num_lags, config.rank_vals)
        
    cell = tf.contrib.rnn.MultiRNNCell(
        [trnn_cell() for _ in range(config.num_layers)])

    feed_prev = not is_training if config.use_error_prop else False
    
    initial_states = []
    for lag in range(config.num_lags):
      initial_state =  cell.zero_state(config.batch_size, dtype= tf.float32)
      initial_states.append(initial_state)

    input_size = 3

    outputs, state, weights  = tensor_rnn_with_feed_prev(cell, inputs, config.num_steps, config.hidden_size,
        config.num_lags, initial_states, input_size, feed_prev=feed_prev, burn_in_steps=config.burn_in_steps)


    prediction = tf.nn.tanh(outputs)
    return prediction
