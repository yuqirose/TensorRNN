
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from trnn import rnn_with_feed_prev

def LSTM(x, is_training, config):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, config.num_steps, 1)
    # Define a lstm cell with tensorflow
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(config.hidden_size,forget_bias=1.0)


#     lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(config.num_layers)])

    # Get lstm cell output
    feed_prev = not is_training if config.use_error_prop else False

    outputs, state  = rnn_with_feed_prev(cell, x, feed_prev, config)

    # Tanh activation
    prediction = tf.nn.tanh(outputs)
    return prediction

