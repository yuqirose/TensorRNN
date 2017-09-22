# %load reader.py
"""Functions for downloading and reading time series data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import random_seed

import tensorflow as tf
from tensorflow.contrib import rnn
from reader import read_data_sets
from trnn import rnn_with_feed_prev,EinsumTensorRNNCell,tensor_rnn_with_feed_prev
import numpy 
from train_config import *

'''
To forecast time series using a recurrent neural network, we consider every 
row as a sequence of short time series. Because dataset times series has 9 dim, we will then
handle 9 sequences for every sample.
'''

# Training Parameters
config = TrainConfig()
# Training Parameters
learning_rate = 0.01
training_steps = 1000
inp_steps = 50
out_steps = 101-inp_steps
num_test_steps = inp_steps #EOS
batch_size = 20
display_step = 200

dataset, stats = read_data_sets("./lorenz.npy", inp_steps, num_test_steps)

# Network Parameters
num_input = stats['num_input']  # dataset data input (time series dimension: 3)
num_hidden = 16 # hidden layer num of features
num_layers = 2 # number of layers

# tf Graph input
X = tf.placeholder("float", [None, inp_steps, num_input])
Y = tf.placeholder("float", [None, out_steps, num_input])

# Decoder output
Z = tf.placeholder("float", [None, out_steps, num_input])


def Model(enc_inps, dec_inps, is_training):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, inp_steps, n_input)
    # Required shape: 'inp_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'inp_steps' tensors of shape (batch_size, n_input)
    # enc_inps = tf.unstack(enc_inps, inp_steps, 1)

    # def lstm_cell():
    #     return tf.contrib.rnn.BasicLSTMCell(config.hidden_size,forget_bias=1.0)

    # cell = tf.contrib.rnn.MultiRNNCell(
    #     [lstm_cell() for _ in range(config.num_layers)])
    def trnn_cell():
        return EinsumTensorRNNCell(config.hidden_size, config.num_lags, config.rank_vals)
        
    cell = tf.contrib.rnn.MultiRNNCell(
        [trnn_cell() for _ in range(config.num_layers)])

    with tf.variable_scope("Encoder", reuse=None):
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell, enc_inps, True, config)

    with tf.variable_scope("Decoder", reuse=None):
        dec_outs, dec_state =  tensor_rnn_with_feed_prev(cell, dec_inps, is_training, config, enc_states)

    # dec_outs, dec_states  = rnn_with_feed_prev(stacked_lstm, dec_inps, is_training, config, enc_states)
    
    return dec_outs

with tf.name_scope("Train"):
    with tf.variable_scope("Model", reuse=None):
        train_pred = Model(X, Y, True)
with tf.name_scope("Test"):
    with tf.variable_scope("Model", reuse=True):
        test_pred = Model(X, Y, False)


# Define loss and optimizer
loss_op = tf.sqrt(tf.reduce_mean(tf.squared_difference(train_pred, Z)))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y, batch_z = dataset.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, Z:batch_z})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss 
            loss = sess.run(loss_op, feed_dict={X: batch_x,Y: batch_y, Z:batch_z})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) )

    print("Optimization Finished!")

    # Calculate accuracy for 128 dataset test inps
    test_len = 100
    test_enc_inps = dataset.test.enc_inps[:test_len].reshape((-1, inp_steps, num_input))
    test_dec_inps = dataset.test.dec_inps[:test_len].reshape((-1, out_steps, num_input))
    test_dec_outs = dataset.test.dec_outs[:test_len].reshape((-1, out_steps, num_input))

    
    # Fetch the predictions 
    fetches = {
        "true":Z,
        "pred":test_pred,
        "loss":loss_op
    }
    vals = sess.run(fetches, feed_dict={X: test_enc_inps, Y: test_dec_inps, Z: test_dec_outs})
    print("Testing Loss:", vals["loss"])

    numpy.save("./result/trnn_seq2seq.npy", (vals["true"], vals["pred"]))
