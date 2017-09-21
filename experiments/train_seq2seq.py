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
import numpy 

'''
To forecast time series using a recurrent neural network, we consider every 
row as a sequence of short time series. Because dataset times series has 9 dim, we will then
handle 9 sequences for every sample.
'''


# Training Parameters
learning_rate = 0.01
training_steps = 200
inp_steps = 50
out_steps = 101-inp_steps
num_test_steps = 50 #EOS
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

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_input]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_input]))
}

def RNN(enc_inps, dec_inps,  weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, inp_steps, n_input)
    # Required shape: 'inp_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'inp_steps' tensors of shape (batch_size, n_input)
    enc_inps = tf.unstack(enc_inps, inp_steps, 1)
    dec_inps = tf.unstack(dec_inps, out_steps, 1)


    # Define a lstm cell with tensorflow
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(num_hidden,forget_bias=1.0)


#     lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(num_layers)])

    # Get lstm cell output --use as encoder
    enc_outs, enc_states = rnn.static_rnn(stacked_lstm, enc_inps, dtype=tf.float32)
    
    dec_outs, dec_states = rnn.static_rnn(stacked_lstm, dec_inps, initial_state = enc_states, dtype=tf.float32)
    
    # Concatenate all hidden states with linear
    logits = []
    for output in dec_outs:
        logit = tf.matmul(output, weights['out']) + biases['out']
        logits.append(logit) 
    logits= tf.stack(logits, 1)
    return logits

with tf.variable_scope('LSTM'):
    logits = RNN(X, Y, weights, biases)
    # Using tanh activation function
    prediction = tf.nn.tanh(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.squared_difference(prediction, Z))
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
    test_len = 10
    test_enc_inps = dataset.test.enc_inps[:test_len].reshape((-1, inp_steps, num_input))
    test_dec_inps = dataset.test.dec_inps[:test_len].reshape((-1, out_steps, num_input))
    test_dec_outs = dataset.test.dec_outs[:test_len].reshape((-1, out_steps, num_input))

    
    # Fetch the predictions 
    fetches = {
        "true":Y,
        "pred":prediction,
        "loss":loss_op
    }
    vals = sess.run(fetches, feed_dict={X: test_enc_inps, Y: test_dec_inps, Z: test_dec_outs})
    print("Testing Loss:", vals["loss"])

    numpy.save("./result/predict.npy", (vals["true"], vals["pred"]))
