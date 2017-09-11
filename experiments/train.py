# %load train.py
""" Recurrent Neural Network Time Series.

A Recurrent Neural Network (LSTM) multivariate time series test_preding implementation 
Minimalist example using TensorFlow library.

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [dataset Dataset](http://yann.lecun.com/exdb/dataset/).
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import dataset data
from reader import read_data_sets
from model import RNN, MRNN, TRNN, MTRNN, LSTM, PLSTM
from train_config import *

'''
To test_pred time series using a recurrent neural network, we consider every 
row as a sequence of short time series. Because dataset times series has 9 dim, we will then
handle 9 sequences for every sample.
'''


# Command line arguments 
flags = tf.flags

flags.DEFINE_string("data_path", "./data.npy",
          "Where the training/test data is stored.")
flags.DEFINE_string("save_path", "/Users/roseyu/Documents/Python/log/",
          "Model output directory.")
flags.DEFINE_bool("use_error_prop", True,
                  "Feed previous output as input in RNN")

flags.DEFINE_integer("hidden_size", 16, "hidden layer size")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_integer("num_steps",10,"Training sequence length")
flags.DEFINE_integer("num_test_steps", 10,"Testing sequence length")

FLAGS = flags.FLAGS


# Training Parameters
config = TrainConfig()
# Update config with cmd args
config.hidden_size = FLAGS.hidden_size
config.learning_rate = FLAGS.learning_rate
config.num_steps = FLAGS.num_steps
config.num_test_steps = FLAGS.num_test_steps

training_steps = 1000
display_step = 200
num_steps = config.num_steps


# Construct dataset
dataset, stats = read_data_sets("./data.npy", num_steps)

# Network Parameters
num_input = stats['num_input'] # dataset data input (time series dimension: 3)


# tf Graph input
X = tf.placeholder("float", [None, num_steps, num_input])
Y = tf.placeholder("float", [None, num_steps, num_input])

with tf.name_scope("Train"):
    with tf.variable_scope("Model", reuse=None):
        train_pred = RNN(X, True, config)

with tf.name_scope("Test"):
    with tf.variable_scope("Model", reuse=True):
        test_pred = RNN(X, False, config)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.squared_difference(train_pred, Y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = dataset.train.next_batch(config.batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss = sess.run(loss_op, feed_dict={X: batch_x,Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) )

    print("Optimization Finished!")


    # Calculate test with error propogation
    num_test_steps = 20
    preds = []
    if config.use_error_prop:
        test_data = dataset.test.inps[:num_steps].reshape((-1, num_steps, num_input))
        test_label = dataset.test.outs[:num_steps]
        pred = sess.run(test_pred, feed_dict={X: test_data, Y: test_label})
        preds.append(pred)
        for i in range(num_test_steps//num_steps):
            test_data = pred
            test_label = dataset.test.outs[(i+1)*num_steps:(i+2)*num_steps]
            pred = sess.run(test_pred, feed_dict={X: test_data, Y: test_label})
            preds.append(pred)

    # Save the variables to disk.
    save_path = saver.save(sess, FLAGS.save_path)
    print("Model saved in file: %s" % save_path)

   
    # Calculate accuracy for test inps
    test_data = dataset.test.inps.reshape((-1, num_steps, num_input))
    test_label = dataset.test.outs
    # Fetch the test_preds 
    fetches = {
        "true":Y,
        "pred":test_pred,
        "loss":loss_op
    }
    vals = sess.run(fetches, feed_dict={X: test_data, Y: test_label})
    print("Testing Loss:", vals["loss"])
