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
from reader_mnist import read_data_sets
from model_seq2seq import *
from trnn import *
import numpy 
from train_config import *

flags = tf.flags
flags.DEFINE_string("model", "TLSTM",
          "Model used for learning.")
flags.DEFINE_string("data_path", "../datasets/mnist.h5",
          "Data input directory.")
flags.DEFINE_string("save_path", "./log/lstm/",
          "Model output directory.")
flags.DEFINE_bool("use_error_prop", True,
                  "Feed previous output as input in RNN")
flags.DEFINE_bool("use_sched_samp", False,
                  "Use scheduled sampling in training")
flags.DEFINE_integer("burn_in_steps", 10, "burn in steps")
flags.DEFINE_integer("test_steps", 10, "test steps size")
flags.DEFINE_integer("hidden_size", 8, "hidden layer size")
flags.DEFINE_integer("batch_size", 20, "batch size")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("decay_rate", 0.8, "learning rate")
flags.DEFINE_integer("rank", 2, "rank for tt decomposition")

FLAGS = flags.FLAGS

'''
To forecast time series using a recurrent neural network, we consider every 
row as a sequence of short time series. Because dataset times series has 9 dim, we will then
handle 9 sequences for every sample.
'''

# Training Parameters
config = TrainConfig()
config.use_error_prop = FLAGS.use_error_prop
config.burn_in_steps = FLAGS.burn_in_steps
config.hidden_size = FLAGS.hidden_size
config.batch_size = FLAGS.batch_size
config.learning_rate = FLAGS.learning_rate
config.decay_rate = FLAGS.decay_rate
config.rank_vals = [FLAGS.rank]

# Scheduled sampling
# = tf.Variable(0.0, trainable=False)
if FLAGS.use_sched_samp:
    config.sample_prob = tf.get_variable("sample_prob", shape=(), initializer=tf.zeros_initializer())
sampling_burn_in = 400

# Training Parameters
training_epochs = config.training_epochs
batch_size = config.batch_size
display_step = 200
inp_steps = config.burn_in_steps
test_steps = FLAGS.test_steps


# Read MINIST Dataset
print('inp steps', inp_steps, 'out steps', test_steps)
dataset, stats = read_data_sets(FLAGS.data_path, inp_steps, test_steps)

# Network Parameters
num_input = stats['num_input']  # dataset data input (time series dimension: 3)
out_steps = test_steps
image_size = int(np.sqrt(num_input))

# tf Graph input
X = tf.placeholder("float", [None, inp_steps, num_input], name="enc_inps")
Y = tf.placeholder("float", [None, out_steps, num_input], name="dec_inps")
# Decoder output
Z = tf.placeholder("float", [None, out_steps, num_input], name="dec_outs")

Model = globals()[FLAGS.model]
with tf.name_scope("Train"):
    with tf.variable_scope("Model", reuse=None):
        train_pred = Model(X, Y, True,  config)
with tf.name_scope("Valid"):
    with tf.variable_scope("Model", reuse=True):
        valid_pred = Model(X, Y, False,  config)
with tf.name_scope("Test"):
    with tf.variable_scope("Model", reuse=True):
        test_pred = Model(X, Y, False,  config)


# Define loss and optimizer
train_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(train_pred, Z)))
valid_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(valid_pred, Z)))
test_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(test_pred, Z)))
# Exponential learning rate decay 
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = config.learning_rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           2000, config.decay_rate, staircase=True)
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_op = optimizer.minimize(train_loss,global_step=global_step)

# Scheduled sampling params
eps_min = 0.1 # minimal prob

# Write summary for losses
train_summary = tf.summary.scalar('train_loss', train_loss)
valid_summary = tf.summary.scalar('valid_loss', valid_loss)
test_summary = tf.summary.scalar('test_loss', test_loss)
lr_summary = tf.summary.scalar('learning_rate', learning_rate)

# Plot summary for predictions
train_true_summary = tf.summary.image("train_true", tf.reshape(Z[0], [out_steps, image_size, image_size, 1]), out_steps)
train_pred_summary = tf.summary.image("train_pred", tf.reshape(train_pred[0],[out_steps, image_size, image_size, 1]), out_steps)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

hist_loss =[]
# Start training
sess_config=tf.ConfigProto()
sess_config.gpu_options.allow_growth=True

with tf.Session(config=sess_config) as sess:
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.save_path,sess.graph)

    # Run the initializer
    sess.run(init)    
    step = 0
    epoch =0
    while(epoch < training_epochs):
        batch_x, batch_y, batch_z = dataset.train.next_batch(batch_size)
        dataset.train.display_data(batch_z)

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, Z:batch_z})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss 
            summary, loss = sess.run([merged,train_loss], feed_dict={X: batch_x,Y: batch_y, Z:batch_z})
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary_writer.add_run_metadata(run_metadata,'step%03d' % step)
            summary_writer.add_summary(summary, step)
            print("Epoch " + str(epoch) + ", Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) )
            
            # Calculate validation
            valid_batch_x, valid_batch_y, valid_batch_z = dataset.validation.next_batch(batch_size)
            va_sum, va_loss = sess.run([valid_summary,valid_loss], \
                                       feed_dict={X: valid_batch_x, Y: valid_batch_y, Z: valid_batch_z})
            summary_writer.add_summary(va_sum, step) 
            print("Validation Loss:", va_loss)
            
            # Overfitting
            hist_loss.append(va_loss)
            if len(hist_loss)>20 and va_loss > np.mean(hist_loss):
                print("Early stopping: step ", step)
                break
          
            #Update sampling prob
            if FLAGS.use_sched_samp and step > sampling_burn_in:
                sample_prob = max(eps_min, 1.0-step/(2*training_steps))
                sess.run(tf.assign(config.sample_prob, sample_prob))
                print('Sampling prob:', sample_prob)
     
        epoch = dataset.train.epochs_completed
        step = step + 1 
    print("Optimization Finished!")

    # Calculate accuracy for test datasets
    test_vals_loss = 1000
    step =0
    while dataset.test.epochs_completed<1:
        test_x, test_y, test_z = dataset.test.next_batch(batch_size, shuffle=False)
    
        # Fetch the predictions 
        fetches = {
            "true":Z,
            "pred":test_pred,
            "loss":test_loss
        }
        test_vals_batch = sess.run(fetches, feed_dict={X: test_x, Y: test_y, Z: test_z})
        if test_vals_batch["loss"]<test_vals_loss:
            test_vals = test_vals_batch
            test_vals_loss = test_vals_batch["loss"]
            print("Batch "+str(step)+ ",Testing Loss:", test_vals_batch["loss"])
        step += 1
    print("Test Loss:",test_vals_loss)
    # Save the variables to disk.
    save_path = saver.save(sess, FLAGS.save_path)
    print("Model saved in file: %s" % save_path)
    # Save predictions 
    numpy.save(save_path+"predict.npy", (test_vals["true"], test_vals["pred"]))
    # Save config file
    with open(save_path+"config.out", 'w') as f:
        f.write('hidden_size:'+ str(config.hidden_size)+'\t'+ 'learning_rate:'+ str(config.learning_rate)+ '\n')
        f.write('train_error:'+ str(loss) +'\t'+ 'valid_error:' + str(va_loss) + '\t'+ 'test_error:'+ str(test_vals["loss"]) +'\n')
