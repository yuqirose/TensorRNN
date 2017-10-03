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
import numpy

# Import dataset data
from reader import read_data_sets
from model import *
from train_config import *

'''
To test_pred time series using a recurrent neural network, we consider every 
row as a sequence of short time series. Because dataset times series has 9 dim, we will then
handle 9 sequences for every sample.
'''


# Command line arguments 
flags = tf.flags
flags.DEFINE_string("data_path", "../data/holstm/",
          "Data input directory.")
flags.DEFINE_string("model", "LSTM", "Model used for learning.")
flags.DEFINE_string("save_path", "./log/holstm/",
          "Model output directory.")

flags.DEFINE_bool("use_error_prop", True,
                  "Feed previous output as input in RNN")
flags.DEFINE_integer("hidden_size", 16, "hidden layer size")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_integer("num_steps",20,"Training sequence length")


FLAGS = flags.FLAGS


# Training Parameters
config = TrainConfig()
# Update config with cmd args
config.use_error_prop = FLAGS.use_error_prop
config.hidden_size = FLAGS.hidden_size
config.learning_rate = FLAGS.learning_rate
config.num_steps = FLAGS.num_steps

training_steps = config.training_steps
display_step = 200
num_steps = config.num_steps
num_test_steps = config.num_test_steps
batch_size = config.batch_size

# Construct dataset
dataset, stats = read_data_sets("./data.npy", num_steps, num_steps)

# Network Parameters
num_input = stats['num_input'] # dataset data input (time series dimension: 3)

# Print exp settings
print('='*80) 

print('|model|', FLAGS.model, '|batch size|', batch_size, '|learn rate|', FLAGS.learning_rate)

print('_'*80)
print('|train steps|', num_steps, '|test steps|', num_test_steps, '|error prop|', config.use_error_prop )

print('='*80)

Model = globals()[FLAGS.model]
# Construct train model
with tf.name_scope("Train"):
    # tf Graph train input
    X = tf.placeholder("float", [None, num_steps, num_input])
    Y = tf.placeholder("float", [None, num_steps, num_input])
    with tf.variable_scope("Model", reuse=None):
        train_pred = Model(X, True, config)
        # Define train loss 
        train_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(train_pred, Y)))
    tf.summary.scalar('loss', train_loss)
    
# Construct test model
with tf.name_scope("Test"):
    # tf Graph test input
    X_test = tf.placeholder("float", [None, num_test_steps, num_input])
    Y_test = tf.placeholder("float", [None, num_test_steps, num_input])
    with tf.variable_scope("Model", reuse=True):      
        test_pred = Model(X_test, False, config)
        # Define test loss 
        test_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(test_pred, Y_test)))

# Define optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate)
train_op = optimizer.minimize(train_loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


# Start training
with tf.Session() as sess:
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.save_path + '/train',sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.save_path + '/test')

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = dataset.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, tr_loss = sess.run([merged, train_loss], feed_dict={X: batch_x,Y: batch_y})
            train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
            train_writer.add_summary(summary, step)
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(tr_loss) )
    print("Optimization Finished!")

    # Calculate accuracy for valid inps
    valid_data = dataset.validation.inps.reshape((-1, num_steps, num_input))
    valid_label = dataset.validation.outs
    va_loss = sess.run(test_loss, feed_dict={X_test: valid_data, Y_test: valid_label})
    print("Validation Loss:", va_loss)

    # Fetch test prediction
    fetches = {
        "true":Y_test,
        "pred":test_pred,
        "loss":test_loss
    }
    # Calculate accuracy for test inps
    test_data = dataset.test.inps.reshape((-1, num_test_steps, num_input))
    test_label = dataset.test.outs
    test_vals = sess.run(fetches, feed_dict={X_test: test_data, Y_test: test_label})
    print("Testing Loss:", test_vals["loss"])

    # Save the variables to disk.
    save_path = saver.save(sess, FLAGS.save_path)
    print("Model saved in file: %s" % save_path)
    # Save predictions 
    numpy.save(save_path+"predict.npy", (test_vals["true"], test_vals["pred"]))
    # Save config file
    with open(save_path+"config.out", 'w') as f:
        f.write('num_layers:'+ str(config.num_layers) +'\t'+'hidden_size:'+ str(config.hidden_size)+
            '\t'+ 'num_steps:'+ str(config.num_steps) +
            '\t'+ 'learning_rate:'+ str(config.learning_rate)  +'\t'+ 'err_prop:'+ str(config.use_error_prop) + '\n')
        f.write('train_error:'+ str(tr_loss) + '\t' + 'valid_error:'+ str(va_loss) + 
                '\t'+ 'test_error:'+ str(test_vals["loss"]) + '\n')

