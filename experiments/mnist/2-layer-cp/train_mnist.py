
# coding: utf-8

# In[1]:

'''
MNIST script modified from TensorNet
Github URL: https://github.com/timgaripov/TensorNet-TF
Local: lisbon:/home/roseyu/Python/TensorNet-TF
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path
import datetime
import shutil
import time
import numpy as np
import tensorflow.python.platform
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import net
import sys
sys.path.append('../../')
from utils.train_utils import *

# Basic model parameters as external flags.
flags = tf.app.flags

flags.DEFINE_integer("num_steps", 12,
                  "Output sequence length")
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', int(1e5), 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('overview_steps', 100, 'Overview period')
flags.DEFINE_integer('evaluation_steps', 1000, 'Overview period')
flags.DEFINE_string('data_dir', '../../data', 'Directory to put the training data.')
flags.DEFINE_string('log_dir', '../../log', 'Directory to put log files.')


# In[2]:

'''
MNIST dataset
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(FLAGS.data_dir+'/MNIST_data', one_hot=False)

def evaluation(sess,
            eval_correct,
            loss,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    sum_loss = 0.0
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set.next_batch(FLAGS.batch_size),
                                   train_phase=False)
        res = sess.run([loss, eval_correct], feed_dict=feed_dict)
        sum_loss += res[0]
        true_count += res[1]
    precision = true_count / num_examples
    avg_loss = sum_loss / (num_examples / FLAGS.batch_size)
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f  Loss: %.2f' %
          (num_examples, true_count, precision, avg_loss))
    return precision, avg_loss


def run_training(extra_opts={}):
    start = datetime.datetime.now()
    start_str = start.strftime('%d-%m-%Y_%H_%M')
    #train, validation = input_data.read_data_sets(FLAGS.data_dir)

    train = mnist.train
    validation = mnist.test#mnist.validation

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        net.build(extra_opts)
        #Precision summaries
        precision_train = tf.Variable(0.0, trainable=False, name='precision_train')
        precision_validation = tf.Variable(0.0, trainable=False, name='precision_validation')

        precision_train_summary = tf.scalar_summary('precision/train',
                                                    precision_train,
                                                    name='summary/precision/train')

        precision_validation_summary = tf.scalar_summary('precision/validation',
                                                         precision_validation,
                                                         name='summary/precision/validation')
        graph = tf.get_default_graph()
        loss = graph.get_tensor_by_name('loss:0')
        train_op = graph.get_tensor_by_name('train_op:0')
        correct_count = graph.get_tensor_by_name('correct_count:0')
        #Create summary stuff
        regular_summaries_names = ['loss', 'learning_rate']
        regular_summaries_list = []
        for name in regular_summaries_names:
            summary = graph.get_tensor_by_name('summary/' + name + ':0')
            regular_summaries_list.append(summary)
        regular_summaries = tf.merge_summary(regular_summaries_list, name='summary/regular_summaries')
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(graph=graph,
                          config=tf.ConfigProto(intra_op_parallelism_threads=3, inter_op_parallelism_threads = 3))
        sess.run(init)
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir,
                                                graph_def=sess.graph_def)
        # And then after everything is built, start the training loop.
        for step in xrange(1, FLAGS.max_steps + 1):
            start_time = time.time()
            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(train.next_batch(FLAGS.batch_size))
            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.


            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)
            duration = time.time() - start_time
            # Write the summaries and print an overview fairly often.
            if step % FLAGS.overview_steps == 0:
                # Print status to stdout.
                data_per_sec = FLAGS.batch_size / duration
                print('Step %d: loss = %.2f (%.3f sec) [%.2f data/s]' %
                      (step, loss_value, duration, data_per_sec))
                # Update the events file.
                summary_str = sess.run(regular_summaries, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save a checkpoint and evaluate the model periodically.
            if (step) % FLAGS.evaluation_steps == 0 or step == FLAGS.max_steps:
                saver.save(sess, FLAGS.log_dir +'/checkpoint', global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                precision_t, obj_t = evaluation(sess,
                                             correct_count,
                                             loss,
                                             train)
                sess.run(precision_train.assign(precision_t))

                # Evaluate against the validation set.
                print('Validation Data Eval:')
                precision_v, obj_v = evaluation(sess,
                                             correct_count,
                                             loss,
                                             validation)
                sess.run(precision_validation.assign(precision_v))

                summary_str_0, summary_str_1 = sess.run([precision_train_summary, precision_validation_summary])
                summary_writer.add_summary(summary_str_0, step)
                summary_writer.add_summary(summary_str_1, step)
                if not os.path.exists('./results'):
                    os.makedirs('./results')
                res_file = open('./results/res_' + str(start_str), 'w')

                res_file.write('Iterations: ' + str(step) + '\n')
                now = datetime.datetime.now()
                delta = now - start
                res_file.write('Learning time: {0:.2f} minutes\n'.format(delta.total_seconds() / 60.0))
                res_file.write('Train precision: {0:.5f}\n'.format(precision_t))
                res_file.write('Train loss: {0:.5f}\n'.format(obj_t))
                res_file.write('Validation precision: {0:.5f}\n'.format(precision_v))
                res_file.write('Validation loss: {0:.5f}\n'.format(obj_v))
                res_file.write('Extra opts: ' + str(extra_opts) + '\n')
                res_file.write('Code:\n')
                net_file = open('./net.py', 'r')
                shutil.copyfileobj(net_file, res_file)
                net_file.close()
                res_file.close()
        return {'precision':precision_v}

def main(_):
    layers = []
    layer = {}
    layer['inp_modes'] = np.array([4, 7, 4, 7], dtype='int32') # 28 * 28
    layer['out_modes'] = np.array([3, 4, 5, 5], dtype='int32') # 300
    layers.append(layer) #300

    layer['inp_modes'] = layer['out_modes']
    layer['out_modes'] = np.array([2, 2, 5, 5], dtype='int32') # 100
    layers.append(layer) #100


    err_rslt = []
    compres_rate = []
    for rank_val in range(10,40,5):
        extra_opts={}
        cp_layer_ranks = [rank_val,rank_val]
        extra_opts['ranks_1'] = cp_layer_ranks[0]
        extra_opts['ranks_2'] = cp_layer_ranks[1]
        rate = compres_ratio_cp(layers, cp_layer_ranks)
        compres_rate.append(rate)
        rslt = run_training(extra_opts)
        err_rslt.append(rslt['precision'])
    with open('./results/res_cp_LeNet300','w') as f:
        for i in range(len(err_rslt)):
            f.write('{0:3}:{1:.5f}\t'.format(compress_rate[i],err_rslt[i]))


if __name__ == '__main__':
    tf.app.run()

