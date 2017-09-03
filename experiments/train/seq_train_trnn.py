from __future__ import division
from __future__ import print_function

import time
import numpy as np
import pandas as pd
import tensorflow as tf
import sys, os

os.sys.path.append("../../")
from models.seq_input import *
from models.seq_model_trnn import *
from train_config import *


flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", "/Users/roseyu/Documents/Python/data/lorenz.npy",
          "Where the training/test data is stored.")
flags.DEFINE_string("save_path", "/Users/roseyu/Documents/Python/lorenz/tt_rnn/",
          "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
          "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("use_error_prop", False,
                  "Feed previous output as input in RNN")

flags.DEFINE_integer("hidden_size", 16, "hidden layer size")
flags.DEFINE_float("learning_rate", 1e-1, "learning rate")
flags.DEFINE_integer("num_train_steps",10,"Output sequence length")
flags.DEFINE_integer("num_test_steps", 10,"Output sequence length")
flags.DEFINE_integer("rank_val","1", "rank of tensor train model")

FLAGS = flags.FLAGS


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  predicts = []
  targets = []
  initial_state = session.run(model.initial_state)

  fetches = {
    "cost": model.cost,
    "final_state": model.final_state,
    "predict": model.predict,
    "input": model.input.input_data,
    "target": model.input.targets
  }

  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, s in enumerate(model.initial_state):
      for j, ss in enumerate(s):
        # This is different from the tutorial
        # for every epoch, we reset the hidden state to zero
        feed_dict[ss] = initial_state[i][j]

  vals = session.run(fetches, feed_dict)

  cost = vals["cost"]
  target = vals["target"]
  predict = vals["predict"] # batch_size x num_step x input_size
  state = vals["final_state"]

  # print("step {0}: predict{1}, cost {2}".format(step, predict, cost))
  # print("cost at step {0}: {1}".format(step, cost))

  # if step %20 == 0:
    
  #    print("step", step, "input\n", vals["input"][0,0:5])
  
  #    print("step", step, "target\n", vals["target"][0,0:5])
 
  #    print("step", step, "predicts\n",vals["predict"][0,0:5])
  # if eval_op is None:
  #   print("target step", step)
  #   print(target[0,:5,1])

  
  predicts.append(predict)
  targets.append(target)

  costs += cost
  iters += 1 #model.input.num_steps

  if verbose and step % (model.input.epoch_size // 10) == 10:
    print("%.3f error: %.3f speed: %.0f wps" %
        (step * 1.0 / model.input.epoch_size, np.sqrt(costs / iters),
         iters * model.input.batch_size / (time.time() - start_time)))

  final_cost = np.sqrt(costs/iters)
  final_rslt = (targets, predicts) 
    
  return final_cost, final_rslt


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = seq_raw_data(FLAGS.data_path)#seq raw data
  train_data, valid_data, test_data = raw_data
  config = TrainConfig()
  config.learning_rate = FLAGS.learning_rate
  config.hidden_size = FLAGS.hidden_size
  config.num_steps = FLAGS.num_train_steps
  config.rank_vals = [FLAGS.rank_val]

  eval_config = TestConfig()
  eval_config.hidden_size = FLAGS.hidden_size
  eval_config.num_steps = FLAGS.num_test_steps
  eval_config.rank_vals = [FLAGS.rank_val]

  if FLAGS.use_error_prop:
        print("Using error prop in RNN!")


  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                          config.init_scale)
    with tf.name_scope("Train"):
      train_input = PTBInput(is_training=True, config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input, use_error_prop=False)
      tf.summary.scalar("Training_Loss", m.cost)
      tf.summary.scalar("Learning_Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(is_training=False, config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input, use_error_prop=False)
      tf.summary.scalar("Validation_Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(is_training=False, config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config, input_=test_input, use_error_prop=FLAGS.use_error_prop)
      tf.summary.scalar("Test_Loss", mtest.cost)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path,save_summaries_secs=10)
    with sv.managed_session(config = tf.ConfigProto(gpu_options=gpu_options)) as session:
      valid_err_old = float('inf') 
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.2E" % (i + 1, session.run(m.lr)))
        train_err, _= run_epoch(session, m, eval_op=m.train_op,
                       verbose=True)
        print("Epoch: %d Train Error: %.3f" % (i + 1, train_err))
        valid_err, _ = run_epoch(session, mvalid)
        print("Epoch: %d Valid Error: %.3f" % (i + 1, valid_err))
        # early stopping
        # if valid_err >= valid_err_old :
        #   print("Early stopping after %d epoch" % i)
        #   break
        valid_err_old = valid_err



  #  if i and i % 10 == 0:
      test_err, test_rslt = run_epoch(session, mtest)
      print("Test Error: %.3f" % test_err)
      np.save(FLAGS.save_path+"predict.npy", test_rslt)

      # Write config file
      with open(FLAGS.save_path+"config_error.out", 'w') as f:
        f.write('num_layers:'+ str(config.num_layers) +'\t'+'hidden_size:'+ str(config.hidden_size)+
            '\t'+ 'num_steps:'+ str(config.num_steps) +
            '\t'+ 'learning_rate:'+ str(config.learning_rate)  +'\t'+ 'err_prop:'+ str(FLAGS.use_error_prop) + '\n')
        f.write('train_error:'+ str(train_err) + '\t' + 'valid_error:'+ str(valid_err) + 
                        '\t'+ 'test_error:'+ str(test_err) + '\n')

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()
