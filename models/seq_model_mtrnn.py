import tensorflow as tf
import numpy as np
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.contrib.rnn import MultiRNNCell, DropoutWrapper
from models.high_order_rnn import EinsumTensorRNNCell, tensor_rnn, tensor_rnn_with_feed_prev
class PTBModel(object):

  def __init__(self, is_training, config, input_, use_error_prop=False):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    hidden_size = config.hidden_size
    input_size = input_.input_size
    num_lags = config.num_lags
    num_orders = config.num_orders
    rank_vals = config.rank_vals

    initializer = tf.random_uniform_initializer(-1,1)
    rnn_cell = EinsumTensorRNNCell(hidden_size, num_lags, rank_vals)

    if is_training and config.keep_prob < 1:
      rnn_cell = DropoutWrapper(rnn_cell, output_keep_prob=config.keep_prob)
      
    cell = MultiRNNCell([rnn_cell] * config.num_layers, state_is_tuple=True)
    initial_states = []
    for lag in range(num_lags):
      initial_state =  cell.zero_state(batch_size, dtype= tf.float32)
      initial_states.append(initial_state)

    self._initial_states = initial_states

    print("num_steps:", num_steps)

    with tf.device("/cpu:0"):
      inputs = input_.input_data

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # print("Predictions now computed inside cell.")
    feed_prev = not is_training if use_error_prop else False
    logits, state, weights  = tensor_rnn_with_feed_prev(cell, inputs, num_steps, hidden_size,
      num_lags, self._initial_states, input_size, feed_prev=feed_prev, burn_in_steps=config.burn_in_steps)

    # softmax_w, softmax_b = weights["softmax_w"], weights["softmax_b"]

    self._predict = logits

    self._cost = cost = tf.reduce_mean(tf.squared_difference(
      tf.reshape(logits, [-1, input_size]), tf.reshape(input_.targets, [batch_size*num_steps,-1]) ))

    self._final_state = state

    # calculate number of training parameters
    total_state_size = (hidden_size * num_lags + 1 )
    mat_ranks = np.concatenate(([1], rank_vals, [hidden_size]))
    mat_dims = np.ones((num_orders,)) * total_state_size
    num_params = np.sum(mat_ranks[:-1] * mat_dims * mat_ranks[1:])
    print("number of parameters : %d " % (num_params))


    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars),
      global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
      tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
    
    
  @property
  def input(self):
    return self._input
  @property
  def predict(self):
    return self._predict

  @property
  def initial_states(self):
    return self._initial_states

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op



