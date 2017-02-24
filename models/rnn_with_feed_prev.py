import numpy as np
import copy
import tensorflow as tf
from collections import deque
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


def _rnn_loop(cell, inputs, num_steps, hidden_size, initial_state, vocab_size, feed_prev=False, burn_in_steps=0):

    prev = None
    _states = []
    _cell_outputs = []
    _outputs = []
    _weights = {}

    state = initial_state

    if feed_prev:
      print("Creating model --> Feeding output back into input.")
    else:
      print("Creating model input = ground truth each timestep.")

    def _hidden_to_input(h):
      softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype= tf.float32)
      softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
      logits = tf.matmul(h, softmax_w) + softmax_b
      return logits, softmax_w, softmax_b

    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):

            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            inp = inputs[:, time_step, :]

            if feed_prev and prev is not None and time_step >= burn_in_steps:
                inp, _, _ = _hidden_to_input(prev)
                print("t", timestep, ">=", burn_in_steps, "--> feeding back output into input.")

            (cell_output, state) = cell(inp, state)
            _cell_outputs.append(cell_output)
            _states.append(state)

            if feed_prev:
              prev = cell_output

            output, w, b = _hidden_to_input(cell_output)
            _outputs.append(output)

    _weights["softmax_w"] = w
    _weights["softmax_b"] = b

    logits = tf.reshape(tf.concat(1, _outputs), [-1, vocab_size])

    return logits, _states, _weights
