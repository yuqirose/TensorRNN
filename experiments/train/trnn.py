from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops

def _hidden_to_output(h, hidden_size, input_size):
    softmax_w = tf.get_variable("softmax_w", [hidden_size, input_size], dtype= tf.float32)
    softmax_b = tf.get_variable("softmax_b", [input_size], dtype=tf.float32)
    output = tf.matmul(h, softmax_w) + softmax_b
    return output, softmax_w, softmax_b


def rnn_with_feed_prev(cell, inputs, feed_prev, config):
    
    prev = None
    outputs = []

    if feed_prev:
      print("Creating model --> Feeding output back into input.")
    else:
      print("Creating model input = ground truth each timestep.")

    with tf.variable_scope("rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        first_input = inputs[0]
        input_shape = first_input.get_shape().with_rank_at_least(2)
        batch_size = array_ops.shape(first_input)[0]
        input_size = input_shape[1]
        


        burn_in_steps = config.burn_in_steps
        # print('batch size','input_size', batch_size, input_size)
        output_size = cell.output_size
        initial_state = cell.zero_state(batch_size, dtype= tf.float32)
        state = initial_state

        for time_step, inp in enumerate(inputs):

            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            if feed_prev and prev is not None and time_step >= burn_in_steps:
                inp, _, _ = _hidden_to_output(prev, output_size, input_size)
                print('feed_prev inp shape', inp.get_shape())
                print("t", time_step, ">=", burn_in_steps, "--> feeding back output into input.")

            (cell_output, state) = cell(inp, state)
            outputs.append(cell_output)

            if feed_prev:
              prev = cell_output

    return outputs, state