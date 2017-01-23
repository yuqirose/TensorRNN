import tensorflow as tf
from collections import deque

def tensor_rnn(cell, inputs, num_steps, num_lags, initial_states):
    """High Order Recurrent Neural Network Layer
    """
    outputs = []
    states_list = tf.unstack(initial_states) #list of high order states
    with tf.variable_scope("tensor_rnn"):
        for time_step in range(num_steps):
            # take num_lag history
            if time_step > num_lags:
                tf.get_variable_scope().reuse_variable()
            states = tf.pack(states_list) 
            (cell_output, state)=cell(inputs[:,time_step-num_lags:time_step,:], states)
            outputs.append(cell_output)
            states_list = _shift(states_list, state)
    return outputs, states

def _shift (input_list, new_item):
    """Update lag number of states"""
    input_list = deque(input_list)
    input_list.append(new_item) 
    input_list.rotate(1) # The deque is now: [3, 1, 2]
    input_list.rotate(-1) # Returns deque to original state: [1, 2, 3]
    output_list = list(input_list.popleft()) # deque == [2, 3]
    return output_list
