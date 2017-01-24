import tensorflow as tf
from collections import deque

def tensor_rnn(cell, inputs, num_steps, num_lags, initial_states):
    """High Order Recurrent Neural Network Layer
    """
    #tuple of 2-d tensor (batch_size, s)
    outputs = []
    states_list = initial_states #list of high order states
    with tf.variable_scope("tensor_rnn"):
        for time_step in range(num_steps):
            # take num_lag history
            if time_step > num_lags:
                tf.get_variable_scope().reuse_variable()
            states = _list_to_states(states_list) 
            input_slice = tf.slice(inputs, [0,time_step, 0], [-1,num_lags, -1])
            (cell_output, state)=cell(input_slice, states)
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

def _list_to_states(states_list):
    """Transform a list of state tuples into an augmented tuple state"""
    num_layers = len(states_list[0])
    states = ()
    for layer in range(num_layers):
        states = states + (tf.concat(0, [tf.pack(state[layer]) for state in states_list]),) 
        # TBD: pack distroys the structure of LSTM state
        # each is state is a tuple of len num_layers
        print("layer %d"%layer,tf.shape(states[layer]))
    return states 
