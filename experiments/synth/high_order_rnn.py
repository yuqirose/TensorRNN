import tensorflow as tf
from collections import deque
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs


class TensorRNNCell(RNNCell):
    """RNN cell with high order correlations"""
    def __init__(self, num_units, num_lags, input_size=None, activation=tanh):
        self._num_units = num_units
        self._num_lags = num_lags
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units * self._num_lags

    @property
    def output_size(self):
        return self._num_units
    
    def __call__(self, inputs, states, scope=None):
        """Now we have multiple states, state->states"""
        with vs.variable_scope(scope or "tensor_rnn_cell"):
            output = tensor_network( inputs, states, self._num_units, self._num_lags, True, scope=scope)
            output = self._activation(output)
        return output, output

def tensor_network(inputs, states, output_size, bias, bias_start=0.0, scope=None):
    """tensor network [inputs, states]-> output with tensor models"""
    print(type(states))
    return states 



def tensor_rnn(cell, inputs, num_steps, num_lags, initial_states):
    """High Order Recurrent Neural Network Layer
    """
    #tuple of 2-d tensor (batch_size, s)
    outputs = []
    states_list = initial_states #list of high order states
    with tf.variable_scope("tensor_rnn"):
        for time_step in range(num_steps):
            # take num_lags history
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
    """Transform a list of state tuples into an augmented tuple state
    customizable function, depends on how long history is used"""
    num_layers = len(states_list[0])# state = (layer1, layer2...), layer1 = (c,h), c = tensor(batch_size, num_steps)
    output_states = ()
    for layer in range(num_layers):
        output_state = ()
        for states in states_list:
            #c,h = states[layer] for LSTM
            output_state += (states[layer],)
        output_states += (output_state,)
        # new cell has s*num_lags states 
        print("layer %d"%layer, len(output_states[layer]))
    return output_states
