from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.util import nest
from tensorflow.contrib.distributions import Bernoulli
from tensorflow.contrib.layers import fully_connected


import numpy as np
import copy
from collections import deque

class MatrixRNNCell(RNNCell):
    """RNN cell with first order concatenation of hidden states"""
    def __init__(self, num_units, num_lags, input_size=None, state_is_tuple=True, activation=tanh):
        self._num_units = num_units
        self._num_lags = num_lags
    #rank of the tensor, tensor-train model is order+1
        self._state_is_tuple= state_is_tuple
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, states, scope=None):
        """Now we have multiple states, state->states"""

        with vs.variable_scope(scope or "tensor_rnn_cell"):
            output = tensor_network_linear( inputs, states, self._num_units, True, scope=scope)
            new_state = self._activation(output)
        if self._state_is_tuple:
            new_state = (new_state)
        return new_state, new_state

class EinsumTensorRNNCell(RNNCell):
    """RNN cell with high order correlations"""
    def __init__(self, num_units, num_lags, rank_vals, input_size=None, state_is_tuple=True, activation=tanh):
            self._num_units = num_units
            self._num_lags = num_lags
    #rank of the tensor, tensor-train model is order+1
            self._rank_vals = rank_vals
            #self._num_orders = num_orders
            self._state_is_tuple= state_is_tuple
            self._activation = activation

    @property
    def state_size(self):
            return self._num_units

    @property
    def output_size(self):
            return self._num_units

    def __call__(self, inputs, states, scope=None):
            """Now we have multiple states, state->states"""

            with vs.variable_scope(scope or "tensor_rnn_cell"):
                    output = tensor_network_tt_einsum( inputs, states, self._num_units,self._rank_vals, True, scope=scope)
                    # dense = tf.contrib.layers.fully_connected(output, self._num_units, activation_fn=None, scope=scope)
                    # output = tf.contrib.layers.batch_norm(output, center=True, scale=True, 
                    #                               is_training=True, scope=scope)
                    new_state = self._activation(output)
            if self._state_is_tuple:
                    new_state = (new_state)
            return new_state, new_state

class MTRNNCell(RNNCell):
    """Multi-resolution Tensor RNN cell """
    def __init__(self, num_units, num_lags, num_freq, rank_vals, input_size=None, state_is_tuple=True, activation=tanh):
        self._num_units = num_units
        self._num_lags = num_lags
        self._num_freq =  num_freq # frequency for the 2nd tt state
    #rank of the tensor, tensor-train model is order+1
        self._rank_vals = rank_vals
        #self._num_orders = num_orders
        self._state_is_tuple= state_is_tuple
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, states, scope=None):
        """Now we have multiple states, state->states"""

        with vs.variable_scope(scope or "tensor_rnn_cell"):
            output = tensor_network_mtrnn( inputs, states, self._num_units,self._rank_vals, self._num_freq,True, scope=scope)
            # dense = tf.contrib.layers.fully_connected(output, self._num_units, activation_fn=None, scope=scope)
            # output = tf.contrib.layers.batch_norm(output, center=True, scale=True, 
            #                               is_training=True, scope=scope)
            new_state = self._activation(output)
        if self._state_is_tuple:
            new_state = (new_state)
        return new_state, new_state

def _hidden_to_output(h, hidden_size, input_size):
    out_w = tf.get_variable("out_w", [hidden_size, input_size], dtype= tf.float32)
    out_b = tf.get_variable("out_b", [input_size], dtype=tf.float32)
    output = tf.matmul(h, out_w) + out_b
    return output


def rnn_with_feed_prev(cell, inputs, is_training, config, initial_state=None):
    
    prev = None
    outputs = []
    sample_prob = config.sample_prob # scheduled sampling probability

    feed_prev = not is_training if config.use_error_prop else False
    is_sample = is_training and sample_prob > 0 

    if feed_prev:
        print("Creating model @ not training  --> Feeding output back into input.")
    else:
        print("Creating model @ training  input = ground truth each timestep.")

    with tf.variable_scope("rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        inputs_shape = inputs.get_shape().with_rank_at_least(3)
        batch_size = tf.shape(inputs)[0] 
        num_steps = inputs_shape[1]
        input_size = int(inputs_shape[2])
        burn_in_steps = config.burn_in_steps
        output_size = cell.output_size

        # phased lstm input
        inp_t = tf.expand_dims(tf.range(1,batch_size+1), 1)

        dist = Bernoulli(probs=sample_prob)
        samples = dist.sample(sample_shape=num_steps)
        # with tf.Session() as sess:
        #     print('bernoulli',samples.eval())
        if initial_state is None:
            initial_state = cell.zero_state(batch_size, dtype= tf.float32)
        state = initial_state

        for time_step in range(num_steps):

            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            inp = inputs[:, time_step, :]
            if is_sample and time_step > 0: 
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    inp = tf.cond(tf.cast(samples[time_step], tf.bool), lambda:fully_connected(cell_output, input_size, activation_fn=tf.sigmoid), \
                        lambda:tf.identity(inp) )
            if feed_prev and prev is not None and time_step >= burn_in_steps:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    inp = fully_connected(prev, input_size,  activation_fn=tf.sigmoid)
                    print("t", time_step, ">=", burn_in_steps, "--> feeding back output into input.")

            if isinstance(cell._cells[0], tf.contrib.rnn.PhasedLSTMCell):
                (cell_output, state) = cell((inp_t, inp), state)
            else:
                (cell_output, state) = cell(inp, state)

            prev = cell_output
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                output = fully_connected(cell_output, input_size, activation_fn=tf.sigmoid)
                outputs.append(output)

    outputs = tf.stack(outputs, 1)

    return outputs, state



def tensor_network_linear(inputs, states, output_size, bias, bias_start=0.0, scope=None):
    """tensor network [inputs, states]-> output with tensor models"""
    # each coordinate of hidden state is independent- parallel
    states_tensor  = nest.flatten(states)
    total_inputs = [inputs]
    total_inputs.extend(states)
    output = _linear(total_inputs, output_size, True, scope=scope)
    return output

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    total_arg_size = 0
    shapes= [a.get_shape() for a in args]
    for shape in shapes:
        total_arg_size += shape[1].value
    dtype = [a.dtype for a in args][0]

    scope = vs.get_variable_scope()

    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable("weights", [total_arg_size, output_size], dtype=dtype)
        """y = [batch_size x total_arg_size] * [total_arg_size x output_size]"""
        res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            biases = vs.get_variable("biases", [output_size], dtype=dtype)
    return  nn_ops.bias_add(res,biases)



def tensor_network_mtrnn(inputs, states, output_size, rank_vals, num_freq, bias, bias_start=0.0, scope=None):
    "states to output mapping for multi-resolution tensor rnn"
    """tensor train decomposition for the full tenosr """
    num_orders = len(rank_vals)+1#alpha_1 to alpha_{K-1}
    num_lags = len(states)
    batch_size = tf.shape(inputs)[0] 
    state_size = output_size #hidden layer size
    input_size= inputs.get_shape()[1].value

    with vs.variable_scope(scope or "tensor_network_mtrnn"):
        # input weights W_x 
        weights_x = vs.get_variable("weights_x", [input_size, output_size] )
        out_x = tf.matmul(inputs, weights_x)


        # 1st tensor train layer W_h
        total_state_size = (state_size * num_lags + 1 )
        mat_dims = np.ones((num_orders,)) * total_state_size
        mat_ranks = np.concatenate(([1], rank_vals, [output_size]))
        mat_ps = np.cumsum(np.concatenate(([0], mat_ranks[:-1] * mat_dims * mat_ranks[1:])),dtype=np.int32)
        mat_size = mat_ps[-1]
        mat = vs.get_variable("weights_h", mat_size) # h_z x h_z... x output_size

        states_vector = tf.concat(states, 1)
        states_vector = tf.concat([states_vector, tf.ones([batch_size, 1])],1)
        """form high order state tensor"""
        states_tensor = states_vector
        for order in range(num_orders-1):
            states_tensor = _outer_product(batch_size, states_tensor, states_vector)

        cores = []
        for i in range(num_orders):
            # Fetch the weights of factor A^i from our big serialized variable weights_h.
            mat_core = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]])
            mat_core = tf.reshape(mat_core, [mat_ranks[i], total_state_size, mat_ranks[i + 1]])   
            cores.append(mat_core)

        print('-'*80)  
        print('1st layer tensor train\n')
        print('|states res|', 1, '|states len|', len(states), '|states size|', states_tensor.get_shape())
        h_1 = tensor_train_contraction(states_tensor, cores)

        # 2nd tensor train layer W_h2
        total_state_size = (state_size * num_lags//num_freq + 1 )
        mat_dims = np.ones((num_orders,)) * total_state_size
        mat_ranks = np.concatenate(([1], rank_vals, [output_size]))
        mat_ps = np.cumsum(np.concatenate(([0], mat_ranks[:-1] * mat_dims * mat_ranks[1:])),dtype=np.int32)
        mat_size = mat_ps[-1]
        mat = vs.get_variable("weights_h2", mat_size) # h_z x h_z... x output_size


        new_states = states[::num_freq]
        states_vector = tf.concat(new_states,1)
        states_vector = tf.concat([states_vector, tf.ones([batch_size, 1])],1)
        """form high order state tensor"""
        states_tensor = states_vector
        for order in range(num_orders-1):
            states_tensor = _outer_product(batch_size, states_tensor, states_vector)

        cores = []
        for i in range(num_orders):
             # Fetch the weights of factor A^i from our big serialized variable weights_h.
             mat_core = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]])
             mat_core = tf.reshape(mat_core, [mat_ranks[i], total_state_size, mat_ranks[i + 1]])   
             cores.append(mat_core)
        print('-'*80)  
        print('2nd layer tensor train\n')
        print('|states res|', num_freq, '|states len|', len(new_states), '|states size|', states_tensor.get_shape())
        h_2 = tensor_train_contraction(states_tensor, cores)

        # Combine two tensor train 
        out_h = h_1 + h_2
        # Compute h_t = W_x*x_t + W_h*H_{t-1}
        res = tf.add(out_x, out_h)

        if not bias:
            return
        biases = vs.get_variable("biases", [output_size])
        return  nn_ops.bias_add(res,biases)

def tensor_train_contraction(states_tensor, cores):
    # print("input:", states_tensor.name, states_tensor.get_shape().as_list())
    # print("mat_dims", mat_dims)
    # print("mat_ranks", mat_ranks)
    # print("mat_ps", mat_ps)
    # print("mat_size", mat_size)

    abc = "abcdefgh"
    ijk = "ijklmnopqrstuvwxy"

    def _get_indices(r):
        indices = "%s%s%s" % (abc[r], ijk[r], abc[r+1])
        return indices

    def _get_einsum(i, s2):
        #
        s1 = _get_indices(i)
        _s1 = s1.replace(s1[1], "")
        _s2 = s2.replace(s2[1], "")
        _s3 = _s2 + _s1
        _s3 = _s3[:-3] + _s3[-1:]
        s3 = s1 + "," + s2 + "->" + _s3
        return s3, _s3

    num_orders = len(cores)
    # first factor
    x = "z" + ijk[:num_orders] # "z" is the batch dimension
    # print mat_core.get_shape().as_list()

    _s3 = x[:1] + x[2:] + "ab"
    einsum = "aib," + x + "->" + _s3
    x = _s3
    # print "order", i, einsum

    out_h = tf.einsum(einsum, cores[0], states_tensor)
    # print(out_h.name, out_h.get_shape().as_list())

    # 2nd - penultimate latent factor
    for i in range(1, num_orders):

        # We now compute the tensor inner product W * H, where W is decomposed
        # into a tensor-train with D factors A^i. Each factor A^i is a 3-tensor,
        # with dimensions [mat_rank[i], hidden_size, mat_rank[i+1] ]
        # The lag index, indexing the components of the state vector H,
        # runs from 1 <= i < K.

        # print mat_core.get_shape().as_list()

        einsum, x = ss, _s3 = _get_einsum(i, x)

        # print "order", i, ss

        out_h = tf.einsum(einsum, cores[i], out_h)
        # print(out_h.name, out_h.get_shape().as_list())

    # print "Squeeze out the dimension-1 dummy dim (first dim of 1st latent factor)"
    out_h = tf.squeeze(out_h, [1])
    return out_h


def tensor_network_tt_einsum(inputs, states, output_size, rank_vals, bias, bias_start=0.0, scope=None):

    # print("Using Einsum Tensor-Train decomposition.")

    """tensor train decomposition for the full tenosr """
    num_orders = len(rank_vals)+1#alpha_1 to alpha_{K-1}
    num_lags = len(states)
    batch_size = tf.shape(inputs)[0] 
    state_size = hidden_size = output_size #hidden layer size
    input_size= inputs.get_shape()[1].value

    with vs.variable_scope(scope or "tensor_network_tt"):

        total_state_size = (state_size * num_lags + 1 )

        # These bookkeeping variables hold the dimension information that we'll
        # use to store and access the transition tensor W efficiently.
        mat_dims = np.ones((num_orders,)) * total_state_size

        # The latent dimensions used in our tensor-train decomposition.
        # Each factor A^i is a 3-tensor, with dimensions [a_i, hidden_size, a_{i+1}]
        # with dimensions [mat_rank[i], hidden_size, mat_rank[i+1] ]
        # The last
        # entry is the output dimension, output_size: that dimension will be the
        # output.
        mat_ranks = np.concatenate(([1], rank_vals, [output_size]))

        # This stores the boundary indices for the factors A. Starting from 0,
        # each index i is computed by adding the number of weights in the i'th
        # factor A^i.
        mat_ps = np.cumsum(np.concatenate(([0], mat_ranks[:-1] * mat_dims * mat_ranks[1:])),dtype=np.int32)
        mat_size = mat_ps[-1]

        # Compute U * x
        weights_x = vs.get_variable("weights_x", [input_size, output_size] )
        out_x = tf.matmul(inputs, weights_x)

        # Get a variable that holds all the weights of the factors A^i of the
        # transition tensor W. All weights are stored serially, so we need to do
        # some bookkeeping to keep track of where each factor is stored.
        mat = vs.get_variable("weights_h", mat_size) # h_z x h_z... x output_size

        #mat = tf.Variable(mat, name="weights")
        states_vector = tf.concat(states, 1)
        states_vector = tf.concat( [states_vector, tf.ones([batch_size, 1])], 1)
        """form high order state tensor"""
        states_tensor = states_vector
        for order in range(num_orders-1):
            states_tensor = _outer_product(batch_size, states_tensor, states_vector)

        # print("tensor product", states_tensor.name, states_tensor.get_shape().as_list())
        cores = []
        for i in range(num_orders):
                # Fetch the weights of factor A^i from our big serialized variable weights_h.
                mat_core = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]])
                mat_core = tf.reshape(mat_core, [mat_ranks[i], total_state_size, mat_ranks[i + 1]])   
                cores.append(mat_core)
        out_h = tensor_train_contraction(states_tensor, cores)
        # Compute h_t = U*x_t + W*H_{t-1}
        res = tf.add(out_x, out_h)

        # print "END OF CELL CONSTRUCTION"
        # print "========================"
        # print ""

        if not bias:
            return res
        biases = vs.get_variable("biases", [output_size])

    return nn_ops.bias_add(res,biases)

def _outer_product(batch_size, tensor, vector):
    """tensor-vector outer-product"""
    tensor_flat= tf.expand_dims(tf.reshape(tensor, [batch_size,-1]), 2)
    vector_flat = tf.expand_dims(vector, 1)
    res = tf.matmul(tensor_flat, vector_flat)
    new_shape =  [batch_size]+_shape_value(tensor)[1:]+_shape_value(vector)[1:]
    res = tf.reshape(res, new_shape )
    return res

def _shape_value(tensor):
    shape = tensor.get_shape()
    return [s.value for s in shape]

def _shift (input_list, new_item):
    """Update lag number of states"""
    output_list = copy.copy(input_list)
    output_list = deque(output_list)
    output_list.append(new_item)
    output_list.rotate(1) # The deque is now: [3, 1, 2]
    output_list.popleft() # deque == [2, 3]
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
    return output_states

def tensor_rnn_with_feed_prev(cell, inputs, is_training, config, initial_states=None):
    """High Order Recurrent Neural Network Layer
    """
    #tuple of 2-d tensor (batch_size, s)
    outputs = []
    prev = None
    sample_prob = config.sample_prob # scheduled sampling probability
    feed_prev = not is_training if config.use_error_prop else False
    is_sample = is_training and sample_prob > 0 

    if feed_prev:
        print("Creating model @ not training --> Feeding output back into input.")
    else:
        print("Creating model @ training --> input = ground truth each timestep.")

    with tf.variable_scope("tensor_rnn") as varscope:
        if varscope.caching_device is None:
                    varscope.set_caching_device(lambda op: op.device)

        inputs_shape = inputs.get_shape().with_rank_at_least(3)
        batch_size = tf.shape(inputs)[0] 
        num_steps = inputs_shape[1]
        input_size = int(inputs_shape[2])
        output_size = cell.output_size
        burn_in_steps =  config.burn_in_steps

        dist = Bernoulli(probs=sample_prob)
        samples = dist.sample(sample_shape=num_steps)

        if initial_states is None:
            initial_states =[]
            for lag in range(config.num_lags):
                initial_state =  cell.zero_state(batch_size, dtype= tf.float32)
                initial_states.append(initial_state)

        states_list = initial_states #list of high order states
    
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            inp = inputs[:, time_step, :]

            if is_sample and time_step > 0: 
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    inp = tf.cond(tf.cast(samples[time_step], tf.bool), lambda:fully_connected(cell_output, input_size, activation_fn=tf.sigmoid), \
                        lambda:tf.identity(inp) )

            if feed_prev and prev is not None and time_step >= burn_in_steps:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    inp = fully_connected(cell_output, input_size, activation_fn=tf.sigmoid)
                    print("t", time_step, ">=", burn_in_steps, "--> feeding back output into input.")

            states = _list_to_states(states_list)
            """input tensor is [batch_size, num_steps, input_size]"""
            (cell_output, state)=cell(inp, states)

            states_list = _shift(states_list, state)

            prev = cell_output
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                output = fully_connected(cell_output, input_size, activation_fn=tf.sigmoid)
                outputs.append(output)

    outputs = tf.stack(outputs,1)
    return outputs, states_list



