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


class TensorRNNCell(RNNCell):
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
            output = tensor_network_tt( inputs, states, self._num_units,self._rank_vals, True, scope=scope)
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
            new_state = self._activation(output)
        if self._state_is_tuple:
            new_state = (new_state)
        return new_state, new_state


def tensor_network_linear(inputs, states, output_size, bias, bias_start=0.0, scope=None):
    """tensor network [inputs, states]-> output with tensor models"""
    # each coordinate of hidden state is independent- parallel
    states_tensor  = nest.flatten(states)
    total_inputs = [inputs]
    total_inputs.extend(states)
    output = _linear(total_inputs, output_size, True, scope=scope)
    return output


def tensor_network(inputs, states, output_size, num_orders, bias, bias_start=0.0, scope=None):
    """form a high-order full tenosr """
    num_lags = len(states)
    batch_size = inputs.get_shape()[0].value
    state_size = output_size #hidden layer size
    input_size= inputs.get_shape()[1].value

    with vs.variable_scope(scope or "tensor_network"):
        total_state_size = (state_size * num_lags + 1 )
        mat_dims = np.ones((num_orders,)) * total_state_size
        mat_size = np.power(total_state_size, num_orders)

        weights_x = vs.get_variable("weights_x", [input_size, output_size] )
        out_x = tf.matmul(inputs, weights_x)
        weights_h = vs.get_variable("weights_h", [mat_size, output_size]) # h_z x h_z... x output_size

        #mat = tf.Variable(mat, name="weights")
        states_vector = tf.concat(1, states)
        states_vector = tf.concat(1, [states_vector, tf.ones([batch_size, 1])])
        """form high order state tensor"""
        states_tensor = states_vector
        for order in range(num_orders-1):
            states_tensor = _outer_product(batch_size, states_tensor, states_vector)
        out_h = tf.reshape(states_tensor, [batch_size,-1]) # batch_size x hidden_size
        out_h = tf.matmul(out_h, weights_h)
        res = tf.reshape(tf.add(out_x, out_h) ,[-1, output_size],name="res")
        if not bias:
            return
        biases = vs.get_variable("biases", [output_size])
        return  nn_ops.bias_add(res,biases)


def tensor_network_tt(inputs, states, output_size, rank_vals, bias, bias_start=0.0, scope=None):
    """tensor train decomposition for the full tenosr """
    num_orders = len(rank_vals)+1#alpha_1 to alpha_{K-1}
    num_lags = len(states)
    batch_size = inputs.get_shape()[0].value
    state_size = output_size #hidden layer size
    input_size= inputs.get_shape()[1].value

    with vs.variable_scope(scope or "tensor_network_tt"):
        total_state_size = (state_size * num_lags + 1 )
        mat_dims = np.ones((num_orders,)) * total_state_size
        mat_ranks = np.concatenate(([1], rank_vals, [output_size]))
        mat_ps = np.cumsum(np.concatenate(([0], mat_ranks[:-1] * mat_dims * mat_ranks[1:])),dtype=np.int32)
        mat_size = mat_ps[-1]

        weights_x = vs.get_variable("weights_x", [input_size, output_size] )
        out_x = tf.matmul(inputs, weights_x)
        mat = vs.get_variable("weights_h", mat_size) # h_z x h_z... x output_size

        #mat = tf.Variable(mat, name="weights")
        states_vector = tf.concat(1, states)
        states_vector = tf.concat(1, [states_vector, tf.ones([batch_size, 1])])
        """form high order state tensor"""
        states_tensor = states_vector
        for order in range(num_orders-1):
            states_tensor = _outer_product(batch_size, states_tensor, states_vector)
        out_h = tf.reshape(states_tensor, [batch_size,-1]) # batch_size x hidden_size
        out_h = tf.transpose(out_h, [1,0])
        for i in range(num_orders):
            out_h = tf.reshape(out_h, [mat_ranks[i] * total_state_size, -1])
            mat_core = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]])
            mat_core = tf.reshape(mat_core, [mat_ranks[i] * total_state_size, mat_ranks[i + 1]])
            mat_core = tf.transpose(mat_core, [1, 0])

            out_h = tf.matmul(mat_core, out_h)
            out_h = tf.reshape(out_h, [1,-1])
            out_h = tf.transpose(out_h, [1,0])
            #print(i,out_h.get_shape())
        out_h = tf.reshape(out_h, [output_size, -1])
        out_h = tf.transpose(out_h, [1, 0])
        res = tf.reshape(tf.add(out_x, out_h) ,[-1, output_size],name="res")
        if not bias:
            return
        biases = vs.get_variable("biases", [output_size])
        return  nn_ops.bias_add(res,biases)


def tensor_network_tt_einsum(inputs, states, output_size, rank_vals, bias, bias_start=0.0, scope=None):

  print("Using Einsum Tensor-Train decomposition.")

  """tensor train decomposition for the full tenosr """
  num_orders = len(rank_vals)+1#alpha_1 to alpha_{K-1}
  num_lags = len(states)
  batch_size = inputs.get_shape()[0].value
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
    states_vector = tf.concat(1, states)
    states_vector = tf.concat(1, [states_vector, tf.ones([batch_size, 1])])
    """form high order state tensor"""
    states_tensor = states_vector
    for order in range(num_orders-1):
      states_tensor = _outer_product(batch_size, states_tensor, states_vector)

    print("tensor product", states_tensor.name, states_tensor.get_shape().as_list())

    def _tensor_net_tt_einsum(states_tensor):
      print("input:", states_tensor.name, states_tensor.get_shape().as_list())
      print("mat_dims", mat_dims)
      print("mat_ranks", mat_ranks)
      print("mat_ps", mat_ps)
      print("mat_size", mat_size)

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

      x = "z" + ijk[:num_orders] # "z" is the batch dimension

      # First latent factor
      i = 0
      mat_core = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]])
      mat_core = tf.reshape(mat_core, [mat_ranks[i], total_state_size, mat_ranks[i + 1]])

      print mat_core.get_shape().as_list()

      _s3 = x[:1] + x[2:] + "ab"
      einsum = "aib," + x + "->" + _s3
      x = _s3
      print "order", i, einsum

      out_h = tf.einsum(einsum, mat_core, states_tensor)
      print(out_h.name, out_h.get_shape().as_list())

      # 2nd - penultimate latent factor
      for i in range(1, num_orders - 1):

        # We now compute the tensor inner product W * H, where W is decomposed
        # into a tensor-train with D factors A^i. Each factor A^i is a 3-tensor,
        # with dimensions [mat_rank[i], hidden_size, mat_rank[i+1] ]
        # The lag index, indexing the components of the state vector H,
        # runs from 1 <= i < K.

        # Fetch the weights of factor A^i from our big serialized variable weights_h.
        mat_core = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]])
        mat_core = tf.reshape(mat_core, [mat_ranks[i], total_state_size, mat_ranks[i + 1]])

        print mat_core.get_shape().as_list()

        einsum, x = ss, _s3 = _get_einsum(i, x)

        print "order", i, ss

        out_h = tf.einsum(einsum, mat_core, out_h)
        print(out_h.name, out_h.get_shape().as_list())


      # last latent factor
      i = num_orders - 1
      mat_core = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]])
      mat_core = tf.reshape(mat_core, [mat_ranks[i], total_state_size, mat_ranks[i + 1]])
      print mat_core.get_shape().as_list()

      einsum, _s3 = _get_einsum(num_orders - 1, x)
      print "order", i, einsum

      out_h = tf.einsum(einsum, mat_core, out_h)
      print(out_h.name, out_h.get_shape().as_list())

      print "Squeeze out the dimension-1 dummy dim (first dim of 1st latent factor)"

      out_h = tf.squeeze(out_h, [1])

      # Compute h_t = U*x_t + W*H_{t-1}
      res = tf.reshape(tf.add(out_x, out_h), [-1, output_size],name="res")

      print res.get_shape()

      # biases = vs.get_variable("biases", [output_size])
      return res

    res = _tensor_net_tt_einsum(states_tensor)

    print "END OF CELL CONSTRUCTION"
    print "========================"
    print ""

    if not bias:
      return res
    biases = vs.get_variable("biases", [output_size])

  return nn_ops.bias_add(res,biases)


def _outer_product(batch_size, tensor, vector):
    """tensor-vector outer-product"""
    tensor_flat= tf.expand_dims(tf.reshape(tensor, [batch_size,-1]), 2)
    vector_flat = tf.expand_dims(vector, 1)
    res = tf.batch_matmul(tensor_flat, vector_flat)
    new_shape =  [batch_size]+_shape_value(tensor)[1:]+_shape_value(vector)[1:]
    res = tf.reshape(res, new_shape )
    return res


def _shape_value(tensor):
    shape = tensor.get_shape()
    return [s.value for s in shape]


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
        res = math_ops.matmul(tf.concat(1, args), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            biases = vs.get_variable("biases", [output_size], dtype=dtype)
    return  nn_ops.bias_add(res,biases)


def tensor_rnn(cell, inputs, num_steps, num_lags, initial_states):
    """High Order Recurrent Neural Network Layer
    """
    #tuple of 2-d tensor (batch_size, s)
    outputs = []
    states_list = initial_states #list of high order states
    with tf.variable_scope("tensor_rnn"):
        for time_step in range(num_steps):
            # take num_lags history
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            states = _list_to_states(states_list)
            """input tensor is [batch_size, num_steps, input_size]"""
            input_slice = inputs[:, time_step, :]#tf.slice(inputs, [0,time_step, 0], [-1,num_lags, -1])
            (cell_output, state)=cell(input_slice, states)
            outputs.append(cell_output)
            states_list = _shift(states_list, state)
    return outputs, states


def tensor_rnn_with_feed_prev(cell, inputs, num_steps, size, num_lags,
    initial_states, vocab_size, feed_prev=False, burn_in_steps=0):
    """High Order Recurrent Neural Network Layer
    """
    #tuple of 2-d tensor (batch_size, s)

    _logits = []
    _cell_outputs = []
    _outputs = []
    weights = {}
    states_list = initial_states #list of high order states

    prev = None
    print(cell.state_size)

    if feed_prev:
      print("Creating model @ not training --> Feeding output back into input.")
    else:
      print("Creating model @ training --> input = ground truth each timestep.")

    def _hidden_to_input(h):
      softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype= tf.float32)
      softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
      logits = tf.matmul(h, softmax_w) + softmax_b
      return logits, softmax_w, softmax_b

    with tf.variable_scope("tensor_rnn"):
      for time_step in range(num_steps):

        if time_step > 0:
          tf.get_variable_scope().reuse_variables()

        inp = inputs[:, time_step, :]


        if feed_prev and prev is not None and time_step >= burn_in_steps:
          inp, _, _ = _hidden_to_input(prev)
          print("t", time_step, ">=", burn_in_steps, "--> feeding back output into input.")

        states = _list_to_states(states_list)
        """input tensor is [batch_size, num_steps, input_size]"""
        input_slice = inputs[:, time_step, :]#tf.slice(inputs, [0,time_step, 0], [-1,num_lags, -1])

        (cell_output, state)=cell(input_slice, states)
        _cell_outputs.append(cell_output)

        states_list = _shift(states_list, state)

        if feed_prev:
          prev = cell_output

        output, w, b = _hidden_to_input(cell_output)
        _outputs.append(output)

      weights["softmax_w"] = w
      weights["softmax_b"] = b

    logits = tf.reshape(tf.concat(1, _outputs), [-1, vocab_size])

    return logits, states, weights


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
