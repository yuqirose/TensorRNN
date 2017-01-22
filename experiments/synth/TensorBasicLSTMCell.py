
# coding: utf-8

# In[2]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
# %load unit_test.py
import tensorflow as tf
import numpy as np


from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

import sys, os
sys.path.append(os.path.abspath('../../'))

import tensornet
from tensorflow.python.ops.rnn_cell import *


class TensorBasicLSTMCell(LSTMCell):
    """Tensor Factorized Long short-term memory unit (LSTM) recurrent network cell.

    """
    def __init__(self, num_units, **kwargs):
        super(TensorBasicLSTMCell, self).__init__(num_units)
#         self._inp_modes = kwargs['inp_modes']
#         self._out_modes = kwargs['out_modes']
        self._mat_ranks = kwargs['mat_ranks']
            
    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(1, 2, state)     
        
            i = linear_tt([inputs, h], self._num_units, self._mat_ranks, bias =True, scope = "i")  
            j = linear_tt([inputs, h], self._num_units, self._mat_ranks, bias =True, scope = "j")   
            f = linear_tt([inputs, h], self._num_units, self._mat_ranks, bias =True, scope = "f")   
            o = linear_tt([inputs, h], self._num_units, self._mat_ranks, bias =True, scope = "o")   
        
#             concat = _linear([inputs, h], 4 * self._num_units, True)
#             # i = input_gate, j = new_input, f = forget_gate, o = output_gate
#             i , j, f, o = array_ops.split(1, 4, concat)

            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat(1, [new_c, new_h])
            return new_h, new_state

def linear_tt(args, output_size,  mat_ranks, bias, bias_start=0.0, scope=None):
    """wrapper for factorization layer"""
    # args = [x, h] solve y = Wx + Uh + b
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]
    dtype = [a.dtype for a in args][0]

    #sin = array_ops.concat(1, args)  batch_size* (x_dim + h_dim)
    with vs.variable_scope(scope or "Linear"):
#         matrix_x = vs.get_variable("Matrix_x", [args[0].get_shape().as_list()[1], output_size])
#         matrix_h = vs.get_variable("Matrix_h", [args[1].get_shape().as_list()[1], output_size])
#         res_x = math_ops.matmul(args[0], matrix_x) #tensornet.layers.tt(args[0], inp_modes['x'], out_modes['x'], mat_ranks['x'])
#         res_h = math_ops.matmul(args[1], matrix_h)#tensornet.layers.tt(args[1], inp_modes['h'], out_modes['h'], mat_ranks['h'])
#         res = res_x +  res_h #batch_size*out_size
        res_x = tensornet.layers.mf_rnn(args[0], [shapes[0][1]], [output_size], mat_ranks['x'], scope ="x")
        res_h = tensornet.layers.mf_rnn(args[1], [shapes[1][1]], [output_size], mat_ranks['h'], scope ="h")
        res = res_x +  res_h
        if not bias:
            return res
        bias_term = vs.get_variable("Bias", [output_size],dtype=dtype,initializer=init_ops.constant_initializer(
                bias_start, dtype=dtype))
      
    return res + bias_term

