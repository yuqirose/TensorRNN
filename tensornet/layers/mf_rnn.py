
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf


class MfRNNCell(tf.nn.rnn_cell.RNNCell):
    '''
    Basic MF_RNN_cell
    matrix factorized version of the most basic RNN cell.
    '''
    def __init__(self, num_units, input_size=None):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units
    


    def __call__(self, inputs, state, inp_modes, out_modes, mat_ranks, scope="mf_rnn"):
        """Most basic RNN: output = new_state = tanh(input * W_x  + state * W_h  + B).
           Factorize W_x = U_x * V_x, W_h = U_h * V_h
           W_x (input_size, output_size), W_h (state_size, output_size)
        """
        mat_ps = np.cumsum( np.concatenate( ([0], [self._input_size*mat_ranks], [mat_ranks*self._output_size],
                                             [self._state_size*mat_ranks],[mat_ranks*self._output_size])))
        mat_size = mat_ps[-1]   
        
        with vs.variable_scope(scope or type(self).__name__):  # "MfRNNCell"
            #output = tanh(linear([inputs, state], self._num_units, True))
            mat = tf.get_variable("weights", mat_size)  #optimize over the entire weight matrix 
            output = tf.reshape(inputs, [-1, np.prod(inp_modes)]) 
             
            U_x = tf.slice(mat, [mat_ps[0]], [mat_ps[1] - mat_ps[0]]) # matrix factorization components
            U_x = tf.reshape(U_x, [-1,mat_ranks])

            V_x = tf.slice(mat, [mat_ps[1]], [mat_ps[2] - mat_ps[1]]) # matrix factorization components
            V_x = tf.reshape(V_x, [mat_ranks,-1])

            W_x = tf.matmul(U_x, V_x)
            #hidden states
            U_h = tf.slice(mat, [mat_ps[0]], [mat_ps[1] - mat_ps[0]]) # matrix factorization components
            U_h = tf.reshape(U_h, [-1,mat_ranks])

            V_h = tf.slice(mat, [mat_ps[1]], [mat_ps[2] - mat_ps[1]]) # matrix factorization components
            V_h = tf.reshape(V_x, [mat_ranks,-1])

            W_h = tf.matmul(U_h, V_h)
            
            output = tf.matmul(output, W_x) +  tf.matmul(states, W_h)
        
        return output, output


