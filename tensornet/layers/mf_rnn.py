
# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs

def mf_rnn(inputs, inp_modes, out_modes, mat_ranks, scope):
    inp_dim = np.prod(inp_modes)
    out_dim = np.prod(out_modes)
    with vs.variable_scope(scope+"TensorTrain"):
        U = vs.get_variable("U", [inp_dim, mat_ranks])
        V = vs.get_variable("V", [mat_ranks, out_dim])
        weights = tf.matmul(U, V)
        out =  math_ops.matmul(inputs, weights)
    return out

