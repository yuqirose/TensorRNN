
# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np

def tt_cnn(inp, inp_modes, out_modes, filter_modes, mat_ranks,strides=1, init=2.0, scope="tt_cnn", use_biases=True, init_params=None):
    """ tt-layer ('old' tt-linear layer, tt-matrix by full tensor product)
    Args:
        inp: input tensor, float - [batch_size, prod(inp_modes)]
        inp_modes: input tensor modes
        out_modes: output tensor modes
        mat_ranks: tt-matrix ranks
        init: lambda function (shape) used for weights initialization
        scope: layer scope name, string
        use_biases: biases using flag, bool
    Returns:
        out: output tensor, float - [batch_size, prod(out_modes)]
    """
    with tf.name_scope(scope):
        #dim = inp_modes.size: TBD-bijection of input&output
        dim = len(filter_modes)
        mat_ps = np.cumsum(np.concatenate(([0], mat_ranks[:-1] * filter_modes * mat_ranks[1:])))

        mat_size = mat_ps[-1]
        if type(init) == float:
            for i in range(dim):
                n_in = mat_ranks[i] * filter_modes[i]
                mat_core = tf.truncated_normal([mat_ps[i + 1] - mat_ps[i]],
                                               0.0,
                                               init / n_in,
                                               tf.float32)
                if (i == 0):
                    mat = mat_core
                else:
                    mat = tf.concat(0, [mat, mat_core])
        else:
            init_params['inp_modes'] = inp_modes
            init_params['out_modes'] = out_modes
            init_params['ranks'] = mat_ranks
            mat = init(init_params)
        mat = tf.Variable(mat, name="weights")

        inp = tf.reshape(inp, np.insert(inp_modes,0,-1))
        W = np.eye(mat_ranks[0],dtype=np.float32) # filter_height, filter_width, in_channels, out_channels
        for i in range(dim):
            mat_core = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]])
            mat_core = tf.reshape(mat_core, [mat_ranks[i], filter_modes[i]*mat_ranks[i + 1]])
            W = tf.matmul(W, mat_core)
            W = tf.reshape(W, [mat_ranks[i+1], -1])
            W = tf.transpose(W, [1, 0])
        W = tf.reshape(W, filter_modes)
        # Change FC to Conv operatoin
        out = tf.nn.conv2d(inp, W, strides=[1, strides, strides, 1], padding='SAME')   
        if use_biases:
            biases = tf.Variable(tf.zeros([out_modes]), name="biases")
            out = tf.add(tf.reshape(out, np.insert(out_modes,0,-1)), biases, name="out")
        else:
            out = tf.reshape(out, np.insert(out_modes,0,-1), name="out")
    return out

