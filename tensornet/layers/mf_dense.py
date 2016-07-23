
# coding: utf-8

# In[ ]:
import numpy as np
import tensorflow as tf
import sys
from math_utils import n_mode_product

def mf_dense(inputs, inp_modes, out_modes, mat_ranks, 
                 init=2.0, scope="mf_dense", use_biases=True, init_params=None):
    ''' Matrix factorization model fully connected layer.
    
    Y = WX + b, where tensor(W) follows a matrix decompostion (R) model
    construct the graph and variables

    Args:
        inputs: input tensor, float - [batch_size, prod(inp_modes)]
        inp_modes: input tensor modes, column mapping
        out_modes: output tensor modes, row mapping
        mat_ranks: matrix rank
        init: lambda function (shape) used for weights initialization
        scope: layer scope name, string
        use_biases: biases using flag, bool
    Return:
        {out: output tensor, float - [batch_size, prod(out_modes)]}
    '''
    with tf.name_scope(scope):
        dim = inp_modes.size
        # allocate memory for parameters: [U,V] mat_ranks is a float value
        mat_ps = np.cumsum( np.concatenate( ([0], [np.prod(inp_modes)*mat_ranks], [mat_ranks*np.prod(out_modes)])))
        mat_size = mat_ps[-1]   
        
        if type(init) == float:
            n_in = np.prod(inp_modes)
            U = tf.truncated_normal([mat_ps[1] - mat_ps[0]],
                                                          0.0,
                                                          init / n_in,
                                                          tf.float32)  
            n_in = np.prod(out_modes)
            V = tf.truncated_normal([mat_ps[2] - mat_ps[1]],
                                                          0.0,
                                                          init / n_in,
                                                          tf.float32)  
 
            mat = tf.concat(0, [U, V])

        else:
            init_params['inp_modes'] = inp_modes
            init_params['out_modes'] = out_modes
            init_params['ranks'] = mat_ranks 
            mat = init(init_params) 
                    
        mat = tf.Variable(mat, name="weights")  #optimize over the entire weight matrix - TBD
        out = tf.reshape(inputs, [-1, np.prod(inp_modes)]) 
            
        U = tf.slice(mat, [mat_ps[0]], [mat_ps[1] - mat_ps[0]]) # matrix factorization components
        U = tf.reshape(U, [-1,mat_ranks])

        V = tf.slice(mat, [mat_ps[1]], [mat_ps[2] - mat_ps[1]]) # matrix factorization components

        V = tf.reshape(V, [mat_ranks,-1])
        
        weights = tf.matmul(U, V)
        out = tf.matmul(out, weights)
                        
        if use_biases:
            biases = tf.Variable(tf.zeros([np.prod(out_modes)]), name="biases")
            out = tf.add(tf.reshape(out, [-1, np.prod(out_modes)]), biases, name="out")
        else:
            out = tf.reshape(out, [-1, np.prod(out_modes)], name="out")
    return out   
    

