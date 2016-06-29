
# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np
def tucker_dense(inputs, inp_modes, out_modes, mat_ranks, 
                 init=2.0, scope="tt", use_biases=True, init_params=None):
    ''' Tucker model fully connected layer.
    
    Y = WX + b, where tensor(W) follows a Tucker(R_1,...R_D) model
    construct the graph and variables

    Args:
        inputs: input tensor, float - [batch_size, prod(inp_modes)]
        inp_modes: input tensor modes, column mapping
        out_modes: output tensor modes, row mapping
        mat_ranks: tucker ranks
        init: lambda function (shape) used for weights initialization
        scope: layer scope name, string
        use_biases: biases using flag, bool
    Return:
        {out: output tensor, float - [batch_size, prod(out_modes)]}
    '''
    with tf.name_scope(scope):
        dim = inp_modes.size
        # allocate memory for parameters: [{U},S]
        mat_ps = np.cumsum( np.concatenate( ([0],inp_modes * out_modes * mat_ranks, np.prod(mat_ranks)))

        mat_size = np.prod(inp_modes)   
        
        if type(init) == float:
            for i in range(dim):
                n_in = mat_ranks[i] * inp_modes[i]
                mat_proj = tf.truncated_normal([mat_ps[i + 1] - mat_ps[i]],
                                               0.0,
                                               init / n_in,
                                               tf.float32)
                if (i == 0):
                    mat = mat_proj
                else:
                    mat = tf.concat(0, [mat, mat_proj])
            tucker_core = tf.truncated_normal([mat_ps[i + 1] - mat_ps[i]],
                                               0.0,
                                               init / n_in,
                                               tf.float32)
            mat = tf.concat(0, [mat, tucker_core])

        else:
            init_params['inp_modes'] = inp_modes
            init_params['out_modes'] = out_modes
            init_params['ranks'] = mat_ranks 
            mat = init(init_params)
                    
        mat = tf.Variable(mat, name="weights")  #optimize over the entire weight matrix - TBD
        out = tf.reshape(inputs, [-1, np.prod(inp_modes)])
        out = tf.transpose(out, [1, 0])
        # implement n-mode product 
        # http://www.mathworks.com/matlabcentral/fileexchange/24268-n-mode-tensor-matrix-product/content/nmodeproduct.m
        for i in range(dim):
            out = tf.reshape(out, [inp_modes[i], -1])
                      
            mat_proj = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]]) # projection matrix
            mat_proj = tf.reshape(mat_proj, [  mat_ranks[i]*out_modes[i], inp_modes[i]])
            mat_proj = tf.transpose(mat_proj, [1, 0])
                           
            out = tf.matmul(mat_proj, out)
            out = tf.reshape(out, [out_modes[i], -1])
            out = tf.transpose(out, [1, 0])
                           
        tucker_core = tf.slice(mat, [mat_ps[i], [mat_ps[i+1] - mat_ps[i]]])
        mat_core = tf.reshape(tucker_core, [np.prod(inp_modes), -1])
        mat_core = tf.transpose(mat_core, [1, 0])

        out = tf.matmul(mat_core,out)
                        
        if use_biases:
            biases = tf.Variable(tf.zeros([np.prod(out_modes)]), name="biases")
            out = tf.add(tf.reshape(out, [-1, np.prod(out_modes)]), biases, name="out")
        else:
            out = tf.reshape(out, [-1, np.prod(out_modes)], name="out")
    return out   
    

