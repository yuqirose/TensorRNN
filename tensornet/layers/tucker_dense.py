
# coding: utf-8

# In[ ]:
import numpy as np
import tensorflow as tf
import sys
from math_utils import n_mode_product

def tucker_dense(inputs, inp_modes, out_modes, mat_ranks, 
                 init=2.0, scope="tucker_dense", use_biases=True, init_params=None):
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
        mat_ps = np.cumsum( np.concatenate( ([0], inp_modes * out_modes * mat_ranks, [np.prod(mat_ranks)])))
        mat_size = mat_ps[-1]   
        
        if type(init) == float:
 
            for i in range(dim):
                n_in = mat_ranks[i] * inp_modes[i]
                mat_proj = tf.truncated_normal([mat_ps[i + 1] - mat_ps[i]],
                                               0.0,
                                               init / n_in,
                                               tf.float32)  
                if i == 0:
                    mat = mat_proj
                else:
                    mat = tf.concat(0, [mat, mat_proj])
            i = i + 1
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
        # efficient implement n-mode product:
        tucker_core = tf.slice(mat, [mat_ps[-2]], [mat_ps[-1]- mat_ps[-2]])

        tucker_core = tf.reshape(tucker_core, mat_ranks)
        
        # iteratively compute tucker product
        weights = tucker_core
        weights_shape = mat_ranks
        for i in range(dim):
            mat_proj = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]]) # projection matrix
            mat_proj = tf.reshape(mat_proj, [inp_modes[i]*out_modes[i],mat_ranks[i]])
            weights = n_mode_product(weights,mat_proj,i, weights_shape, [inp_modes[i]*out_modes[i],mat_ranks[i]] )
            weights_shape[i] = inp_modes[i]*out_modes[i]
            '''
            try:            
                out = tf.matmul(mat_proj, out)
            except ValueError:
                init = tf.initialize_all_variables()
                sess = tf.InteractiveSession()
                sess.run(init)
                print(sess.run(tf.shape(mat_proj), feed_dict = {out: np.ones((inp_modes[i], 28*28)) }))

                sess.close()
            '''
        weights = tf.reshape(weights, [np.prod(inp_modes), -1])
        out = tf.matmul(out, weights)
                        
        if use_biases:
            biases = tf.Variable(tf.zeros([np.prod(out_modes)]), name="biases")
            out = tf.add(tf.reshape(out, [-1, np.prod(out_modes)]), biases, name="out")
        else:
            out = tf.reshape(out, [-1, np.prod(out_modes)], name="out")
    return out   
    

