
# coding: utf-8

# In[ ]:
import numpy as np
import tensorflow as tf
import sys
from math_utils import n_mode_product

def cp_dense(inputs, inp_modes, out_modes, mat_ranks, 
                 init=2.0, scope="cp_dense", use_biases=True, init_params=None):
    ''' Tucker model fully connected layer.
    
    Y = WX + b, where tensor(W) follows a CP decomposition (R) model
    construct the graph and variables

    Args:
        inputs: input tensor, float - [batch_size, prod(inp_modes)]
        inp_modes: input tensor modes, column mapping
        out_modes: output tensor modes, row mapping
        mat_ranks: CP rank 
        init: lambda function (shape) used for weights initialization
        scope: layer scope name, string
        use_biases: biases using flag, bool
    Return:
        {out: output tensor, float - [batch_size, prod(out_modes)]}
    '''
    with tf.name_scope(scope):
        dim = inp_modes.size
        # allocate memory for parameters: [{U},Lambda] mat_ranks components
        mat_ps = np.cumsum( np.concatenate( ([0], np.tile(inp_modes * out_modes, mat_ranks) , [mat_ranks])))
        mat_size = mat_ps[-1]   
        
        if type(init) == float:        
            for r in range(mat_ranks):
                 for d in range(dim):
                    i = r*dim + d
                    n_in = inp_modes[d]*out_modes[d]
                    mat_proj = tf.truncated_normal([mat_ps[i + 1] - mat_ps[i]],
                                                   0.0,
                                                   init / n_in,
                                                   tf.float32)  
                    
                    if i == 0:
                        mat = mat_proj
                    else:
                        mat = tf.concat(0, [mat, mat_proj])
     
            cp_core = tf.truncated_normal([mat_ps[-1] - mat_ps[-2]],
                                              0.0,
                                              init / n_in,
                                              tf.float32)
            mat = tf.concat(0, [mat, cp_core])

        else:
            init_params['inp_modes'] = inp_modes
            init_params['out_modes'] = out_modes
            init_params['ranks'] = mat_ranks 
            mat = init(init_params) 
                    
        mat = tf.Variable(mat, name="weights")  #optimize over the entire weight matrix - TBD
        out = tf.reshape(inputs, [-1, np.prod(inp_modes)])

        cp_core = tf.slice(mat, [mat_ps[-2]], [mat_ps[-1]- mat_ps[-2]])        
        # iteratively compute cp  outer-product
        for r in range(mat_ranks):
            for d in range(dim):
                i = r*dim + d
                mat_proj = tf.slice(mat, [mat_ps[i]], [mat_ps[i + 1] - mat_ps[i]]) # rank-1 components
                mat_proj = tf.reshape(mat_proj, [-1,1])
                if d == 0:
                    weights_r = mat_proj
                else: 
                    weights_r = tf.mul(mat_proj,weights_r)
                weights_r = tf.reshape(weights_r, [1, -1])  
  
       
         
            if r ==0:
                weights = cp_core[r] * weights_r
            else:   
                weights = weights + cp_core[r] * weights_r
        weights = tf.reshape(weights, [np.prod(inp_modes),-1])
        '''
        init = tf.initialize_all_variables()
        sess = tf.InteractiveSession()
        sess.run(init)
        print(sess.run(tf.shape(weights_r), feed_dict = {out: np.ones((28*28,np.prod(inp_modes))) }))
        sess.close()
        '''
        out = tf.matmul(out, weights)
                        
        if use_biases:
            biases = tf.Variable(tf.zeros([np.prod(out_modes)]), name="biases")
            out = tf.add(tf.reshape(out, [-1, np.prod(out_modes)]), biases, name="out")
        else:
            out = tf.reshape(out, [-1, np.prod(out_modes)], name="out")
    return out   
    

