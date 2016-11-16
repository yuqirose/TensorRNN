
# coding: utf-8

# In[112]:

import tensorflow as tf
import numpy as np
def n_mode_product(T_a, M_u, n, T_a_shape, M_u_shape):
    '''
    Perform tensor matrix n-model-product 
    http://www.mathworks.com/matlabcentral/fileexchange/24268-n-mode-tensor-matrix-product/content/nmodeproduct.m
    Args:
        T_a: N-mode tensor 
        M_u: matrix 
    Return:
        T_b: N-mode product
    '''          
    num_dim = T_a_shape.size
    perm_order = np.roll(np.arange(num_dim),n)
    M_a = tf.reshape(tf.transpose(T_a, perm=perm_order), [T_a_shape[n],-1])
    
    T_b_shape = T_a_shape
    T_b_shape[n] = M_u_shape[0]
    M_b = tf.matmul(M_u, M_a)
    perm_order = np.roll(np.arange(num_dim),-n)
    T_b = tf.transpose(tf.reshape(M_b, T_b_shape ), perm=perm_order)
    return T_b
    


# In[113]:

# unit test of n_mode_product
def test_n_mode_product():
    '''
    Unit test of n_mode_product
    '''
    input_sz = np.array([4,5,6])
    rank_sz = np.array([2,2,2])
    num_dim = (input_sz).ndim
    
    tucker_core = tf.random_normal(rank_sz)

    i = 2
    proj_mat= tf.random_normal([input_sz[i],rank_sz[i]])
    out = n_mode_product(tucker_core, proj_mat, i, rank_sz, np.array([input_sz[i],rank_sz[i]]))
    # start a computation graph
    init = tf.initialize_all_variables()

    # Launch the graph.
    sess = tf.Session()
    sess.run(init)
    with tf.Session():
        print out.eval()
        print 

if __name__=='__main__':
    test_n_mode_product()

