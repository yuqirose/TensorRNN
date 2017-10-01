
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from trnn import *

def LSTM(enc_inps, dec_inps, is_training, config):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Define a lstm cell with tensorflow
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(config.hidden_size,forget_bias=1.0, reuse=None)
    #if is_training and config.keep_prob < 1:
    #    cell = tf.contrib.rnn.DropoutWrapper(
    #      lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(config.num_layers)])

    # Get encoder output
    with tf.variable_scope("Encoder", reuse=None):
        enc_outs, enc_states = rnn_with_feed_prev(cell, enc_inps, True, config)
    # Get decoder output
    with tf.variable_scope("Decoder", reuse=None):
        config.burn_in_steps = 0
        dec_outs, dec_states = rnn_with_feed_prev(cell, dec_inps, is_training, config, enc_states)
    
    return dec_outs

def PLSTM(enc_inps, dec_inps, is_training, config):
    def plstm_cell():
        return tf.contrib.rnn.PhasedLSTMCell(config.hidden_size)
    cell = plstm_cell()
    if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=config.keep_prob)        
    cell = tf.contrib.rnn.MultiRNNCell(
        [cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        enc_outs, enc_states = rnn_with_feed_prev(cell, enc_inps, True, config)

    with tf.variable_scope("Decoder", reuse=None):
        config.burn_in_steps = 0
        dec_outs, dec_states = rnn_with_feed_prev(cell, dec_inps, is_training, config, enc_states) 
    return dec_outs


def RNN(enc_inps, dec_inps,is_training, config):
    def rnn_cell():
        return tf.contrib.rnn.BasicRNNCell(config.hidden_size)
    if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
          rnn_cell(), output_keep_prob=config.keep_prob)        
    cell = tf.contrib.rnn.MultiRNNCell(
        [rnn_cell() for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        enc_outs, enc_states = rnn_with_feed_prev(cell, enc_inps, True, config)

    with tf.variable_scope("Decoder", reuse=None):
        config.burn_in_steps = 0
        dec_outs, dec_states = rnn_with_feed_prev(cell, dec_inps, is_training, config, enc_states) 
    return dec_outs    


def MRNN(enc_inps, dec_inps, is_training, config):
    def mrnn_cell():
        return MatrixRNNCell(config.hidden_size,config.num_lags)
    cell = mrnn_cell()
    if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=config.keep_prob)        
    cell = tf.contrib.rnn.MultiRNNCell(
        [cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell, enc_inps, True, config)

    with tf.variable_scope("Decoder", reuse=None):
        config.burn_in_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell, dec_inps, is_training, config, enc_states) 
    return dec_outs    


def HORNN(enc_inps, dec_inps, is_training, config):
    def hornn_cell():
        return HighOrderRNNCell(config.hidden_size,config.num_lags, config.num_orders)
    cell = hornn_cell()
    if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=config.keep_prob)        
    cell = tf.contrib.rnn.MultiRNNCell(
        [cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell, enc_inps, True, config)

    with tf.variable_scope("Decoder", reuse=None):
        config.burn_in_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell, dec_inps, is_training, config, enc_states) 
    return dec_outs    

def HOLSTM(enc_inps, dec_inps, is_training, config):
    def holstm_cell():
        return HighOrderLSTMCell(config.hidden_size,config.num_lags, config.num_orders)
    cell = holstm_cell()
    if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=config.keep_prob)        
    cell = tf.contrib.rnn.MultiRNNCell(
        [cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell, enc_inps, True, config)

    with tf.variable_scope("Decoder", reuse=None):
        config.burn_in_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell, dec_inps, is_training, config, enc_states) 
    return dec_outs    

def TRNN(enc_inps, dec_inps, is_training, config):
    def trnn_cell():
        return EinsumTensorRNNCell(config.hidden_size, config.num_lags, config.rank_vals)
    cell= trnn_cell() 
    if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=config.keep_prob)        
    cell = tf.contrib.rnn.MultiRNNCell(
        [cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell, enc_inps, True, config)

    with tf.variable_scope("Decoder", reuse=None):
        config.burn_in_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell, dec_inps, is_training, config, enc_states) 
    return dec_outs    


def MTRNN(enc_inps, dec_inps, is_training, config):
    def mtrnn_cell():
        return MTRNNCell(config.hidden_size, config.num_lags, config.num_freq, config.rank_vals)
    cell= mtrnn_cell()
    if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=config.keep_prob)        
    cell = tf.contrib.rnn.MultiRNNCell(
        [cell for _ in range(config.num_layers)])
    with tf.variable_scope("Encoder", reuse=None):
        enc_outs, enc_states = tensor_rnn_with_feed_prev(cell, enc_inps, True, config)

    with tf.variable_scope("Decoder", reuse=None):
        config.burn_in_steps = 0
        dec_outs, dec_states = tensor_rnn_with_feed_prev(cell, dec_inps, is_training, config, enc_states) 
    return dec_outs    

