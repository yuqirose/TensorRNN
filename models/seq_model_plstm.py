import tensorflow as tf
import numpy as np
from models.high_order_rnn import plstm_with_feed_prev
from tensorflow.contrib.rnn import PhasedLSTMCell, MultiRNNCell, DropoutWrapper
import inspect
class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_, use_error_prop=False):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        hidden_size = config.hidden_size
        input_size = input_.input_size

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        def rnn_cell():
            if 'reuse' in inspect.getargspec(
                    PhasedLSTMCell.__init__).args:
                rnn_cell = PhasedLSTMCell(hidden_size, ratio_on=0.05, reuse=tf.get_variable_scope().reuse)
            else:
                rnn_cell = PhasedLSTMCell(hidden_size, ratio_on=0.05)
            return rnn_cell
        attn_cell = rnn_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return DropoutWrapper(attn_cell(), output_keep_prob=config.keep_prob)

        cell = MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, dtype= tf.float32)

        with tf.device("/cpu:0"):
            inputs = input_.input_data
            """
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
            """

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = [tf.squeeze(input_step, [1])
        #           for input_step in tf.split(1, num_steps, inputs)]
        # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)



        # outputs = []
        # state = self._initial_state
        # with tf.variable_scope("RNN"):
        #     for time_step in range(num_steps):
        #         if time_step > 0: tf.get_variable_scope().reuse_variables()
        #         (cell_output, state) = cell(inputs[:, time_step, :], state)
        #         outputs.append(cell_output)

        # output = tf.reshape(tf.concat(1, outputs), [-1, size])
        # softmax_w = tf.get_variable(
        #     "softmax_w", [size, vocab_size], dtype= tf.float32)
        # softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        # logits = tf.matmul(output, softmax_w) + softmax_b

        print("num_steps:", num_steps)

        feed_prev = not is_training if use_error_prop else False

        logits, states, weights = rnn_with_feed_prev(cell, inputs,
            num_steps, hidden_size, self._initial_state, input_size, feed_prev=feed_prev, burn_in_steps=config.burn_in_steps)
        state = states[-1]


        self._predict = tf.reshape(logits, [batch_size, num_steps, -1])
        self._cost = cost = tf.reduce_mean(tf.squared_difference(
            tf.reshape(logits, [-1,input_size]), tf.reshape(input_.targets, [batch_size*num_steps,-1]) ))
        self._final_state = states
        # calculate number of parameters 
        num_params = hidden_size * (hidden_size + 1)*4
        print("number of parameters : %d " % (num_params))

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost
    @property
    def predict(self):
        return self._predict

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op



