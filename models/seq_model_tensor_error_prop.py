import tensorflow as tf
from tensorflow.python.ops.math_ops import sigmoid
from high_order_rnn import TensorRNNCell, tensor_rnn
class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    num_lags = config.num_lags
    rank_vals = config.rank_vals
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.


    initializer = tf.random_uniform_initializer(-1,1)
    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    rnn_cell = TensorRNNCell(size, num_lags, rank_vals)

    if is_training and config.keep_prob < 1:
      rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
        rnn_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * config.num_layers, state_is_tuple=True)
    initial_states = []
    for lag in range(num_lags):
      # initial_state: tuple of len num_layers, each element state_size(2 for lstm,c,h) x batch_size x input_size
      initial_state =  cell.zero_state(batch_size, dtype= tf.float32)
      initial_states.append(initial_state)
    self._initial_states = initial_states
    with tf.device("/cpu:0"):
      inputs = input_.input_data
      """
      embedding = tf.get_variable(
        "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
      """

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)


    prev = None
    outputs = []
    states = self._initial_states

    if not is_training:
      print("Creating model @ not training --> Feeding output back into input.")
    else:
      print("Creating model @ training --> input = ground truth each timestep.")

    def _hidden_to_data(h):
      softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype= tf.float32)
      softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
      logits = tf.matmul(h, softmax_w) + softmax_b
      return logits

    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):

        if time_step > 0:
          tf.get_variable_scope().reuse_variables()

        inp = inputs[:, time_step, :]

        if not is_training and prev is not None:
          inp = _hidden_to_data(prev)


        (cell_output, state) = cell(inp, states)

        if not is_training:
          prev = cell_output

        output = _hidden_to_data(cell_output)

        outputs.append(output)


    logits = tf.reshape(tf.concat(1, outputs), [-1, vocab_size])

    self._predict = logits

    beta = 0.0
    self._cost = cost = tf.reduce_mean(tf.squared_difference(
      logits, tf.reshape(input_.targets, [batch_size*num_steps,-1]) )
      + beta*tf.nn.l2_loss(output)
      + beta*tf.nn.l2_loss(softmax_w) + beta*tf.nn.l2_loss(softmax_b))
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                      config.max_grad_norm)
    # optimizer = tf.train.GradientDescentOptimizer(self._lr)
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
  def predict(self):
    return self._predict

  @property
  def initial_states(self):
    return self._initial_states

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op



