
class TrainConfig(object):
  """Tiny config, for testing."""
  burn_in_steps = 5
  init_scale = 0.1
  learning_rate = 1e-3
  max_grad_norm = 10
  num_layers = 1
  num_steps = 35 # stops gradients after num_steps
  horizon = 1
  num_lags = 4 # num prev hiddens
  num_orders = 2 # tensor prod order
  rank_vals= [1]
  num_freq = 2
  hidden_size = 64 # dim of h
  max_epoch = 20 # keep lr fixed
  max_max_epoch = 100 # decaying lr
  keep_prob = 1.0 # dropout
  lr_decay = 0.99
  batch_size = 5
  rand_init = False



class TrainConfig(object):
  """Tiny config, for testing."""
  burn_in_steps = 5
  init_scale = 0.1
  learning_rate = 1e-3
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35 # stops gradients after num_steps
  horizon = 1
  num_lags = 4 # num prev hiddens
  num_orders = 2 # tensor prod order
  rank_vals= [1]
  num_freq = 2
  hidden_size = 64 # dim of h
  max_epoch = 20 # keep lr fixed
  max_max_epoch = 100 # decaying lr
  keep_prob = 1.0 # dropout
  lr_decay = 0.99
  batch_size = 1
  rand_init = False
