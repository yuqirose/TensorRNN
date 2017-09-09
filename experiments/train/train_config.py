
class TrainConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1e-2
  max_grad_norm = 10
  hidden_size = 128 # dim of h
  num_layers = 1
  burn_in_steps = 5
  num_steps = 10 # stops gradients after num_steps
  num_test_steps = 10
  horizon = 1
  num_lags = 4 # num prev hiddens
  num_orders = 2 # tensor prod order
  rank_vals= [1]
  num_freq = 2
  max_epoch = 1 # keep lr fixed
  max_max_epoch = 2 # decaying lr
  keep_prob = 1.0 # dropout
  lr_decay = 0.99
  batch_size = 128
  use_error_prop = False




