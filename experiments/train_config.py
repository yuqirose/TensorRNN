
class TrainConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1e-3
  max_grad_norm = 10
  hidden_size = 64 # dim of h
  num_layers = 2
  burn_in_steps = 5
  num_steps = 10 # stops gradients after num_steps
  num_test_steps = 10 # long term forecasting steps
  horizon = 1
  num_lags = 2 # num prev hiddens
  num_orders = 2 # tensor prod order
  rank_vals= [2]
  num_freq = 2
  training_steps = 5000
  keep_prob = 1.0 # dropout
  batch_size = 128
  use_error_prop = True




