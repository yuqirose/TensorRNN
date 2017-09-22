
class TrainConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1e-2
  max_grad_norm = 10
  hidden_size = 32 # dim of h
  num_layers = 1
  burn_in_steps = 5
  num_steps = 10
  num_test_steps = 10
  horizon = 1
  num_lags = 3 # num prev hiddens
  num_orders = 2 # tensor prod order
  rank_vals= [4]
  num_freq = 2
  training_steps = 10000
  keep_prob = 1.0 # dropout
  sample_prob = 0.0 # sample predictions
  batch_size = 32
  use_error_prop = True
