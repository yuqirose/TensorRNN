
class TrainConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1e-2
  max_grad_norm = 10
  hidden_size = 16 # dim of h
  num_layers = 2
  burn_in_steps = 5
  num_steps = 20
  num_test_steps = 20
  horizon = 1
  num_lags = 3 # num prev hiddens
  num_orders = 2 # tensor prod order
  rank_vals= [2]
  num_freq = 2
  training_steps = 3000
  keep_prob = 1.0 # dropout
  sample_prob = 0.0 # sample ground true
  batch_size = 50
  use_error_prop = True
