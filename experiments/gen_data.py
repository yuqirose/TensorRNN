import math
import numpy as np

n_range = 4 #bandwith
n_times = 100
n_dims = int(1e3)
np.random.seed(2)
x = np.empty((n_dims, n_times), 'int64')
start = np.random.randint(-4*n_range, 4*n_range, n_dims).reshape(n_dims, 1)

x[:] = start + np.array(range(n_times)) 
data = np.sin(x / 1.0 / n_range).astype('float64')
data = np.expand_dims(data, axis=2)
np.save('./data.npy', data)

