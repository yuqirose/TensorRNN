import numpy as np
import cPickle as pickle

class Config(object):
    num_time = int(1e5)
    num_dim = 1
    alpha = 4.0
    init_val = 0.1

def gen_ts_rnd_init(file_name = "chaotic_ts_mat.pkl"):
    """generate set of chaotic time series with randomly selected initial"""
    data_path= "../../../"
    config = Config()
    num_series = 50
    init_range = np.linspace(0.1, 1.0, num_series)
    x_mat = np.ndarray((config.num_time,1, num_series ))# num_time x num_series , a collection of time series with different initial values
    for init, i in zip(init_range, range(num_series)):
        config.init_val = init
        x = np.ndarray((config.num_time) )
        x[0]  = config.init_val 
        f = lambda  x ,t: config.alpha* x[t] * (1.0 - x[t]) 
        for t in range(config.num_time-1):
            x[t+1] = f(x,t)
        x_mat[:,0, i]  = x
    pickle.dump(x_mat,open(data_path+file_name,"wb"))

def gen_chaotic_ts(config,file_name="chaotic_ts.pkl"):
    data_path = "../../../"
    x = np.ndarray((config.num_time) )
    x[0]  = config.init_val 
    f = lambda  x ,t: config.alpha* x[t] * (1.0 - x[t]) 
    for t in range(config.num_time-1):
        x[t+1] = f(x,t)
    pickle.dump(x,open(data_path+file_name,"wb"))
    
def main():
    train_config = Config()
    #gen_chaotic_ts("train_ts.pkl")
    gen_ts_rnd_init()
if __name__== "__main__":
    main()
