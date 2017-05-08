import numpy as np
import cPickle as pickle



def gen_logistic_series(x0, num_steps):
    alpha = 4.0
    num_steps = num_steps
    x = np.ndarray((num_steps,1) )
    x[0]  = x0
    f = lambda  x ,t: alpha* x[t] * (1.0 - x[t]) 
    for t in range(num_steps-1):
        x[t+1] = f(x,t)
    logistic_series = x
    return logistic_series


def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def gen_lorenz_series(x0, y0, z0, num_steps, num_freq):
    dt = 0.01
    stepCnt = num_steps

    # Need one more for the initial values
    xs = np.empty((stepCnt,))
    ys = np.empty((stepCnt,))
    zs = np.empty((stepCnt,))

    # Setting initial values
    #xs[0], ys[0], zs[0] = (0., 1., 1.05)
    xs[0] = x0
    ys[0] = y0
    zs[0] = z0

    xss = np.empty((stepCnt//num_freq,))
    yss = np.empty((stepCnt//num_freq,))
    zss = np.empty((stepCnt//num_freq,))
    # Stepping through "time".
    j = 0
    for i in range(stepCnt-1):
        # Derivatives of the X, Y, Z state
        if i%num_freq ==0:
            xss[j] = xs[i]
            yss[j] = ys[i]
            zss[j] = zs[i]
            j += 1
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    #save the sequence for training
    lorenz_series = np.transpose(np.vstack((xss,yss,zss)))
    return lorenz_series

def gen_lorenz_dataset(file_name="lorenz.pkl"):
    #define initial range
    num_samples = int(1e5)
    num_freq = int(5)
    num_steps = int(1e2)*num_freq
    
    init_range = np.random.uniform(-20,20,(num_samples,3))
   
    lorenz_series_mat = np.ndarray((num_samples, num_steps//num_freq, 3))

    for i in range(num_samples):
        x0,y0,z0 = init_range[i,:]
        series = gen_lorenz_series(x0,y0,z0, num_steps, num_freq )
        lorenz_series_mat[i,:,:] = series
                
    pickle.dump(lorenz_series_mat, open(file_name,"wb")) 


def gen_logistic_dataset(file_name = "logistic.pkl"):
    """generate set of chaotic time series with randomly selected initial"""
    num_series = 50
    num_steps = int(1e2)

    init_range = np.random.uniform(0.1, 1.0, num_series)
    x_mat = np.ndarray((num_series, num_steps, 1 ))# num_time x num_series , a collection of time series with different initial values
    for init, i in zip(init_range, range(num_series)):
        series=gen_logistic_series(init,num_steps)
        x_mat[i, :, :]  = series
    pickle.dump(x_mat,open(file_name,"wb"))

def main():
    data_path = "/Users/roseyu/Documents/Python/"#"/home/roseyu/data/tensorRNN/"

    file_name = data_path+"logistic.pkl"
    gen_logistic_dataset(file_name)
    print("Finish generating logistic")

    file_name = data_path+"lorenz.pkl"
    gen_lorenz_dataset(file_name)

    print("Finish generating lorenz")
if __name__== "__main__":
    main()