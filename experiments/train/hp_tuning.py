import subprocess
import os
import time
import numpy as np
import itertools
from six.moves import shlex_quote

def gen_command(lr, hz, i):
    cmd = ['python','seq_train_tensor_einsum.py','--learning_rate', lr , '--hidden_size', hz,
            '--save_path','../../../log/lorenz/tensor_rnn_einsum_'+str(i)+'/' ]
    cmd = [shlex_quote(str(v)) for v in cmd]
    return cmd

def rand_grid_search(array_list, num_sample):
    """Cartesian product of list and draw random samples """
    results= []
    for element in itertools.product(*array_list):
        results.append(element)
    print(len(results), type(results))
    sample_idx = np.random.choice(np.arange(len(results)), num_sample)
    print(sample_idx)
    samples = [results[i] for i in sample_idx.tolist()]
    return samples

def parallel_run():
    """execute the program in parallel for different parameter settings"""
    lr_range = np.logspace(-6, 0, num=20, base=10).tolist()
    hz_range = np.logspace(6, 8, num=3, base=2, dtype=int).tolist() # start, stop, num ,base
    num_sample = 10
    #horizon_range = np.arange(1,50, 2) 
    #search_space = horizon_range
    #search_sz = len(search_space)
    search_space= rand_grid_search((lr_range, hz_range), num_sample)

    processes = set()
    max_processes = 10
    
    for (lr, hz), i in zip(search_space, range(num_sample)):
        command = gen_command(lr, hz, i)        
        print( " ".join(shlex_quote(str(v)) for v in command))
        processes.add(subprocess.Popen(command))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([
            p for p in processes if p.poll() is not None])

if __name__=="__main__":
    parallel_run()
