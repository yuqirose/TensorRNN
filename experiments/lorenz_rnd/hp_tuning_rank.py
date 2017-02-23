import subprocess
import os
import time
import numpy as np
import itertools
from six.moves import shlex_quote

def gen_command(rank,  i):
    cmd = ['python','seq_train_tensor.py', '--rank_val', rank,#'--learning_rate', lr , '--hidden_size', hz,
            '--save_path','../log/lorenz_rnd_exp/tt_rnn_rank-'+str(i)+'/' ]
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
    #lr_range = np.linspace(1e-3, 1.0, 20).tolist()
    #hz_range = np.linspace(4, 10, 6, 2, dtype =int).tolist() # start, stop, num ,base
    #num_sample = 10
    rank_range = np.arange(1,20, 2) 
    search_space = rank_range
    search_sz = len(search_space)
    #search_space= rand_grid_search((lr_range, hz_range), num_sample)

    processes = set()
    max_processes = 36
    
    for rank, i in zip(search_space, range(search_sz)):#range(num_sample)):
        command = gen_command(rank, i)        
        print( " ".join(shlex_quote(str(v)) for v in command))
        processes.add(subprocess.Popen(command))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([
            p for p in processes if p.poll() is not None])

if __name__=="__main__":
    parallel_run()
