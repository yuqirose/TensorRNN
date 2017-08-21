import numpy as np
import torch
from torch.autograd import Variable


def normalize_columns(arr):
    rows, cols = arr.shape
    for col in range(cols):
        arr_col = arr[:,col]
        arr[:,col] = (arr_col - arr_col.min() )/ (arr_col.max()- arr_col.min())
    return arr

def seq_raw_data(data_path="logistic.npy", val_size = 0.1, test_size = 0.1):
    """ this approach is fundamentally flawed if time series is non-statinary"""
    print("loading sequence data ...")
    data = np.load(data_path)
    if (np.ndim(data)==1):
        data = np.expand_dims(data, axis=1)
    """normalize the data"""
    print("normalize to (0-1)")
    data = normalize_columns(data)

    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data[:ntest]) * (1 - val_size)))
    train_data, valid_data, test_data = data[:nval, ], data[nval:ntest, ], data[ntest:,]
    return train_data, valid_data, test_data

def seq_to_batch(data, bsz, is_cuda=False):
    """sequence to mini-batches, generate batch_size x bptt x dim
    
    Args:
        data (TYPE): Description
        bsz (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    data = torch.Tensor(data).float()
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, nbatch, -1).contiguous()
    data = torch.transpose(data, 0, 1)
    if is_cuda:
        data = data.cuda()
    return data

def get_batch(source, i, bptt, evaluation=False):
    seq_len = min(bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len,:], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len,:])
    return data, target
