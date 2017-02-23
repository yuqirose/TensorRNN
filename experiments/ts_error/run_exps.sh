#!/bin/bash

use_error=--use_error_prop
d=/cs/ml/datasets/stephan/tensorcompress/chaotic_ts.pkl

python seq_train.py --data_path=$d $use_error
python seq_train_lstm.py --data_path=$d $use_error
python seq_train_matrix.py --data_path=$d $use_error
python seq_train_tensor.py --data_path=$d $use_error
