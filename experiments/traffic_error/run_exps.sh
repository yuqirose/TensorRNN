#!/bin/bash

use_error=--use_error_prop
d=/cs/ml/datasets/stephan/tensorcompress/traffic_9sensors.pkl

# chaotic_ts_mat.pkl  chaotic_ts.pkl  lorenz_series_mat.pkl  lorenz_series.pkl  traffic_9sensors.pkl  ushcn_CA.pkl



python seq_train.py --data_path=$d $use_error
python seq_train_lstm.py --data_path=$d $use_error
python seq_train_matrix.py --data_path=$d $use_error
python seq_train_tensor.py --data_path=$d $use_error
