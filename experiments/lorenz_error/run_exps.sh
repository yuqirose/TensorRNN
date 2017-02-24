#!/bin/bash

timestamp() {
  date +"%T"
}
datestamp() {
  date +"%D"
}

t=$(timestamp)
t="$(echo ${t} | tr ':' '-')"

d=$(datestamp)
d=$(echo ${d} | tr '/' '-')

start_time="$d-$t"

use_error=--use_error_prop
d=/cs/ml/datasets/stephan/tensorcompress/lorenz_series.pkl
exp=lorenz_error_exp


base_dir=/tmp/tensorcompress/log/$exp/$start_time

save_path=$base_dir/basic_rnn
python seq_train.py --data_path=$d --save_path=$save_path $use_error

save_path=$base_dir/basic_lstm
python seq_train_lstm.py --data_path=$d --save_path=$save_path $use_error

save_path=$base_dir/matrix_rnn
python seq_train_matrix.py --data_path=$d --save_path=$save_path $use_error

save_path=$base_dir/tt_rnn
python seq_train_tensor.py --data_path=$d --save_path=$save_path $use_error
