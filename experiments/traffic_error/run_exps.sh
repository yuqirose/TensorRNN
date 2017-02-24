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

use_error= #-use_error_prop
data_path=/cs/ml/datasets/stephan/tensorcompress/traffic_9sensors.pkl
# chaotic_ts_mat.pkl  chaotic_ts.pkl  lorenz_series_mat.pkl  lorenz_series.pkl  traffic_9sensors.pkl  ushcn_CA.pkl

exp=traffic_error_exp

base_dir=/tmp/tensorcompress/log/$exp_${use_error//--/""}/$start_time

echo $base_dir

hidden_size=128

save_path=$base_dir/basic_rnn
python seq_train.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size $use_error

save_path=$base_dir/basic_lstm
python seq_train_lstm.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size $use_error

save_path=$base_dir/matrix_rnn
python seq_train_matrix.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size $use_error

# save_path=$base_dir/tt_rnn
# python seq_train_tensor.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size $use_error

save_path=$base_dir/einsum_tt_rnn
python seq_train_tensor_einsum.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size $use_error
