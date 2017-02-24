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
use_error_path=/feed_prev
data_path=/cs/ml/datasets/stephan/tensorcompress/traffic_9sensors.pkl
# chaotic_ts_mat.pkl  chaotic_ts.pkl  lorenz_series_mat.pkl  lorenz_series.pkl  traffic_9sensors.pkl  ushcn_CA.pkl

exp=traffic_error_exp

num_steps=50
hidden_size=128

base_dir=/tmp/tensorcompress/log/$exp/rollout_$num_steps$use_error_path/$start_time
echo $base_dir

save_path=$base_dir/basic_rnn
python seq_train.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error

save_path=$base_dir/basic_lstm
python seq_train_lstm.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error

save_path=$base_dir/matrix_rnn
python seq_train_matrix.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error

# save_path=$base_dir/tt_rnn
# python seq_train_tensor.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error

save_path=$base_dir/einsum_tt_rnn
python seq_train_tensor_einsum.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error
