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

use_error= #--use_error_prop
use_error_path=/no_feed_prev
data_path=/cs/ml/datasets/stephan/tensorcompress/lorenz_series_mat.pkl
# chaotic_ts_mat.pkl  chaotic_ts.pkl  lorenz_series_mat.pkl  lorenz_series.pkl  traffic_9sensors.pkl  ushcn_CA.pkl

exp=lorenz_mat_error_exp


_num_steps=(12 50)
hidden_size=128
burn_in_steps=5 # just for naming purposes

# With error-prop

for num_steps in "${_num_steps[@]}"
do

# use_error= #--use_error_prop
# use_error_path=/no_feed_prev
# base_dir=/tmp/tensorcompress/log/$start_time/$exp/rollout_${num_steps}_burnin_$burn_in_steps$use_error_path
# echo $base_dir

# save_path=$base_dir/basic_rnn
# python seq_train.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error

# save_path=$base_dir/basic_lstm
# python seq_train_lstm.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error

# save_path=$base_dir/matrix_rnn
# python seq_train_matrix.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error

# save_path=$base_dir/tt_rnn
# python seq_train_tensor.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error

# save_path=$base_dir/einsum_tt_rnn
# python seq_train_tensor_einsum.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error


# Without error-prop

use_error=--use_error_prop
use_error_path=/feed_prev
base_dir=/tmp/tensorcompress/log/$start_time/$exp/rollout_${num_steps}_burnin_$burn_in_steps$use_error_path
echo $base_dir

save_path=$base_dir/basic_rnn
python seq_train.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error

save_path=$base_dir/basic_lstm
python seq_train_lstm.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error

save_path=$base_dir/matrix_rnn
python seq_train_matrix.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error

save_path=$base_dir/tt_rnn
python seq_train_tensor.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error

save_path=$base_dir/einsum_tt_rnn
python seq_train_tensor_einsum.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps $use_error

done