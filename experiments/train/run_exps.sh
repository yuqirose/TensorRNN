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

use_error=True #--use_error_prop

#data_path=../../../data/lorenz_series.pkl
#chaotic_ts_mat.pkl  chaotic_ts.pkl  lorenz_series_mat.pkl  lorenz_series.pkl  traffic_9sensors.pkl  ushcn_CA.pkl

hidden_size=64
burn_in_steps=5 # just for naming purposes
learning_rate=0.005

for exp in logistic lorenz  
do	

data_path=/home/roseyu/data/tensorRNN/${exp}.pkl

num_test_steps= 5
# for num_test_steps in 10 15 20 25 30 35 40 45 
# do
# base_dir=/tmp/tensorRNN/log/$exp/$start_time/ts_$num_test_steps/
base_dir=/tmp/tensorRNN/log/$exp/$start_time/
echo $base_dir

save_path=$base_dir/basic_rnn/
python seq_train_rnn.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_test_steps=$num_steps --learning_rate=$learning_rate --use_error_prop=$use_error

save_path=$base_dir/basic_lstm/
python seq_train_lstm.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_test_steps=$num_steps --learning_rate=$learning_rate --use_error_prop=$use_error

save_path=$base_dir/matrix_rnn/
python seq_train_matrix.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_test_steps=$num_steps --learning_rate=$learning_rate --use_error_prop=$use_error

#save_path=$base_dir/tensor_rnn/
#python seq_train_tensor.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_steps=$num_steps --use_error_prop=$use_error

save_path=$base_dir/tensor_rnn_einsum/
python seq_train_tensor_einsum.py --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_test_steps=$num_steps --learning_rate=$learning_rate --use_error_prop=$use_error
#done
done
