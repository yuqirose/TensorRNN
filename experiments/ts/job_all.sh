#!/bin/sh

now=$(date +"%I_%M")

#DATA_PATH="../../../traffic_9sensors.pkl"
DATA_PATH="../../../ushcn_CA_0.pkl"

#SAVE_PATH="../log/traffic_exp_${now}/"
SAVE_PATH="../log/climate_exp_${now}/"

python seq_train.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_rnn/"&
python seq_train_lstm.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_lstm/"&
python seq_train_matrix.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}matrix_rnn/"&
#python seq_train_tensor.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}tensor_rnn/"& 

DATA_PATH="../../../traffic_9sensors.pkl"
SAVE_PATH="../log/traffic_exp_${now}/"

python seq_train.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_rnn/"&
python seq_train_lstm.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_lstm/"&
python seq_train_matrix.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}matrix_rnn/"&
#python seq_train_tensor.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}tensor_rnn/"& 
DATA_PATH="../../../lorenz_series.pkl"
SAVE_PATH="../log/lorenz_exp_${now}/"

python seq_train.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_rnn/"&
python seq_train_lstm.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_lstm/"&
python seq_train_matrix.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}matrix_rnn/"&
#python seq_train_tensor.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}tensor_rnn/"& 
DATA_PATH="../../../chaotic_ts.pkl"
SAVE_PATH="../log/ts_exp_${now}/"

python seq_train.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_rnn/"&
python seq_train_lstm.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_lstm/"&
python seq_train_matrix.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}matrix_rnn/"&

DATA_PATH="../../../lorenz_series_mat.pkl"
SAVE_PATH="../log/lorenz_rnd_exp_${now}/"

python seq_train.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_rnn/"&
python seq_train_lstm.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_lstm/"&
python seq_train_matrix.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}matrix_rnn/"&

DATA_PATH="../../../chaotic_ts_mat.pkl"
SAVE_PATH="../log/ts_rnd_exp_${now}/"

python seq_train.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_rnn/"&
python seq_train_lstm.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_lstm/"&
python seq_train_matrix.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}matrix_rnn/"&


