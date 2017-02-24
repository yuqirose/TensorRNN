#!/bin/sh

now=$(date +"%I_%M")

DATA_PATH="../../../chaotic_ts.pkl"
SAVE_PATH="../log/ts_exp_${now}/"
BATCH_SIZE=64
#DATA_PATH="../../../traffic_9sensors.pkl"
#DATA_PATH="../../../ushcn_CA_0.pkl"

#SAVE_PATH="../log/traffic_exp_${now}/"
#SAVE_PATH="../log/climate_exp_${now}/"

python seq_train.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_rnn/" --batch_size=$BATCH_SIZE&
python seq_train_lstm.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}basic_lstm/" --batch_size=$BATCH_SIZE&
python seq_train_matrix.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}matrix_rnn/" --batch_size=$BATCH_SIZE&
python seq_train_tensor.py --data_path=$DATA_PATH --save_path="${SAVE_PATH}tensor_rnn/" --batch_size=$BATCH_SIZE& 
