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

#start_time="$d-$t"
start_time=10-11-17-22-19-29
use_error_prop=True #--use_error_prop

#data_path=../../../data/lorenz_series.pkl
#chaotic_ts_mat.pkl  chaotic_ts.pkl  lorenz_series_mat.pkl  lorenz_series.pkl  traffic_9sensors.pkl  ushcn_CA.pkl

#hidden_size=32
rank=2
learning_rate=1e-3
burn_in_steps=5 # just for naming purposes
test_steps=95

for exp in f2 
do
  for test_steps in 20 40 60 80; do 
base_path=/tmp/tensorRNN/log/genz/$exp/$start_time
data_path=/home/qiyu/data/${exp}.npy
    for model in TLSTM MLSTM LSTM 
    do
	for hidden_size in 16 
        do
	    for learning_rate in $learning_rate   
	    do
            save_path=${base_path}/$model/hz-$hidden_size/ts-$test_steps/
            echo $save_path
            mkdir -p $save_path
            python train_seq2seq.py --model=$model --data_path=$data_path --save_path=$save_path --burn_in_steps=$burn_in_steps --test_steps=$test_steps --hidden_size=$hidden_size --learning_rate=$learning_rate --rank=$rank 
            done
        done
    done
  done
done

cp $(pwd)/run_genz.sh ${base_path}/run_genz.sh
