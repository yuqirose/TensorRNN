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
start_time=10-19-17-23-29-41
use_error_prop=True #--use_error_prop

data_path=/home/qiyu/data/traffic_s2s.npy
#chaotic_ts_mat.pkl  chaotic_ts.pkl  lorenz_series_mat.pkl  lorenz_series.pkl  traffic_9sensors.pkl  ushcn_CA.pkl

hidden_size=16
rank=4
learning_rate=1e-3
burn_in_steps=12 # 1 hour burn in

for exp in traffic
do
for run_num in 2 
do
#data_path=/home/qiyu/data/${exp}_s2s.npy
base_path=/tmp/tensorRNN/log/$exp/$start_time
    for model in TLSTM 
    do
	for learning_rate in 1e-3 
        do
	    for test_steps in 19 
	    do
            test_steps_traffic=$(($test_steps * 12))
            save_path=${base_path}/$model/run-$run_num/ts-$test_steps/
            echo $save_path
            mkdir -p $save_path
            python train_seq2seq.py --model=$model --data_path=$data_path --save_path=$save_path --burn_in_steps=$burn_in_steps --test_steps=$test_steps_traffic --hidden_size=$hidden_size --learning_rate=$learning_rate --rank=$rank 
            done
        done
    done
done
done
cp $(pwd)/run_traffic.sh ${base_path}/run_traffic.sh
