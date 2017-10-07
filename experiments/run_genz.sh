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
#start_time=09-29-17-22-36-36
use_error_prop=True #--use_error_prop

#data_path=../../../data/lorenz_series.pkl
#chaotic_ts_mat.pkl  chaotic_ts.pkl  lorenz_series_mat.pkl  lorenz_series.pkl  traffic_9sensors.pkl  ushcn_CA.pkl

#hidden_size=32
rank=4
learning_rate=1e-3
burn_in_steps=5 # just for naming purposes

for exp in f1 f2 f3 f4 f5 f6
do
base_path=/var/tmp/tensorRNN/log/$exp/$start_time
data_path=/home/roseyu/data/tensorRNN/${exp}.npy
    for model in TLSTM LSTM
    do
	for hidden_size in 32 
        do
	    for learning_rate in $learning_rate   
	    do
            save_path=${base_path}/genz-$exp/$model/
            echo $save_path
            mkdir -p $save_path
            python train_seq2seq.py --model=$model --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --learning_rate=$learning_rate --rank=$rank 
            done
        done
    done
done

cp $(pwd)/run_genz.sh ${base_path}/run_genz.sh
