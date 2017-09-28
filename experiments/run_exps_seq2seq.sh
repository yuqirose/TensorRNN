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

use_error=False #--use_error_prop

#data_path=../../../data/lorenz_series.pkl
#chaotic_ts_mat.pkl  chaotic_ts.pkl  lorenz_series_mat.pkl  lorenz_series.pkl  traffic_9sensors.pkl  ushcn_CA.pkl

#hidden_size=32
rank=2
learning_rate=1e-2
burn_in_steps=5 # just for naming purposes
#num_test_steps=100

for exp in lorenz #climate traffic 
do
    for model in LSTM HORNN TRNN 
    do
	for hidden_size in 8 16 32 64 128
        do
            save_path=/var/tmp/tensorRNN/log/$exp/$start_time/$model/$hidden_size/
            echo $save_path
            mkdir -p $save_path
            python train_seq2seq.py --model=$model --rank=$rank --save_path=$save_path --hidden_size=$hidden_size --learning_rate=$learning_rate   
        done
    done
done

cp $(pwd)/run_exps_seq2seq.sh ${save_path}run_exps_seq2seq.sh
