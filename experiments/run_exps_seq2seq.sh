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

#hidden_size=128
burn_in_steps=5 # just for naming purposes
num_steps=20
#num_test_steps=100


for exp in lorenz #climate traffic 
do
    for hidden_size in 16 32 64 
    do
        for learning_rate in 1e-1 1e-2 1e-3 1e-4 1e-5
        do
            save_path=/var/tmp/tensorRNN/log/$exp/$start_time/trnn_${hidden_size}_${learning_rate}/
            echo $save_path
            mkdir -p $save_path
            python train_seq2seq.py --save_path=$save_path --hidden_size=$hidden_size --learning_rate=$learning_rate   
	done
    done
done

cp $(pwd)/run_exps.sh ${save_path}run_exps.sh
