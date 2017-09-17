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


# hidden_size=128
burn_in_steps=5 # just for naming purposes
num_steps=20
#num_test_steps=100


for exp in lorenz #climate traffic 
do 
data_path=/home/roseyu/data/tensorRNN/${exp}.npy

    for hidden_size in 128
    do
        for learning_rate in 1e-2 
        do
            for num_test_steps in 20 40 60 80 100 
            do
            for model in MRNN TRNN LSTM PLSTM
            do
            save_path=/var/tmp/tensorRNN/log/$exp/$start_time/$model/st_$num_test_steps/
            echo $save_path
            mkdir -p $save_path
            python train.py --model=$model --data_path=$data_path --save_path=$save_path --hidden_size=$hidden_size --num_train_steps=$num_train_steps --num_test_steps=$num_test_steps --learning_rate=$learning_rate --use_error_prop=$use_error  
            done
	    done
	done
    done
done

cp $(pwd)/run_exps.sh ${save_path}run_exps.sh
