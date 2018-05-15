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
use_error_prop=True #--use_error_prop

data_path=/home/qiyu/TensorRNN/datasets/mnist.h5

batch_size=80
hidden_size=16
rank=2
learning_rate=1e-3
burn_in_steps=5 # 1 hour burn in

for exp in mnist
do
base_path=/home/qiyu/TensorRNN/experiments/log/$exp/$start_time
    for model in TLSTM 
    do
	for learning_rate in 1e-3 
        do
	    for test_steps in 10 
	    do
            save_path=${base_path}/$model/ts-$test_steps/
            echo $save_path
            mkdir -p $save_path
            python train_mnist.py --model=$model --data_path=$data_path --save_path=$save_path --burn_in_steps=$burn_in_steps --test_steps=$test_steps --hidden_size=$hidden_size --batch_size=$batch_size --learning_rate=$learning_rate --rank=$rank 
            done
        done
    done
done
cp $(pwd)/run_mnist.sh ${base_path}/run_mnist.sh
