#!/bin/sh
python seq_train.py&
python seq_train_lstm.py&
python seq_train_matrix.py&
python seq_train_tensor.py&
