#!/bin/bash

stdout_file=./model_run_data/train.out
printf "Running train.py \n"
nohup python3 ./train.py --gpu=0 --end_epoch=150 --learning_rate=0.001 --learning_rate_decay_every=200 --learning_rate_decay_rate=0.65 $* > $stdout_file 2>&1 &

printf "stdout: $stdout_file \n"
