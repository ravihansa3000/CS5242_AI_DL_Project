#!/bin/bash

stdout_file=./model_run_data/nohup.out
printf "Running train.py \n"
nohup python3 ./train.py --gpu=0 $* > $stdout_file 2>&1 &

printf "stdout: $stdout_file \n"
