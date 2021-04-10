#!/bin/bash

stdout_file=./model_run_data/eval.out
printf "Running eval.py \n"
nohup python3 ./eval.py --gpu=0 --batch_size 60 $* > $stdout_file 2>&1 &

printf "stdout: $stdout_file \n"
