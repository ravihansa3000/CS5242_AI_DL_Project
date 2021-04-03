#!/bin/bash

stdout_file=./model_run_data/test.out
printf "Running test.py \n"
nohup python3 ./test.py --gpu=0 --batch_size 40 $* > $stdout_file 2>&1 &

printf "stdout: $stdout_file \n"
