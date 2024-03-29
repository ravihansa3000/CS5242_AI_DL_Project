# CS5242_AI_DL_Project
Video Captioning of ILVSRC2016-VID dataset

## Train

- Run the bash script on a GPU enabled system

```
conda activate cs5242_proj
bash ./run_train.sh
```

## Test

```
conda activate cs5242_proj
bash ./run_test.sh --data_split train --trained_model ./model_run_data/model_<index>.pth
```

- Run with options

```
conda activate cs5242_proj
bash ./run_train.sh --batch_size 25 --learning_rate 3e-5
```


## Setup conda environment

```
# for CUDA 10.1 + pytorch 1.7.1
conda env create --file environment_pyt171_cuda101.yml --name cs5242_proj
```


## Sync to SoC Cluster

```
rsync -art ./ nus_soc_xgpe1:~/CS5242_AI_DL_Project.git
```

## Sync to NSCC

```
rsync -art ./ nus_nscc_stu:~/Documents/CS5242_AI_DL_Project.git

rsync -art ./ nus_nscc_proj:~/Akila/CS5242_AI_DL_Project.git
```

## Download model to local

```
rsync nus_soc_xgpe1:~/CS5242_AI_DL_Project.git/model_run_data/model_<index>.pth  ~/Downloads/trained_model.pth

ln -sf ~/Downloads/trained_model.pth ./model_run_data/trained_model.pth
```

## Download csv output to local

```
rsync nus_soc_xgpe1:~/CS5242_AI_DL_Project.git/model_run_data/*.csv  ./model_run_data/
```




## Build Singularity image

```
#build
sudo singularity build ./pytorch-1.7.1.simg ./nscc/Singularity.pytorch-1.7.1

# test pytorch
singularity exec --nv ./pytorch-1.7.1.simg python3 -c "import torch;print('pytorch version: ' + torch.__version__)"
```

## Run Docker image on NSCC

```
nvidia-docker -u $UID:$GID -v ~/Documents/CS5242_AI_DL_Project.git:/app/CS5242_AI_DL_Project.git \
    --rm -i --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 run nvcr.io/nvidia/pytorch:20.12-py3 /bin/sh
```
## Reproducing Kaggle submission result
Ensure the entire `cs-5242-project-nus-2021-semester2.zip` is extracted and placed in `./data/`. The path to the test set must be `data/test/test`.
Download the [I3D pretrained model](https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt) and place it in `./data/` as a `.pth` file.
Ensure that `./model_run_data/` is created. Edit `run_test.sh` to add the path to the trained model and execute `./run_test.sh`
```
#!/bin/bash

stdout_file=./model_run_data/test.out
printf "Running test.py \n"
nohup python3 ./test.py --gpu=0 --trained_model=<path_to_trained_model> --batch_size 60 $* > $stdout_file 2>&1 &

printf "stdout: $stdout_file \n"

```
Check `model_run_data/test.out` for output logs. The following csv file would contain the results:
```
model_run_data/preds_sub.csv
```

