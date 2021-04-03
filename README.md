# CS5242_AI_DL_Project
Video Captioning of ILVSRC2016-VID dataset

## Sync to SoC Cluster

rsync -art ./ nus_soc_xgpe1:~/CS5242_AI_DL_Project.git

## Sync to NSCC

rsync -art ./ nus_nscc_stu:~/Documents/CS5242_AI_DL_Project.git

rsync -art ./ nus_nscc_proj:~/Akila/CS5242_AI_DL_Project.git

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
conda env create --file environment.yml --name cs5242_proj
```

## Download model to local

```
rsync nus_soc_xgpe1:/home/stuproj/cs5242b4/CS5242_AI_DL_Project.git/model_run_data/model_<index>.pth  ~/Downloads/trained_model.pth

ln -sf ~/Downloads/trained_model.pth ./model_run_data/trained_model.pth
```

## Download csv output to local

```
rsync nus_soc_xgpe1:/home/stuproj/cs5242b4/CS5242_AI_DL_Project.git/model_run_data/*.csv  ./model_run_data/
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