# CS5242_AI_DL_Project
Video Captioning of ILVSRC2016-VID dataset

## Sync to SoC Cluster

rsync -art ./ nus_soc_xgpe1:~/CS5242_AI_DL_Project.git


## Train

- Run the bash script on a GPU enabled system

```
conda activate cs5242_proj
bash ./run_train.sh
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
rsync nus_soc_xgpe1:/home/stuproj/cs5242b4/CS5242_AI_DL_Project.git/model_run_data_<name>/model_<index>.pth  ~/Downloads/
```
