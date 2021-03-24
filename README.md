# CS5242_AI_DL_Project
Video Captioning of ILVSRC2016-VID dataset

## Sync to SoC Cluster

rsync -art ./ nus_soc_xgpe1:~/CS5242_AI_DL_Project.git


## Train

Run the bash script on a GPU enabled system

```
conda activate cs5242_proj
bash ./run_train.sh
```

## Setup conda environment

```
conda env create --file environment.yml --name cs5242_proj
```