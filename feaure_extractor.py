# Courtesy of https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks

from dataset import VRDataset
from encoder_cnn import EncoderCNN
import torch
import torchvision
from torchvision import transforms, datasets, models, utils
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

output_feature_dims = 500
input_size = 224
train_dataset_path = "data/train/train"

data_transformations = {
	'train': transforms.Compose([
		transforms.Resize((input_size, input_size)),
		# transforms.CenterCrop(input_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize((input_size, input_size)),
		# transforms.CenterCrop(input_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
}

vrdataset = VRDataset(img_root=train_dataset_path, transform=data_transformations['train'])
dataloader = DataLoader(vrdataset, batch_size=1, shuffle=False, num_workers=0)

encoder = EncoderCNN()
for i, sample in enumerate(dataloader):
    encoded = encoder.forward(sample['frames'])
    print(encoded.shape)
