# Courtesy of https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks

from dataset import VRDataset
from encoder import EncoderCNN
import torch
import torchvision
from torchvision import transforms, datasets, models, utils
import torch.nn as nn
from torch.utils.data import DataLoader
from helper_utils import HelperUtils 
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_feature_dims = 2048
output_feature_dims = 500
input_size = 224
train_dataset_path = "data/train/train"
train_annotation_path = "data/training_annotation.json"

FILE = './save/ENCODER_CNN_STATE_DICT.pth'

data_transformations = {
	'train': transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
}

vrdataset = VRDataset(img_root=train_dataset_path, transform=data_transformations['train'])
dataloader = DataLoader(vrdataset, batch_size=1, shuffle=False, num_workers=0)

with open (train_annotation_path) as f:
	training_annotation = json.load(f)

if os.path.isfile(FILE):
	encoder.load_state_dict(torch.load(FILE))
	encoder.eval()
else:
	for i, sample in enumerate(dataloader):
		for j, frame in enumerate(sample['frames']):
			features = encoder(sample['frames'][j])

	# save CNN encoder state dict
	torch.save(encoder.state_dict(), FILE)encoder = EncoderCNN(input_feature_dims, output_feature_dims)encoder = EncoderCNN(input_feature_dims, output_feature_dims)