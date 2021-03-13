
# Courtesy of https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks

import torch, torchvision.models as models
from torchvision import transforms, datasets
import torch.nn as nn
from dataset import VRDataset
from torch.utils.data import DataLoader

output_feature_dims = 500
input_size = 224
train_dataset_path = "data/train/train"

data_transformations = {
	'train': transforms.Compose([
		transforms.RandomResizedCrop(input_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize(input_size),
		transforms.CenterCrop(input_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
}

def test_model(model):
	model.eval()
	imagenet_path = "/media/aravind95/ubuntu_a/datasets/imagenet/val"
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	val_dataset = datasets.ImageFolder(imagenet_path, transform=data_transformations['val'])
	
	val_loader = torch.utils.data.DataLoader(
			val_dataset, batch_size=10,
			shuffle=True)
	
	x, y = next(iter(val_loader))
	
	print ("Running validation...")
	predictions = model(x)
	print ("Top-1 accuracy on a batch of size 10:", torch.sum(torch.argmax(predictions, dim=1) == y) / y.shape[0])

model = models.resnet50(pretrained=True)

# Test on imagenet
test_model(model)

for param in model.parameters():
	param.requires_grad = False

model.fc = nn.Linear(2048, output_feature_dims)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(train_dataset_path, transform=data_transformations['train'])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=31, shuffle=False)

x, y = next(iter(train_loader))

print (y)


vrdataset = VRDataset(img_root='data/train/train', transform=data_transformations['train'])
dataloader = DataLoader(vrdataset, batch_size=1, shuffle=False, num_workers=0)

sample = next(iter(dataloader))
print (sample['video_id'], sample['frames'][0].shape)
