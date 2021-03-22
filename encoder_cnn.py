import torch
from torchvision import transforms, datasets, models, utils


class EncoderCNN(torch.nn.Module):
	def __init__(self, output_feature_dims=500):
		super(EncoderCNN, self).__init__()

		self.model = models.resnet50(pretrained=True)

		for param in self.model.parameters():
			param.requires_grad = False

		self.model.fc = torch.nn.Linear(2048, output_feature_dims)

		self.relu = torch.nn.ReLU()

	def forward(self, images):
		return self.relu(self.model(images))
