import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderCNN(nn.Module):
	def __init__(self, output_feature_dims=500):
		"""Load the pretrained ResNet-50 and replace top fc layer."""
		super(EncoderCNN, self).__init__()
		self.resnet = models.resnet50(pretrained=True).to(device)
		# no need to train parameters
		for param in self.resnet.parameters():
			param.requires_grad = False
		self.resnet.fc = nn.Linear(2048, output_feature_dims)

	def forward(self, images):
		"""Extract feature vectors from input images."""
		images = images.to(device)
		return self.resnet(images)
