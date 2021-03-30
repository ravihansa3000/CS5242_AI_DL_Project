import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderCNN(nn.Module):
	def __init__(self, output_feature_dims=500):
		"""Load the pretrained VGG-11 and replace top fc layer."""
		super(EncoderCNN, self).__init__()
		self.vgg11 = models.vgg11(pretrained=True).to(device)
		for param in self.vgg11.parameters():
			param.requires_grad = False
		self.vgg11.classifier = nn.Linear(25088, output_feature_dims)

	def forward(self, images):
		"""Extract feature vectors from input images."""
		images = images.to(device)
		return self.vgg11(images)
