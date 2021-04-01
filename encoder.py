import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderCNN(nn.Module):
	def __init__(self, output_feature_dims=500):
		"""Load the pretrained ResNet-50 and replace top fc layer."""
		super(EncoderCNN, self).__init__()

		self.output_feature_dims = output_feature_dims
		self.resnet = models.resnet50(pretrained=True).to(device)

		# no need to train parameters
		for param in self.resnet.parameters():
			param.requires_grad = False

		self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_feature_dims)

	def forward(self, images):
		"""
		Extract feature vectors from input images.
		:param images: Tensor containing video features of shape
			(batch_size, n_frames, cnn_input_c, cnn_input_h, cnn_input_w)
			n_frames is the number of video frames
		"""
		images = images.to(device)
		return self.resnet(images)


class EncoderR152CNN(nn.Module):
	def __init__(self, output_feature_dims=500):
		"""Load the pretrained ResNet-152 and replace top fc layer."""
		super(EncoderR152CNN, self).__init__()
		resnet = models.resnet152(pretrained=True)
		modules = list(resnet.children())[:-1]  # delete the last fc layer.
		self.resnet = nn.Sequential(*modules).to(device)
		for param in self.resnet.parameters():
			param.requires_grad = False

		self.linear = nn.Linear(resnet.fc.in_features, output_feature_dims).to(device)
		self.bn = nn.BatchNorm1d(output_feature_dims, momentum=0.01).to(device)
		self.init_weights()

	def init_weights(self):
		"""Initialize the weights."""
		self.linear.weight.data.normal_(0.0, 0.02)
		self.linear.bias.data.fill_(0)

	def forward(self, images):
		"""Extract the image feature vectors."""
		features = self.resnet(images)
		features = Variable(features.data)
		features = features.view(features.size(0), -1)
		features = self.bn(self.linear(features))
		return features


class EncoderVGG11CNN(nn.Module):
	def __init__(self, output_feature_dims=500):
		"""Load the pretrained VGG-11 and replace top fc layer."""
		super(EncoderVGG11CNN, self).__init__()
		self.vgg11 = models.vgg11(pretrained=True).to(device)
		for param in self.vgg11.parameters():
			param.requires_grad = False

		self.vgg11.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, output_feature_dims)
		)
		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, images):
		"""Extract feature vectors from input images."""
		images = images.to(device)
		return self.vgg11(images)
