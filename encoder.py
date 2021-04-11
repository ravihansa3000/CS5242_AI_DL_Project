import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

from utils import init_hidden

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):

	def __init__(self, dim_vid=500, dim_hidden=500, rnn_cell=nn.LSTM, rnn_dropout_p=0):
		"""Load the pretrained resnet-50 and replace top fc layer."""

		super(Encoder, self).__init__()

		self.dim_vid = dim_vid
		self.dim_hidden = dim_hidden
		self.rnn_cell = rnn_cell

		self.resnet50 = models.resnet50(pretrained=True).to(device)
		self.resnet50.eval()
		for param in self.resnet50.parameters():
			param.requires_grad = False
		self.resnet50.fc = nn.Linear(2048, self.dim_vid)

		# encoder RNN
		self.rnn = rnn_cell(self.dim_vid, self.dim_hidden, num_layers=1,
		                    batch_first=True, dropout=rnn_dropout_p, bidirectional=False).to(device)

		for name, param in self.rnn.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			elif 'weight' in name:
				nn.init.xavier_normal_(param)

	def train(self, mode=True):
		super().train(mode)
		self.resnet50.eval()
		self.resnet50.fc.train(mode)

	def forward(self, x):
		"""Convert a batch of videos into embeddings and feed them into the encoder RNN"""

		batch_size = x.shape[0]
		vid_imgs_encoded = []
		for i in range(batch_size):
			vid_imgs_encoded.append(self.resnet50(x[i]))

		vid_feats = torch.stack(vid_imgs_encoded, dim=0)  # batch_size, 30, dim_vid

		state = init_hidden(batch_size, 1, self.dim_hidden)
		return self.rnn(vid_feats, state)  # batch_size, 30, dim_hidden
