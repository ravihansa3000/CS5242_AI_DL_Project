import math

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

from utils import init_hidden

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
	def __init__(self, dim_vid=500, dim_opf=500, dim_hidden=1000, rnn_cell=nn.LSTM,
	             n_layers=1, rnn_dropout_p=0.5):
		"""Load the pretrained CNNs and replace top fc layer."""
		super(Encoder, self).__init__()

		self.dim_vid = dim_vid
		self.dim_opf = dim_opf
		self.dim_hidden = dim_hidden
		self.rnn_cell = rnn_cell

		# CNN for encoding original images
		enc_cnn_img = models.vgg19(pretrained=True).to(device)
		for param in enc_cnn_img.parameters():
			param.requires_grad = False

		enc_cnn_img.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(p=0.8),
			nn.Linear(4096, self.dim_vid),
			nn.ReLU(True),
			nn.Dropout(p=0.7),
		).to(device)
		self.enc_cnn_img = enc_cnn_img

		# CNN for encoding optical flow images
		enc_cnn_opf = models.vgg16(pretrained=True).to(device)
		for param in enc_cnn_opf.parameters():
			param.requires_grad = False

		enc_cnn_opf.classifier = self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(p=0.8),
			nn.Linear(4096, self.dim_opf),
			nn.ReLU(True),
			nn.Dropout(p=0.7),
		).to(device)
		self.enc_cnn_opf = enc_cnn_opf

		# encoder RNN
		self.rnn = rnn_cell(
			self.dim_vid + self.dim_opf,
			self.dim_hidden,
			n_layers,
			batch_first=True,
			dropout=rnn_dropout_p,
			bidirectional=False
		).to(device)

		for name, param in self.rnn.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			elif 'weight' in name:
				nn.init.xavier_normal_(param)

		self._init_modules()

	def forward(self, x_vid: torch.Tensor, x_opf: torch.Tensor):
		"""Convert a batch of videos into embeddings and feed them into the encoder RNN"""
		assert (x_vid.shape[0] == x_opf.shape[0])
		assert (x_vid.shape[1] == x_opf.shape[1])
		batch_size = x_vid.shape[0]
		n_frames = x_vid.shape[1]
		vid_encoded = []
		opf_encoded = []
		for i in range(batch_size):
			vid_encoded.append(self.enc_cnn_img(x_vid[i]))
			opf_encoded.append(self.enc_cnn_opf(x_opf[i]))

		vid_feats = torch.stack(vid_encoded, dim=0)  # batch_size, n_frames, dim_vid
		opf_feats = torch.stack(opf_encoded, dim=0)  # batch_size, n_frames, dim_opf

		# concat original image features and optical flow features; (batch_size, n_frames, dim_vid + dim_opf)
		combined_feats = torch.cat((vid_feats[:, :n_frames, :], opf_feats[:, :n_frames, :]), dim=2)

		# add padding frames for sequence of words (3 elements)
		padding_frames = Variable(
			torch.empty(batch_size, 3, self.dim_vid + self.dim_opf, dtype=vid_feats.dtype)
		).zero_().to(device)
		rnn_input = torch.cat((combined_feats, padding_frames), dim=1)

		state = None
		output, _ = self.rnn(rnn_input, state)  # batch_size, n_frames + 3, dim_hidden
		return output

	def _init_modules(self):
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
