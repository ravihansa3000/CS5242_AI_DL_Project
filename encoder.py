import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

from utils import init_hidden

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
	def __init__(self, dim_vid=500, dim_opf=500, dim_r3d=500, dim_hidden=1000, rnn_cell=nn.LSTM, n_layers=1,
				 rnn_dropout_p=0.5):
		"""Load the pretrained CNNs and replace top fc layer."""
		super(Encoder, self).__init__()

		self.dim_vid = dim_vid
		self.dim_opf = dim_opf
		self.dim_r3d = dim_r3d
		self.dim_hidden = dim_hidden
		self.rnn_cell = rnn_cell

		# CNN for encoding original images
		enc_cnn_vid = models.resnet152(pretrained=True).to(device)
		enc_cnn_vid.eval()
		for param in enc_cnn_vid.parameters():
			param.requires_grad = False

		enc_cnn_vid.fc = nn.Sequential(
			nn.Linear(2048, self.dim_vid),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
		).to(device)
		self.enc_cnn_vid = enc_cnn_vid

		# CNN for encoding optical flow images
		enc_cnn_opf = models.resnet50(pretrained=True).to(device)
		enc_cnn_opf.eval()
		for param in enc_cnn_opf.parameters():
			param.requires_grad = False

		enc_cnn_opf.fc = nn.Sequential(
			nn.Linear(2048, self.dim_opf),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
		).to(device)
		self.enc_cnn_opf = enc_cnn_opf

		# CNN for activity detection
		# enc_cnn_r3d = nn.Sequential(
		#     *list(models.video.r3d_18(pretrained=True).children())[:-1]
		# ).to(device)
		# enc_cnn_r3d.eval()
		# for param in enc_cnn_r3d.parameters():
		#     param.requires_grad = False

		# self.enc_cnn_r3d = enc_cnn_r3d
		# self.fc1_r3d = nn.Linear(512, self.dim_r3d).to(device)
		# torch.nn.init.xavier_uniform_(self.fc1_r3d.weight)
		# self.dropout_r3d = nn.Dropout2d(0.3).to(device)

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
			vid_encoded.append(self.enc_cnn_vid(x_vid[i]))
			opf_encoded.append(self.enc_cnn_opf(x_opf[i]))

		vid_feats = torch.stack(vid_encoded, dim=0)  # batch_size, n_frames, dim_vid
		opf_feats = torch.stack(opf_encoded, dim=0)  # batch_size, n_frames, dim_opf

		# concat original image features and optical flow features; (batch_size, n_frames, dim_vid + dim_opf)
		combined_feats = torch.cat((vid_feats[:, :n_frames, :], opf_feats[:, :n_frames, :]), dim=2)

		output_r3d = None
		# output_r3d = self.enc_cnn_r3d(x_vid.permute(0, 2, 1, 3, 4)).squeeze(4).squeeze(3).squeeze(2)
		# output_r3d = F.relu(self.fc1_r3d(output_r3d))
		# output_r3d = self.dropout_r3d(output_r3d)

		# add padding frames for sequence of words (3 elements)
		padding_frames = Variable(
			torch.empty(batch_size, 3, self.dim_vid + self.dim_opf, dtype=vid_feats.dtype)
		).zero_().to(device)
		rnn_input = torch.cat((combined_feats, padding_frames), dim=1)

		state = None
		output_rnn, _ = self.rnn(rnn_input, state)  # batch_size, n_frames + 3, dim_hidden
		return output_rnn, output_r3d

	def _init_modules(self):
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def train(self, mode=True):
		super().train(mode)
		# set eval for batch normalization layers in pretrained models
		self.enc_cnn_vid.eval()
		self.enc_cnn_vid.fc.train(mode)

		self.enc_cnn_opf.eval()
		self.enc_cnn_opf.fc.train(mode)

		# self.enc_cnn_r3d.eval()
