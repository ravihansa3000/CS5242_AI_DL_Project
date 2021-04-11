import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.i3d import InceptionI3d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
	def __init__(self, dim_vid=500, dim_opf=500, dim_hidden=1000, rnn_cell=nn.LSTM, n_layers=1, rnn_dropout_p=0.5):
		"""Load the pretrained CNNs and replace top fc layer."""
		super(Encoder, self).__init__()

		self.dim_vid = dim_vid
		self.dim_opf = dim_opf
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
		enc_cnn_opf = InceptionI3d(
			num_classes=400,
			spatial_squeeze=True,
			final_endpoint='Logits',
			in_channels=3,
			dropout_keep_prob=0.6
		).to(device)
		assert os.path.isfile("data/rgb_imagenet.pth")
		checkpoint = torch.load("data/rgb_imagenet.pth")
		enc_cnn_opf.load_state_dict(checkpoint)

		enc_cnn_opf.eval()
		for param in enc_cnn_opf.parameters():
			param.requires_grad = False

		enc_cnn_opf.replace_logits(self.dim_opf, device=device)
		self.enc_cnn_opf = enc_cnn_opf

		# encoder RNN
		self.rnn = rnn_cell(
			self.dim_vid + self.dim_opf,
			self.dim_hidden,
			n_layers,
			batch_first=True,
			dropout=rnn_dropout_p,
			bidirectional=True
		).to(device)

		for name, param in self.rnn.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			elif 'weight' in name:
				nn.init.xavier_normal_(param)

	def forward(self, x_vid: torch.Tensor, x_opf: torch.Tensor):
		"""Convert a batch of videos into embeddings and feed them into the encoder RNN"""
		assert (x_vid.shape[0] == x_opf.shape[0])
		batch_size = x_vid.shape[0]
		n_frames = x_vid.shape[1]
		vid_encoded = []
		for i in range(batch_size):
			vid_encoded.append(self.enc_cnn_vid(x_vid[i]))

		vid_feats = torch.stack(vid_encoded, dim=0)  # batch_size, n_frames, dim_vid
		opf_feats = self.enc_cnn_opf(x_opf)
		t = x_opf.size(2)
		opf_feats = F.interpolate(opf_feats, t, mode="linear", align_corners=False)
		opf_feats = opf_feats.permute(0, 2, 1)

		# concat original image features and optical flow features; (batch_size, n_frames, dim_vid + dim_opf)
		combined_feats = torch.cat((vid_feats[:, :n_frames, :], opf_feats[:, :n_frames, :]), dim=2)

		state = None
		output_rnn, output_state = self.rnn(combined_feats, state)  # batch_size, n_frames + 3, dim_hidden
		return output_rnn, output_state

	def train(self, mode=True):
		super().train(mode)
		# set eval for batch normalization layers in pretrained models
		self.enc_cnn_vid.eval()
		self.enc_cnn_vid.fc.train(mode)

		self.enc_cnn_opf.eval()
		self.enc_cnn_opf.logits.train(mode)
