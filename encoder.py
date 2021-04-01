import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
	def __init__(self, output_feature_dims=500, dim_hidden=500, rnn_cell=nn.LSTM, rnn_dropout_p=0):
		"""Load the pretrained VGG-16 and replace top fc layer."""
		super(Encoder, self).__init__()

		self.output_feature_dims = output_feature_dims
		self.dim_hidden = dim_hidden
		self.rnn_cell = rnn_cell

		self.vgg16 = models.vgg16(pretrained=True).to(device)
		for param in self.vgg16.parameters():
			param.requires_grad = False

		# Learn weights of the final Conv layer
		# for i in [18, 19, 20]:
		# 	for param in self.vgg16.features[i].parameters():
		# 		param.requires_grad = True

		self.vgg16.classifier = nn.Linear(25088, output_feature_dims)

		# encoder BiRNN
		self.rnn = rnn_cell(self.output_feature_dims, self.dim_hidden, 1,
							batch_first=True, dropout=rnn_dropout_p).to(device)


	def forward(self, logging, x):
		"""Convert a batch of videos into embeddings and feed them into the encoder BiRNN"""
		
		batch_size = x.shape[0]
		vid_imgs_encoded = []
		for i in range(batch_size):
			vid_imgs_encoded.append(self.vgg16(x[i]))
		
		vid_feats = torch.stack(vid_imgs_encoded, dim=0) # batch_size, 30, output_feature_dims

		padding_frames = [Variable(
			torch.empty(vid_feats.shape[0], self.output_feature_dims, dtype=vid_feats.dtype)).zero_().to(device) for _ in range(3)]
		
		vid_feats_list = [vid_feats[:, i, :] for i in range(vid_feats.shape[1])]
		vid_feats_list.extend(padding_frames)

		rnn_input = torch.stack(vid_feats_list, dim=1)

		state = None
		output, _ = self.rnn(rnn_input, state) # batch_size, 33, dim_hidden
		return output
