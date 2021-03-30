import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from encoder import EncoderCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class S2VTModel(nn.Module):
	def __init__(self, vocab_size=117, dim_hidden=500, dim_word=500, max_len=4, dim_vid=500, sos_id=117,
	             n_layers=1, rnn_cell='lstm', rnn_dropout_p=0.2, cnn_output_feature_dims=500):
		super(S2VTModel, self).__init__()
		if rnn_cell.lower() == 'lstm':
			self.rnn_cell = nn.LSTM
		elif rnn_cell.lower() == 'gru':
			self.rnn_cell = nn.GRU

		# initialize the encoder cnn
		self.encoder = EncoderCNN(output_feature_dims=cnn_output_feature_dims).to(device)

		# features of video frames are embedded to a 500 dimensional space
		self.dim_vid = dim_vid

		# object vocab: 35 + relationships vocab: 82
		self.dim_output = vocab_size
		self.dim_hidden = dim_hidden

		# words are transformed to 500 feature dimension
		self.dim_word = dim_word

		# annotation attributes: <sos>, object1, relationship
		# object2 is only predicted at the end
		self.max_length = max_len

		# start-of-sentence and end-of-sentence ids
		self.sos_id = sos_id

		# word embeddings lookup table with; + 1 for <sos>
		self.embedding = nn.Embedding(self.dim_output + 1, self.dim_word)

		self.rnn1 = self.rnn_cell(self.dim_vid, self.dim_hidden, n_layers,
		                          batch_first=True, dropout=rnn_dropout_p).to(device)
		self.rnn2 = self.rnn_cell(self.dim_hidden + self.dim_word, self.dim_hidden, n_layers,
		                          batch_first=True, dropout=rnn_dropout_p).to(device)

		self.out = nn.ModuleList([ \
			nn.Linear(self.dim_hidden, 35).to(device), \
			nn.Linear(self.dim_hidden, 82).to(device), \
			nn.Linear(self.dim_hidden, 35).to(device)])

	def forward(self, logging, x: torch.Tensor, target_variable=None, opts=None):
		"""
		:param x: Tensor containing video features of shape (batch_size, n_frames, cnn_input_c, cnn_input_h, cnn_input_w)
			n_frames is the number of video frames

		:param target_variable: target labels of the ground truth annotations of shape (batch_size, max_length - 1)
			Each row corresponds to a set of training annotations; (object1, relationship, object2)

		:param opts: not used

		:return:
		"""

		vid_imgs_encoded = []
		for i in range(x.shape[0]):
			vid_imgs_encoded.append(self.encoder(x[i]))

		vid_feats = torch.stack(vid_imgs_encoded, dim=0)  # vid_feats: (batch_size, n_frames, dim_vid)

		batch_size, n_frames, _ = vid_feats.shape

		# https://github.com/pytorch/pytorch/issues/3920
		# paddings to be used for the 2nd layer
		padding_words = Variable(
			torch.empty(batch_size, n_frames, self.dim_word, dtype=vid_feats.dtype)).zero_().to(device)

		# paddings to be used for the 1st layer, added one by one in loop; shape of (batch_size * 1)
		padding_frames = Variable(
			torch.empty(batch_size, 1, self.dim_vid, dtype=vid_feats.dtype)).zero_().to(device)

		# hidden and cell states of 2 LSTM layers
		state1 = None
		state2 = None

		# feed the video features into the first layer of RNN
		# only 30 steps will be performed since n_frames is always 30 (based on train/test data)
		output1, state1 = self.rnn1(vid_feats, state1)  # output1: (batch_size, 30, dim_vid)

		# concatenate paddings (for the 2nd layer) with output from the 1st layer
		input2 = torch.cat((output1, padding_words), dim=2)  # input2: (batch_size, 30, dim_word)

		# feed concatenated output from 1st layer to the 2nd layer
		output2, state2 = self.rnn2(input2, state2)  # output2: (batch_size, 30, dim_word)

		seq_probs = []
		seq_preds = []

		# By this point we have already fed input features (of 30 frames) to 1st layer of LSTM and padded concatenated
		# inputs to 2nd layer of LSTM. Remaining 3 steps will be performed using word embeddings

		if self.training:
			sos_tensor = Variable(torch.LongTensor([[self.sos_id]] * batch_size)).to(device)
			target_variable = torch.cat((sos_tensor, target_variable), dim=1)
			for i in range(self.max_length - 1):
				# generate word embeddings using the i-th column (batch_size * 1)
				current_words = self.embedding(target_variable[:, i])
				# optimize for GPU (applicable only when CUDA/GPU capability is present in the system)
				self.rnn1.flatten_parameters()
				self.rnn2.flatten_parameters()

				# input frame paddings to 1st layer for the last 3 time steps
				output1, state1 = self.rnn1(padding_frames, state1)  # output1: (batch_size, i, dim_vid)
				# concatenate word embeddings with output from the 1st layer
				input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2)  # input2: (batch_size, i, dim_word)
				output2, state2 = self.rnn2(input2, state2)  # output2: (batch_size, i, dim_word)

				# feed RNN output to linear layer and get probabilities for each classification
				logits = self.out[i](output2.squeeze(1))  # logits: (batch_size, dim_output)
				seq_probs.append(logits)  # seq_probs: (batch_size, 1, dim_output)
		else:
			current_words = self.embedding(Variable(torch.LongTensor([self.sos_id] * batch_size)).to(device))
			for i in range(self.max_length - 1):
				# optimize for GPU (applicable only when CUDA/GPU capability is present in the system)
				self.rnn1.flatten_parameters()
				self.rnn2.flatten_parameters()

				output1, state1 = self.rnn1(padding_frames, state1)  # output1: (batch_size, 1, dim_vid)
				input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2)
				output2, state2 = self.rnn2(input2, state2)  # output2: (batch_size, 1, dim_word)
				logits = self.out[i](output2.squeeze(1))  # logits: (batch_size, dim_output)
				logits = F.log_softmax(logits, dim=1)
				seq_probs.append(logits)  # seq_probs: (batch_size, 1, dim_output)

				# get word embeddings for the next step using the indices of best predictions in the prev step
				preds = torch.argmax(logits, dim=1)  # preds: (batch_size, 1)
				preds = torch.LongTensor(preds)
				current_words = self.embedding(torch.add(preds, 35) if i == 1 else preds)
				seq_preds.append(preds.unsqueeze(1))  # seq_preds: (batch_size, 1, 1)

			seq_preds = torch.stack(seq_preds, 1)  # seq_probs: (batch_size, 3, 1)

		return seq_probs, seq_preds
