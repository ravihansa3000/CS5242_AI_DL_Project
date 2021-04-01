import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')


def init_hidden(batch_size, n_units):
	"""
	the weights are of the form (batch_size, n_units)
	note that batch_first=True does not affect the shape of hidden states
	:param batch_size:
	:param n_layers:
	:param n_units:
	:return:
	"""
	hidden_a = torch.randn(batch_size, n_units)
	hidden_b = torch.randn(batch_size, n_units)

	hidden_a = Variable(hidden_a).to(device)
	hidden_b = Variable(hidden_b).to(device)

	return hidden_a, hidden_b


class S2VTModel(nn.Module):
	def __init__(self, vocab_size=117 + 1, dim_hidden=200, dim_word=200, max_len=3, dim_vid=500, sos_id=117,
	             rnn_cell='lstm'):
		super(S2VTModel, self).__init__()
		if rnn_cell.lower() == 'lstm':
			self.rnn_cell = nn.LSTMCell
		elif rnn_cell.lower() == 'gru':
			self.rnn_cell = nn.GRUCell

		# features of video frames are embedded to a 500 dimensional space
		self.dim_vid = dim_vid
		self.dim_hidden = dim_hidden

		# words are transformed to 500 feature dimension
		self.dim_word = dim_word

		# annotation vocabulary size + <sos>
		self.vocab_size = vocab_size

		# annotation attributes: <sos>, object1, relationship
		# object2 is only predicted at the end
		self.max_length = max_len

		# start-of-sentence and end-of-sentence ids
		self.sos_id = sos_id

		# word embeddings lookup tables for each element; object1, relationship, object2, <sos>
		self.embedding = nn.Embedding(self.vocab_size, self.dim_word).to(device)

		self.rnn1 = self.rnn_cell(self.dim_vid, self.dim_hidden).to(device)
		self.rnn2 = self.rnn_cell(self.dim_hidden + self.dim_word, self.dim_hidden).to(device)

		# final linear layer to decode the words
		self.linear_layers = [nn.Linear(self.dim_hidden, C).to(device) for C in [35, 82, 35]]

		self.state1 = None
		self.state2 = None

	def forward(self, vid_feats: torch.Tensor, target_variable=None, tf_mode=True, top_k=5):
		"""
		:param vid_feats: tensor containing encoded features of shape: (batch_size, n_frames, dim_vid)

		:param target_variable: target labels of the ground truth annotations of shape (batch_size, max_length)
			Each row corresponds to a set of training annotations; (<sos>, object1, relationship, object2)

		:param tf_mode: teacher forcing mode
			True: ground truth labels are used as input to the 2nd layer of RNN
			False: only predictions are used as input

		:return:
		"""

		batch_size, n_frames, _ = vid_feats.shape

		# https://github.com/pytorch/pytorch/issues/3920
		# paddings to be used for the 2nd layer
		padding_words = Variable(torch.empty(batch_size, self.dim_word, dtype=vid_feats.dtype)).zero_().to(device)

		# paddings to be used for the 1st layer, added one by one in loop; shape of (batch_size * 1)
		padding_frames = Variable(torch.empty(batch_size, self.dim_vid, dtype=vid_feats.dtype)).zero_().to(device)

		# reset the hidden and cell states of 2 LSTM layers
		# must be done before running a new batch, otherwise it will treat a new batch as a continuation of a sequence
		self.state1 = init_hidden(batch_size, self.dim_hidden)
		self.state2 = init_hidden(batch_size, self.dim_hidden)

		for vid_step_idx in range(n_frames):
			# feed the video features of current step into the first layer of RNN
			vid_idx_t = torch.tensor([vid_step_idx]).to(device)
			vid_step_feats = torch.index_select(vid_feats, 1, vid_idx_t).squeeze(1)
			self.state1 = self.rnn1(vid_step_feats, self.state1)

			# concatenate paddings (to 2nd layer) with output from the 1st layer; input2: (batch_size, 30, dim_word)
			input2 = torch.cat((self.state1[0], padding_words), dim=1)

			# feed concatenated output from 1st layer to the 2nd layer
			self.state2 = self.rnn2(input2, self.state2)

		# By this point we have already fed input features (of 30 frames) to 1st layer of LSTM and padded concatenated
		# inputs to 2nd layer of LSTM. Remaining 3 steps will be performed using word embeddings
		model_outputs = []
		seq_k_preds = []
		net_out = None

		if self.training:
			sos_tensor = Variable(torch.LongTensor([[self.sos_id]] * batch_size)).to(device)
			target_variable = torch.cat((sos_tensor, target_variable), dim=1)
			for i in range(self.max_length):
				if tf_mode or i == 0:
					current_word_embed = self.embedding(target_variable[:, i])
				else:
					# use predicted output instead of ground truth
					logits = F.log_softmax(net_out, dim=1)
					_, top_preds = torch.max(logits, dim=1)

					# offset embeddings for <relationship> entity type since predictions are 0-indexed
					current_word_embed = self.embedding(torch.add(top_preds, 35) if i == 1 else top_preds)
					logging.info(f"i: {i}, target_variable: {target_variable.tolist()} \n"
					              f"top_preds: {top_preds.tolist()}")

				# input frame paddings to 1st layer for the (n_frames + i + 1)th time step
				self.state1 = self.rnn1(padding_frames, self.state1)

				# concatenate word embeddings with output from the 1st layer
				input2 = torch.cat((self.state1[0], current_word_embed), dim=1)

				# feed to 2nd layer of LSTM and get the final network output
				self.state2 = self.rnn2(input2, self.state2)
				net_out = self.linear_layers[i](self.state2[0])
				model_outputs.append(net_out)

		else:
			current_word_embed = self.embedding(Variable(torch.LongTensor([self.sos_id] * batch_size)).to(device))
			for i in range(self.max_length):
				self.state1 = self.rnn1(padding_frames, self.state1)
				input2 = torch.cat((self.state1[0], current_word_embed), dim=1)
				self.state2 = self.rnn2(input2, self.state2)
				net_out = self.linear_layers[i](self.state2[0])
				model_outputs.append(net_out)

				logits = F.log_softmax(net_out, dim=1)

				# get word embeddings for the next step using the indices of best predictions in the prev step
				_, topk_preds = torch.topk(logits, k=top_k, dim=1)
				_, top_preds = torch.max(logits, dim=1)
				seq_k_preds.append(topk_preds)

				# offset embeddings for <relationship> entity type since predictions are 0-indexed
				current_word_embed = self.embedding(torch.add(top_preds, 35) if i == 1 else top_preds)

		return model_outputs, seq_k_preds
