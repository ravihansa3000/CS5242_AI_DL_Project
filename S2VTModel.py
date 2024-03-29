import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from encoder import Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class S2VTModel(nn.Module):
	def __init__(self, vocab_size=117 + 1, dim_hidden=250, dim_word=250, max_len=3, dim_vid=500, dim_opf=500,
				 sos_id=117, n_layers=1, rnn_cell='lstm', input_dropout_p=0.5, rnn_dropout_p=0.5):
		super(S2VTModel, self).__init__()
		if rnn_cell.lower() == 'lstm':
			self.rnn_cell = nn.LSTM
		elif rnn_cell.lower() == 'gru':
			self.rnn_cell = nn.GRU

		self.dim_vid = dim_vid  # features of video frames are embedded to a 500 dimensional space
		self.dim_opf = dim_opf  # features of optical flow frames are embedded to a 500 dimensional space
		self.dim_outputs = [35, 82, 35]  # objects: 35, relationships: 82; <object1>,<relationship>,<object2>
		self.vocab_size = vocab_size  # number of total embeddings
		self.dim_hidden = dim_hidden  # LSTM hidden feature dimension
		self.dim_word = dim_word  # words are transformed to 500 feature dimension

		# annotation attributes: <sos>, object1, relationship
		# object2 is only predicted at the end
		self.max_length = max_len

		# start-of-sentence and end-of-sentence ids
		self.sos_id = sos_id

		# word embeddings lookup table with; + 1 for <sos>
		self.embedding = nn.Embedding(self.vocab_size, self.dim_word).to(device)

		# initialize the encoder cnn
		self.encoder = Encoder(
			dim_vid=self.dim_vid,
			dim_opf=self.dim_opf,
			dim_hidden=self.dim_hidden,
			rnn_cell=self.rnn_cell,
			n_layers=n_layers,
			rnn_dropout_p=rnn_dropout_p,
		).to(device)

		# dropout for RNN output
		self.output_dropout = nn.Dropout(p=0.4).to(device)

		# linear layers that predict each element of a record
		self.dim_rnn_out = 2 * self.dim_hidden if self.encoder.rnn.bidirectional else self.dim_hidden
		self.dim_lin_in = [self.dim_rnn_out, self.dim_rnn_out, self.dim_rnn_out]
		self.out_lin_mods = nn.ModuleList(
			[nn.Linear(item[0], item[1]).to(device) for item in zip(self.dim_lin_in, self.dim_outputs)]
		)
		for m_lin in self.out_lin_mods:
			torch.nn.init.xavier_uniform_(m_lin.weight)

	def forward(self, x_vid: torch.Tensor, x_opf: torch.Tensor, target_y=None, tf_rate=0.5, top_k=5):
		"""
		:param x_vid: Tensor containing video features of shape (batch_size, n_frames, cnn_input_c, cnn_input_h, cnn_input_w)
			n_frames is the number of video frames

		:param x_opf: Tensor containing features from optical flow images

		:param target_y: target labels of the ground truth annotations of shape (batch_size, max_length)
			Each row corresponds to a set of training annotations; (object1, relationship, object2)

		:param tf_rate: Probability for choosing ground truth (teacher forcing) instead of LSTM output

		:param top_k: top K predictions will be returned in eval mode

		:return:
		"""
		batch_size = x_vid.shape[0]
		enc_out, state = self.encoder(x_vid, x_opf)

		seq_probs = []
		seq_k_preds = []
		net_out = None

		if self.training:
			sos_tensor = Variable(torch.LongTensor([[self.sos_id]] * batch_size)).to(device)
			target_y = torch.cat((sos_tensor, target_y), dim=1)
			for i in range(self.max_length):
				# offset <relationship> element to avoid conflicts (all target_y labels are 0-indexed)
				# apply teacher forcing probabilistically
				# in case of 1st prediction use <sos> token since it's always present in both training and testing
				if i == 0 or random.random() < tf_rate:
					current_word_embed = self.embedding(torch.add(target_y[:, i], 35) if i == 1 else target_y[:, i])
				else:
					# use predicted output instead of ground truth
					logits = F.log_softmax(net_out, dim=1)
					_, top_preds = torch.max(logits, dim=1)

					# offset embeddings for <relationship> entity type since predictions are 0-indexed
					current_word_embed = self.embedding(torch.add(top_preds, 35) if i == 1 else top_preds)

				self.encoder.rnn.flatten_parameters()  # optimize for GPU
				rnn_out, state = self.encoder.rnn(current_word_embed.unsqueeze(1), state)

				dropped_out = self.output_dropout(rnn_out)
				dropped_out = dropped_out.view(batch_size, self.dim_rnn_out)

				net_out = self.out_lin_mods[i](dropped_out)
				seq_probs.append(net_out)
		else:
			current_word_embed = self.embedding(Variable(torch.LongTensor([self.sos_id] * batch_size)).to(device))
			for i in range(self.max_length):
				self.encoder.rnn.flatten_parameters()  # optimize for GPU
				rnn_out, state = self.encoder.rnn(current_word_embed.unsqueeze(1), state)

				dropped_out = self.output_dropout(rnn_out)
				dropped_out = dropped_out.view(batch_size, self.dim_rnn_out)

				net_out = self.out_lin_mods[i](dropped_out)
				logits = F.log_softmax(net_out, dim=1)
				seq_probs.append(logits)

				# get word embeddings for the next step using the indices of best predictions in the prev step
				_, topk_preds = torch.topk(logits, k=top_k, dim=1)
				_, top_preds = torch.max(logits, dim=1)
				current_word_embed = self.embedding(torch.add(top_preds, 35) if i == 1 else top_preds)
				seq_k_preds.append(topk_preds)

		return seq_probs, seq_k_preds
