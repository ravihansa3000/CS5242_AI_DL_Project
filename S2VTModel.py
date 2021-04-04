import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from encoder import Encoder
from utils import init_hidden

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class S2VTModel(nn.Module):

	def __init__(self, vocab_size=117 + 1, dim_hidden=500, dim_word=500, max_len=3, dim_vid=500, sos_id=117,
	             n_layers=1, rnn_cell='lstm', rnn_dropout_p=0.2):

		super(S2VTModel, self).__init__()
		if rnn_cell.lower() == 'lstm':
			self.rnn_cell = nn.LSTM
		elif rnn_cell.lower() == 'gru':
			self.rnn_cell = nn.GRU

		# features of video frames are embedded to a 500 dimensional space
		self.dim_vid = dim_vid

		# initialize the encoder cnn
		self.encoder = Encoder(dim_vid=self.dim_vid, dim_hidden=dim_hidden, rnn_cell=self.rnn_cell).to(device)

		# objects: 35, relationships: 82; <object1>,<relationship>,<object2>
		self.dim_outputs = [35, 82, 35]

		# number of total embeddings
		self.vocab_size = vocab_size

		# LSTM hidden feature dimension
		self.dim_hidden = dim_hidden

		# words are transformed to 500 feature dimension
		self.dim_word = dim_word

		# annotation attributes: <sos>, object1, relationship
		# object2 is only predicted at the end
		self.max_length = max_len

		# start-of-sentence and end-of-sentence ids
		self.sos_id = sos_id

		# word embeddings lookup table with; + 1 for <sos>
		self.embedding = nn.Embedding(self.vocab_size, self.dim_word)

		self.rnn = self.rnn_cell(self.dim_hidden + self.dim_word, self.dim_hidden, n_layers,
		                         batch_first=True, dropout=rnn_dropout_p).to(device)
		for name, param in self.rnn.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			elif 'weight' in name:
				nn.init.xavier_normal_(param)

		self.out_lin_mods = nn.ModuleList([nn.Linear(self.dim_hidden, dim_out).to(device) for dim_out in self.dim_outputs])
		for m_lin in self.out_lin_mods:
			torch.nn.init.xavier_uniform_(m_lin.weight)

	def forward(self, x: torch.Tensor, target_variable=None, tf_mode=True, top_k=5):
		"""
		:param x: Tensor containing video features of shape (batch_size, n_frames, cnn_input_c, cnn_input_h, cnn_input_w)
			n_frames is the number of video frames

		:param target_variable: target labels of the ground truth annotations of shape (batch_size, max_length)
			Each row corresponds to a set of training annotations; (object1, relationship, object2)

		:param tf_mode: teacher forcing mode
			True: ground truth labels are used as input to the 2nd layer of RNN
			False: only predictions are used as input

		:param top_k: top K predictions will be returned in eval mode

		:return:
		"""
		batch_size = x.shape[0]
		enc_out = self.encoder(x)

		input1 = enc_out[:, :30, :]  # input1: (batch_size, 30, dim_vid)
		input2 = enc_out[:, 30:, :]  # input2: (batch_size, 3, dim_word)

		# https://github.com/pytorch/pytorch/issues/3920
		# paddings to be used for the 2nd layer
		padding_words = Variable(torch.empty(batch_size, 30, self.dim_word, dtype=x.dtype)).zero_().to(device)

		# concatenate paddings (for the 2nd layer) with output from the 1st layer
		input1 = torch.cat((input1, padding_words), dim=2)

		# feed concatenated output from 1st layer to the 2nd layer
		state = init_hidden(batch_size, 1, self.dim_hidden)
		rnn_out, state = self.rnn(input1, state)  # (batch_size, 30, dim_word)

		seq_probs = []
		seq_k_preds = []
		net_out = None

		# By this point we have already fed input features (of 30 frames) to 1st layer of LSTM and padded concatenated
		# inputs to 2nd layer of LSTM. Remaining 3 steps will be performed using word embeddings

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

				self.rnn.flatten_parameters()

				input1 = torch.cat((input2[:, i, :].unsqueeze(1), current_word_embed.unsqueeze(1)), dim=2)
				rnn_out, state = self.rnn(input1, state)


				net_out = self.out_lin_mods[i](rnn_out.squeeze(1))
				seq_probs.append(net_out)
		else:
			current_word_embed = self.embedding(Variable(torch.LongTensor([self.sos_id] * batch_size)).to(device))
			for i in range(self.max_length):
				# optimize for GPU (applicable only when CUDA/GPU capability is present in the system)
				self.rnn.flatten_parameters()

				input1 = torch.cat((input2[:, i, :].unsqueeze(1), current_word_embed.unsqueeze(1)), dim=2)
				rnn_out, state = self.rnn(input1, state)
				net_out = self.out_lin_mods[i](rnn_out.squeeze(1))
				logits = F.log_softmax(net_out, dim=1)

				seq_probs.append(logits)

				# get word embeddings for the next step using the indices of best predictions in the prev step
				_, topk_preds = torch.topk(logits, k=top_k, dim=1)
				_, top_preds = torch.max(logits, dim=1)
				current_word_embed = self.embedding(torch.add(top_preds, 35) if i == 1 else top_preds)
				seq_k_preds.append(topk_preds)

		return seq_probs, seq_k_preds
