import json
import os
import argparse
import numpy as np
import torch
import torch as nn
import torch.nn.functional as F
import torch.optim as optim
from S2VTModel import S2VTModel
from feature_extractor import dataloader, training_annotation, encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataloader, model, optimizer, lr_scheduler, opts):
	loss_fn = torch.nn.CrossEntropyLoss()

	for epoch in range(opts["epochs"]):
		batch_feats = []
		batch_targets = []
		batch_targets_one_hot = []

		# step = 0

		for video in dataloader:
			annot = training_annotation[video['video_id'][0]]
			annot_one_hot = F.one_hot(torch.LongTensor(annot), num_classes=opts["vocab_size"])
			batch_targets.append(torch.LongTensor(training_annotation[video['video_id'][0]]))
			batch_targets_one_hot.append(annot_one_hot)
			frame_feats = []
			for frame in video['frames']:
				frame_feats.append(torch.Tensor(encoder(frame)))
			batch_feats.append(torch.cat(frame_feats))

			# step += 1
			# if step == 2:
			# 	break

		batch_feats = torch.cat(batch_feats).reshape(-1, len(video['frames']), opts["dim_vid"])
		batch_targets = torch.cat(batch_targets).reshape(-1, opts["max_len"] - 1)
		batch_targets_one_hot = torch.cat(batch_targets_one_hot).reshape(-1, opts["max_len"] - 1, opts["vocab_size"])
		outputs = model(batch_feats, batch_targets)

		# loss = loss_fn(outputs[0].reshape(-1), batch_targets_one_hot.reshape(-1))
		optimizer.zero_grad()
		# loss.backward()
		optimizer.step()
		lr_scheduler.step()

		# if epoch % 1 == 0:
		# 	print (f'Epoch {epoch + 1}/{opts["epochs"]}, Loss: {loss.item():.4f}')

def main(opts):
	model = None
	opts["vocab_size"] = 117

	if opts["model"] == 'S2VTModel':
		model = S2VTModel(
			vocab_size=opts["vocab_size"],
			dim_hidden=opts['dim_hidden'],
			dim_word=opts['dim_word'],
			max_len=opts["max_len"],
			dim_vid=opts["dim_vid"],
			n_layers=opts['num_layers'],
			rnn_cell=opts['rnn_type'],
			rnn_dropout_p=opts["rnn_dropout_p"])

	# model = model.cuda()

	optimizer = optim.Adam(model.parameters(), lr=opts["learning_rate"], weight_decay=opts["weight_decay"])

	exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opts["learning_rate_decay_every"],
	                                             gamma=opts["learning_rate_decay_rate"])

	train(dataloader, model, optimizer, exp_lr_scheduler, opts)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default='S2VTModel', help="with model to use")
	parser.add_argument("--max_len", type=int, default=4, help='max length of captions(containing <sos>)')
	parser.add_argument('--dim_hidden', type=int, default=500, help='size of the rnn hidden layer')
	parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')
	parser.add_argument('--input_dropout_p', type=float, default=0.2,
	                    help='strength of dropout in the Language Model RNN')
	parser.add_argument('--rnn_type', type=str, default='lstm', help='lstm or gru')
	parser.add_argument('--rnn_dropout_p', type=float, default=0,
	                    help='strength of dropout in the Language Model RNN')
	parser.add_argument('--dim_word', type=int, default=500,
	                    help='the encoding size of each token in the vocabulary, and the video.')
	parser.add_argument('--dim_vid', type=int, default=500, help='dim of features of video frames')
	parser.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
	parser.add_argument('--learning_rate_decay_every', type=int, default=200,
	                    help='every how many iterations thereafter to drop LR?(in epoch)')
	parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)

	parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
	parser.add_argument('--batch_size', type=int, default=128, help='minibatch size')
	parser.add_argument('--save_checkpoint_every', type=int, default=50,
	                    help='how often to save a model checkpoint (in epoch)?')
	parser.add_argument('--checkpoint_path', type=str, default='save', help='directory to store checkpointed models')
	parser.add_argument('--weight_decay', type=float, default=5e-4,
	                    help='weight_decay. strength of weight regularization')

	parser.add_argument('--gpu', type=str, default='0', help='gpu device number')
	args = parser.parse_args()

	opts = vars(args)
	os.environ['CUDA_VISIBLE_DEVICES'] = opts["gpu"]
	opt_json = os.path.join(opts["checkpoint_path"], 'opt_info.json')
	if not os.path.isdir(opts["checkpoint_path"]):
		os.mkdir(opts["checkpoint_path"])

	print(json.dumps(opts, indent=4))
	main(opts)
