import argparse

import torch
from torchvision import transforms

from S2VTModel import S2VTModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_options():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default='S2VTModel', help="with model to use")
	parser.add_argument('--trained_model', type=str, default='trained_model.pth', help="load trained model for testing")
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
	parser.add_argument('--vocab_size', type=int, default=117, help='vocabulary size')
	parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
	parser.add_argument('--learning_rate_decay_every', type=int, default=60,
						help='every how many iterations thereafter to drop LR?(in epoch)')
	parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)
	parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch number (useful in restarts)')
	parser.add_argument('--end_epoch', type=int, default=12, help='ending epoch number')
	parser.add_argument('--batch_size', type=int, default=20, help='minibatch size')
	parser.add_argument('--save_checkpoint_every', type=int, default=1,
						help='how often to save a model checkpoint (in epoch)?')
	parser.add_argument('--checkpoint_path', type=str, default='./model_run_data',
						help='directory to store checkpointed models')
	parser.add_argument('--weight_decay', type=float, default=5e-4,
						help='weight_decay. strength of weight regularization')

	parser.add_argument('--gpu', type=str, default='', help='gpu device number')
	parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (*.pth)')
	parser.add_argument('--shuffle', type=bool, default=True, help="boolean indicating shuffle required or not")
	parser.add_argument('--train_dataset_path', type=str, default="data/train/train", help="train dataset path")
	parser.add_argument('--test_dataset_path', type=str, default="data/test/test", help="test dataset path")
	parser.add_argument('--num_workers', type=int, default=0, help="number of workers to load batch")
	parser.add_argument('--train_annotation_path', type=str, default="data/training_annotation.json",
						help="path to training annotations")
	parser.add_argument('--resolution', type=int, default=224, help="frame resolution")
	return parser.parse_args()

def model_provider(opts):
	model = None
	if opts["model"] == 'S2VTModel':
		model = S2VTModel(
			vocab_size=opts["vocab_size"],
			dim_hidden=opts['dim_hidden'],
			dim_word=opts['dim_word'],
			max_len=opts["max_len"],
			dim_vid=opts["dim_vid"],
			n_layers=opts['num_layers'],
			rnn_cell=opts['rnn_type'],
			rnn_dropout_p=opts["rnn_dropout_p"]).to(device)
	return model

def data_transformations(opts):
	return transforms.Compose([
		transforms.Resize((opts["resolution"], opts["resolution"])),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])