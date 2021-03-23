import json
import os
import argparse
import numpy as np
import torch
import torch as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

from S2VTModel import S2VTModel
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import VRDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataloader, model, optimizer, lr_scheduler, opts):
	loss_fns = [torch.nn.CrossEntropyLoss() for i in range(3)]
	with open(opts["train_annotation_path"]) as f:
		training_annotation = json.load(f)
	
	print ("Starting training for {} epochs...".format(opts["epochs"]))

	for epoch in range(opts["epochs"]):
		step = 0
		true_pos = 0
		total = 0
		for video_batch in dataloader:
			model.zero_grad()
			model.train()

			video_ids, videos_tensor = video_batch
			annots = torch.LongTensor([training_annotation[id] for id in video_ids])
			output, _ = model(x=videos_tensor, target_variable=annots)

			loss = loss_fns[0](output[:, 0, :], annots[:, 0]) + loss_fns[1](output[:, 1, :], annots[:, 1]) + loss_fns[2](output[:, 2, :], annots[:, 2])
			loss.backward()
			optimizer.step()
			lr_scheduler.step()
			print (f"Loss at step {step}: ", loss.item())

			true_pos_per_step = 0
			total_per_step = len(annots) * 3
			preds = torch.max(output, dim=2)
			for (pred, annot) in zip(preds, annots):
				true_pos_per_step += float((pred == annot).sum())
			true_pos += true_pos_per_step
			total += total_per_step
			print(f'Accuracy at step {step}: ', true_pos_per_step / total_per_step * 100)

			step += 1
		print(f'Accuracy at epoch {epoch}: ', true_pos / total * 100)
	
	return model

def main(opts):
	model = None
	opts["vocab_size"] = 117

	data_transformations = {
		'train': transforms.Compose([
			transforms.Resize((opts["resolution"], opts["resolution"])),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			transforms.Resize((opts["resolution"], opts["resolution"])),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	}

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

	vrdataset = VRDataset(img_root=opts["train_dataset_path"], transform=data_transformations["train"])
	dataloader = DataLoader(vrdataset, batch_size=opts["batch_size"], shuffle=opts["shuffle"], num_workers=opts["num_workers"])
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
	parser.add_argument('--batch_size', type=int, default=10, help='minibatch size')
	parser.add_argument('--save_checkpoint_every', type=int, default=50,
	                    help='how often to save a model checkpoint (in epoch)?')
	parser.add_argument('--checkpoint_path', type=str, default='save', help='directory to store checkpointed models')
	parser.add_argument('--weight_decay', type=float, default=5e-4,
	                    help='weight_decay. strength of weight regularization')
	
	parser.add_argument('--gpu', type=str, default='0', help='gpu device number')
	parser.add_argument('--shuffle', type=bool, default=False, help="boolean indicating shuffle required or not")
	parser.add_argument('--train_dataset_path', type=str, default="data/train/train", help="train dataset path")
	parser.add_argument('--num_workers', type=int, default=0, help="number of workers to load batch")
	parser.add_argument('--train_annotation_path', type=str, default="data/training_annotation.json", help="path to training annotations")
	parser.add_argument('--resolution', type=int, default=224, help="frame resolution")

	args = parser.parse_args()

	opts = vars(args)
	os.environ['CUDA_VISIBLE_DEVICES'] = opts["gpu"]
	opt_json = os.path.join(opts["checkpoint_path"], 'opt_info.json')
	if not os.path.isdir(opts["checkpoint_path"]):
		os.mkdir(opts["checkpoint_path"])

	print(json.dumps(opts, indent=4))
	main(opts)
