import sys
import json
import os
import argparse
import time
import subprocess
import logging

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from S2VTModel import S2VTModel
from dataset import VRDataset
from utils import save_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')


def train(dataloader, model, optimizer, lr_scheduler, opts):
	loss_fns = [torch.nn.CrossEntropyLoss() for _ in range(3)]
	with open(opts["train_annotation_path"]) as f:
		training_annotation = json.load(f)

	logging.info(f"Starting training at {opts['start_epoch']} epochs and will run for {opts['end_epoch']} epochs "
	             f"using device: {device}")
	loss = None
	for epoch in range(opts["start_epoch"], opts["end_epoch"]):
		step = 0
		true_pos = 0
		total = 0
		for batch_idx, (video_ids, videos_tensor) in enumerate(dataloader):
			model.zero_grad()
			model.train()

			videos_tensor = videos_tensor.to(device)
			annot = torch.LongTensor([[training_annotation[item][0], training_annotation[item][1] + 35, training_annotation[item][2]]
									 for item in video_ids]).to(device)
			output, _ = model(x=videos_tensor, target_variable=annot)

			loss = loss_fns[0](output[:, 0, :], annot[:, 0]) + \
			       loss_fns[1](output[:, 1, :], annot[:, 1]) + \
			       loss_fns[2](output[:, 2, :], annot[:, 2])

			loss.backward()
			optimizer.step()
			lr_scheduler.step()
			logging.info(f"Step update | batch_idx: {batch_idx}, step: {step}, loss: {loss.item()}")

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
			
		logging.info(f"Epoch update | epoch: {epoch}, loss: {loss.item()}")

		if epoch % opts["save_checkpoint_every"] == 0:
			save_file_path = os.path.join(opts["checkpoint_path"], f"model_{epoch}.pth")
			save_checkpoint({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict()
			}, filename=save_file_path)
			logging.info(f"Model saved to {save_file_path}")

			model_info_path = os.path.join(opts["checkpoint_path"], 'model_score.txt')
			with open(model_info_path, 'a') as fh:
				time_str = time.strftime("%Y-%m-%d %H:%M:%S")
				fh.write(f"{time_str} | model update --- epoch: {epoch}, loss: {loss:.6f} \n\n")
	
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
		'validate': transforms.Compose([
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
			rnn_dropout_p=opts["rnn_dropout_p"]).to(device)

	if opts["resume"]:
		if os.path.isfile(opts["resume"]):
			logging.info(f'loading checkpoint {opts["resume"]}')
			checkpoint = torch.load(args.resume)
			model.load_state_dict(checkpoint['state_dict'])
			logging.info(f'loaded checkpoint {opts["resume"]}')
		else:
			logging.info(f'no checkpoint found at {opts["resume"]}')

	optimizer = optim.Adam(model.parameters(), lr=opts["learning_rate"], weight_decay=opts["weight_decay"])

	exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opts["learning_rate_decay_every"],
	                                             gamma=opts["learning_rate_decay_rate"])

	vrdataset = VRDataset(img_root=opts["train_dataset_path"], transform=data_transformations["train"])
	dataloader = DataLoader(vrdataset, batch_size=opts["batch_size"], shuffle=opts["shuffle"],
	                        num_workers=opts["num_workers"])
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
	parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch number (useful in restarts)')
	parser.add_argument('--end_epoch', type=int, default=30, help='ending epoch number')
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
	parser.add_argument('--num_workers', type=int, default=0, help="number of workers to load batch")
	parser.add_argument('--train_annotation_path', type=str, default="data/training_annotation.json",
	                    help="path to training annotations")
	parser.add_argument('--resolution', type=int, default=224, help="frame resolution")

	args = parser.parse_args()

	opts = vars(args)
	opts["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
	opt_json = os.path.join(opts["checkpoint_path"], 'opt_info.json')
	if not os.path.isdir(opts["checkpoint_path"]):
		os.mkdir(opts["checkpoint_path"])

	logging.info(json.dumps(opts, indent=4))
	logging.info(f'__Python VERSION: {sys.version}')
	logging.info(f'__pyTorch VERSION: {torch.__version__}')
	with open(opt_json, 'w') as fh:
		json.dump(opts, fh, indent=4)

	# https://discuss.pytorch.org/t/cuda-visible-devices-make-gpu-disappear/21439
	os.environ['CUDA_VISIBLE_DEVICES'] = opts["gpu"]
	if opts["gpu"]:
		logging.info(f'__CUDNN VERSION: {torch.backends.cudnn.version()}')
		logging.info(f'__Number CUDA Devices: {torch.cuda.device_count()}')
		cuda_result = subprocess.call(
			["nvidia-smi", "--format=csv",
			 "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
		logging.info(f'Available devices {torch.cuda.device_count()}')
		logging.info(f'Current CUDA Device: GPU {torch.cuda.current_device()}')
		logging.info(f'Current CUDA Device Name: {torch.cuda.get_device_name(int(opts["gpu"]))}')

	main(opts)
