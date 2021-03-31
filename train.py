import sys
import json
import os
import time
import subprocess
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from dataset import VRDataset
from utils import save_checkpoint
from model_config import model_options, model_provider, data_transformations

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

			annots = torch.LongTensor([[training_annotation[item][0], training_annotation[item][1] + 35, training_annotation[item][2]]
							 			for item in video_ids]).to(device)
			output, _ = model(logging, x=videos_tensor, target_variable=annots)
			annots[:, 1] = torch.sub(annots[:, 1], 35)
			loss = loss_fns[0](output[0], annots[:, 0]) + \
			       loss_fns[1](output[1], annots[:, 1]) + \
			       loss_fns[2](output[2], annots[:, 2])

			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
			optimizer.step()
			lr_scheduler.step()

			lr_params = []
			for param_group in optimizer.param_groups:
				lr_params.append(param_group["lr"])
			logging.info(f'lr_params: {lr_params}')
			logging.info(f"Step update | batch_idx: {batch_idx}, step: {step}, loss: {loss.item()}")

			with torch.no_grad():
				preds = [torch.nn.functional.log_softmax(op, dim=1) for op in output]
				true_pos_per_step = 0
				total_per_step = opts["batch_size"]
				preds = torch.stack([torch.argmax(pred, dim=1) for pred in preds], dim=1)
				for (pred, annot) in zip(preds, annots):
					logging.info(f"{pred}, {annot}")
					if torch.equal(pred, annot):
						true_pos_per_step += 1
				true_pos += true_pos_per_step
				total += total_per_step
				step_acc = true_pos_per_step / total_per_step * 100
				logging.info(f'Accuracy at step {step}: {step_acc}')

			step += 1
		epoch_acc = true_pos / total * 100
		logging.info(f'Accuracy at epoch {epoch}: {epoch_acc}')
			
		logging.info(f"Epoch update | epoch: {epoch}, loss: {loss.item()}")

		if epoch % opts["save_checkpoint_every"] == 0:
			save_file_path = os.path.join(opts["checkpoint_path"], f"model_{epoch}.pth")
			save_checkpoint({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict()
			}, filename=save_file_path)
			logging.info(f"Model saved to {save_file_path}")

			model_info_path = os.path.join(opts["checkpoint_path"], 'score.txt')
			with open(model_info_path, 'a') as fh:
				time_str = time.strftime("%Y-%m-%d %H:%M:%S")
				fh.write(f"{time_str} | model update --- epoch: {epoch}, loss: {loss:.6f} \n\n")
	
	logging.info("Training completed")
	return model


def main(opts):
	model = model_provider(opts)

	if opts["resume"]:
		if os.path.isfile(opts["resume"]):
			logging.info(f'loading checkpoint {opts["resume"]}')
			checkpoint = torch.load(args.resume)
			model.load_state_dict(checkpoint['state_dict'])
			logging.info(f'loaded checkpoint {opts["resume"]}')
		else:
			logging.info(f'no checkpoint found at {opts["resume"]}')
	logging.info(model)
	optimizer = optim.Adam(model.parameters(), lr=opts["learning_rate"], weight_decay=opts["weight_decay"])
	exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opts["learning_rate_decay_every"],
	                                             gamma=opts["learning_rate_decay_rate"])
	vrdataset = VRDataset(img_root=opts["train_dataset_path"], len=447, transform=data_transformations(opts, mode='train'))
	dataloader = DataLoader(vrdataset, batch_size=opts["batch_size"], shuffle=opts["shuffle"],
	                        num_workers=opts["num_workers"])
	train(dataloader, model, optimizer, exp_lr_scheduler, opts)


if __name__ == '__main__':
	opts = vars(model_options())
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
