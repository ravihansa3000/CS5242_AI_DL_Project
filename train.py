import json
import logging
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from dataset import VRDataset
from model_config import model_options, model_provider, data_transformations
from utils import save_checkpoint

from optical_flow import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')


def train(dataloader, model, optimizer, lr_scheduler, opts):
	loss_fns = [torch.nn.CrossEntropyLoss().to(device) for _ in range(3)]
	with open(opts["train_annotation_path"]) as f:
		train_ann_dict = json.load(f)

	logging.info(f"Starting training at {opts['start_epoch']} epochs and will run for {opts['end_epoch']} epochs "
	             f"using device: {device}")

	for epoch in range(opts["start_epoch"], opts["end_epoch"]):
		step = 0
		true_pos = 0
		total = 0
		losses_str = ''
		mAPk_scores_str = ''
		optimizer_lr_str = ''

		# disable teacher forcing after epoch milestone has elapsed
		tf_mode = False if epoch > opts["disable_tf_after_epoch"] else True

		for batch_idx, (video_ids, videos_tensor, videos_tensor_alternate) in enumerate(dataloader):
			# zero the parameter gradients
			model.train()
			optimizer.zero_grad()

			videos_tensor = videos_tensor.to(device)
			videos_tensor_alternate = videos_tensor_alternate.to(device)
			# offset <relationship> annotations when generating word embeddings
			batch_ann_t = torch.LongTensor([
				[train_ann_dict[item][0], train_ann_dict[item][1] + 35, train_ann_dict[item][2]]
				for item in video_ids
			]).to(device)
			output, _ = model(x=videos_tensor, x_alternate=videos_tensor_alternate, target_variable=batch_ann_t, tf_mode=tf_mode)

			# de-offset <relationship> annotations when calculating loss since linear output has 35 nodes
			batch_ann_t[:, 1] = torch.sub(batch_ann_t[:, 1], 35)
			loss1 = loss_fns[0](output[0], batch_ann_t[:, 0])
			loss2 = loss_fns[1](output[1], batch_ann_t[:, 1])
			loss3 = loss_fns[2](output[2], batch_ann_t[:, 2])
			total_loss = loss1 + loss2 + loss3
			total_loss.backward()

			# optimize with a gradient clip
			torch.nn.utils.clip_grad_norm_(model.parameters(), opts["grad_clip"])
			optimizer.step()

			# print stats for the step
			losses_str = f"{total_loss:.6f} :: {loss1:.6f}, {loss2:.6f}, {loss3:.6f}"
			optimizer_lr_str = f'{optimizer.param_groups[0]["lr"]:.3f}'
			logging.info(f"Step update | epoch: {epoch}, batch_idx: {batch_idx}, step: {step}, "
			             f"loss: {losses_str} | optimizer_lr: {optimizer_lr_str}")

			with torch.no_grad():
				preds = [F.log_softmax(op, dim=1) for op in output]
				true_pos_per_step = 0
				total_per_step = opts["batch_size"]
				preds = torch.stack([torch.argmax(pred, dim=1) for pred in preds], dim=1)
				for (pred, annot) in zip(preds, batch_ann_t):
					if torch.equal(pred, annot):
						true_pos_per_step += 1
				true_pos += true_pos_per_step
				total += total_per_step
				step_acc = true_pos_per_step / total_per_step * 100
				logging.info(f'Accuracy at step {step}: {step_acc}')

			step += 1

		# calculate per epoch
		lr_scheduler.step()
		epoch_acc = true_pos / total * 100
		logging.info(f'Accuracy at epoch {epoch}: {epoch_acc:.3f}')

		# calculate overall MAP@k
		if epoch % opts["mAP_k_print_interval"] == 0:
			obj1_score, rel_score, obj2_score = utils.calculate_training_mAPk(
				dataloader, model, train_ann_dict, opts)
			mAPk_scores_str = f'{obj1_score:.3f}, {rel_score:.3f}, {obj2_score:.3f}'
			logging.info(f'Epoch update | epoch: {epoch}, loss: {losses_str}, '
			             f'optimizer_lr_str: {optimizer_lr_str}, epoch_mAP@k: {mAPk_scores_str}')

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
				fh.write(f'{time_str} | epoch: {epoch}, loss: {losses_str}, '
				         f'accuracy: {epoch_acc:.3f}, epoch_mAP@k: {mAPk_scores_str} \n')

	return model


def main(opts):
	generate_optical_flow_images(opts, mode='train')
	model = model_provider(opts)
	if opts["resume"]:
		if os.path.isfile(opts["resume"]):
			logging.info(f'loading checkpoint {opts["resume"]}')
			checkpoint = torch.load(opts["resume"])
			model.load_state_dict(checkpoint['state_dict'])
			logging.info(f'loaded checkpoint {opts["resume"]}')
		else:
			raise RuntimeError(f'no checkpoint found at {opts["resume"]}')

	logging.info(model)
	optimizer = optim.Adam(model.parameters(), lr=opts["learning_rate"], weight_decay=opts["weight_decay"])
	exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opts["learning_rate_decay_every"],
	                                             gamma=opts["learning_rate_decay_rate"])
	vrdataset = VRDataset(img_root=opts["train_dataset_path"], img_root_alternate=os.path.join(opts['optical_flow_train_dataset_path'], opts['optical_flow_type']), len=opts["train_dataset_size"],
	                      transform=data_transformations(opts, mode='train'), transform_alternate=data_transformations(opts, mode='default'))
	dataloader = DataLoader(vrdataset, batch_size=opts["batch_size"], shuffle=opts["shuffle"],
	                        num_workers=opts["num_workers"])

	train(dataloader, model, optimizer, exp_lr_scheduler, opts)
	logging.info("Training completed")


if __name__ == '__main__':
	cli_opts = vars(model_options())
	cli_opts["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
	opt_json = os.path.join(cli_opts["checkpoint_path"], 'opt_info.json')
	if not os.path.isdir(cli_opts["checkpoint_path"]):
		os.mkdir(cli_opts["checkpoint_path"])

	logging.info(json.dumps(cli_opts, indent=4))
	logging.info(f'__Python VERSION: {sys.version}')
	logging.info(f'__pyTorch VERSION: {torch.__version__}')
	with open(opt_json, 'w') as fh:
		json.dump(cli_opts, fh, indent=4)

	utils.print_device(cli_opts)
	main(cli_opts)
