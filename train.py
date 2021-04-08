import json
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

import utils
from dataset import VRDataset
from model_config import model_provider, data_transformations_vid, data_transformations_opf
from optical_flow import *
from utils import save_checkpoint
from eval import eval, get_eval_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')


def train(dataloader, model, optimizer, lr_scheduler, eval_dataloader, opts):
	loss_fns = [torch.nn.CrossEntropyLoss().to(device) for _ in range(3)]
	with open(opts["train_annotation_path"]) as f:
		train_ann_dict = json.load(f)

	logging.info(f"Starting training at {opts['start_epoch']} epochs and will run for {opts['end_epoch']} epochs "
				 f"using device: {device}")

	for epoch in range(opts["start_epoch"], opts["end_epoch"]):
		epoch_count = epoch + 1
		step = 0
		true_pos = 0
		total = 0
		losses_str = ''
		optimizer_lr_str = ''
		mAPk_scores_str = 'N/A'

		for batch_idx, (video_ids, vid_tensor, opf_tensor) in enumerate(dataloader):
			# zero the parameter gradients
			model.train()
			optimizer.zero_grad()

			vid_tensor = vid_tensor.to(device)
			opf_tensor = opf_tensor.to(device)

			batch_ann_t = torch.LongTensor([
				[train_ann_dict[item][0], train_ann_dict[item][1], train_ann_dict[item][2]]
				for item in video_ids
			]).to(device)
			output, _ = model(x_vid=vid_tensor, x_opf=opf_tensor, target_y=batch_ann_t, tf_rate=opts["tf_rate"])

			# calculate loss for each element in a batch prediction
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
			optimizer_lr_str = f'{lr_scheduler.get_last_lr()[0]:.6f}'

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

			logging.info(f"Step update | epoch_count: {epoch_count}, batch_idx: {batch_idx}, step: {step}, "
						 f"loss: {losses_str}, step_acc: {step_acc} | optimizer_lr: {optimizer_lr_str}")
			step += 1

		# calculate per epoch
		lr_scheduler.step()
		epoch_acc = true_pos / total * 100

		# calculate overall MAP@k for eval data
		if epoch_count % opts["eval_print_interval"] == 0:
			eval(eval_dataloader, model, opts)

		# calculate overall MAP@k for train data
		if epoch_count % opts["train_print_interval"] == 0:
			obj1_score, rel_score, obj2_score = utils.calculate_training_mAPk(dataloader, model, train_ann_dict, opts)
			mAPk_scores_str = f'{obj1_score:.3f}, {rel_score:.3f}, {obj2_score:.3f}'

		# write epoch status update to file
		model_info_path = os.path.join(opts["checkpoint_path"], 'score.txt')
		with open(model_info_path, 'a') as fh:
			time_str = time.strftime("%Y-%m-%d %H:%M:%S")
			fh.write(f'{time_str} | epoch_count: {epoch_count}, loss: {losses_str}, optimizer_lr: {optimizer_lr_str}, '
					 f'accuracy: {epoch_acc:.3f}, mAP@k: {mAPk_scores_str} \n')

		# save the model at every checkpointing milestone OR if the loss is lower than in the previous epoch
		if epoch_count % opts["save_checkpoint_every"] == 0:
			save_file_path = os.path.join(opts["checkpoint_path"], f"model_{epoch_count}.pth")
			save_checkpoint({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict()
			}, filename=save_file_path)
			logging.info(f"Model saved to {save_file_path}")

		logging.info(
			f'Epoch update | epoch_count: {epoch_count}, loss: {losses_str}, optimizer_lr: {optimizer_lr_str}, '
			f'accuracy: {epoch_acc:.3f}, mAP@k: {mAPk_scores_str}')

	return model


def main(opts):
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
	optimizer = optim.Adam(
		model.parameters(),
		lr=opts["learning_rate"],
		weight_decay=opts["weight_decay"]
	)

	# set scheduler last_epoch in case resuming from a previous run
	if opts["start_epoch"] == 0:
		last_epoch = -1
	else:
		last_epoch = opts["start_epoch"]

	lr_scheduler = optim.lr_scheduler.StepLR(
		optimizer,
		step_size=opts["learning_rate_decay_every"],
		gamma=opts["learning_rate_decay_rate"],
		last_epoch=last_epoch
	)
	vrdataset = VRDataset(
		vid_root=opts["train_dataset_path"],
		opf_root=os.path.join(opts['optical_flow_train_dataset_path'], opts['optical_flow_type']),
		n_samples=opts["train_dataset_size"],
		transform_vid=data_transformations_vid(opts, mode='train'),
		transform_opf=data_transformations_opf(opts)
	)
	dataloader = DataLoader(
		vrdataset,
		batch_size=opts["batch_size"],
		shuffle=opts["shuffle"],
		num_workers=opts["num_workers"]
	)

	eval_dataloader = get_eval_dataloader(opts)
	train(dataloader, model, optimizer, lr_scheduler, eval_dataloader, opts)
	logging.info("Training completed")


if __name__ == '__main__':
	cli_opts = vars(model_options())
	cli_opts["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
	opt_json = os.path.join(cli_opts["checkpoint_path"], 'opt_info.json')
	if not os.path.isdir(cli_opts["checkpoint_path"]):
		os.mkdir(cli_opts["checkpoint_path"])

	logging.info(json.dumps(cli_opts, indent=4))
	logging.info(f'__PID: {os.getpid()}')
	logging.info(f'__Python VERSION: {sys.version}')
	logging.info(f'__pyTorch VERSION: {torch.__version__}')
	with open(opt_json, 'w') as fh:
		json.dump(cli_opts, fh, indent=4)

	utils.print_device(cli_opts)
	main(cli_opts)
