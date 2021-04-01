import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from dataset import VRDataset
from model_config import model_options, model_provider, data_transformations
from utils import save_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')


def calculate_training_mAPk(dataloader, encoder, decoder, training_annotation, opts=None):
	mAPk_obj1_scores = []
	mAPk_rel_scores = []
	mAPk_obj2_scores = []
	for batch_idx, (video_ids, videos_tensor) in enumerate(dataloader):
		videos_tensor = videos_tensor.to(device)
		annotations_t = torch.LongTensor(
			[[training_annotation[item][0], training_annotation[item][1], training_annotation[item][2]]
			 for item in video_ids]
		).to(device)

		encoder.eval()
		decoder.eval()
		with torch.no_grad():
			vid_imgs_encoded = []
			for i in range(videos_tensor.shape[0]):
				vid_imgs_encoded.append(encoder(videos_tensor[i]))

			vid_feats = torch.stack(vid_imgs_encoded, dim=0)  # vid_feats: (batch_size, n_frames, dim_vid)
			_, topk_preds_list = decoder(vid_feats, top_k=opts["mAP_k"])

			# calculate mean average precision
			mAPk_scores = utils.calculate_mapk_batch(topk_preds_list, annotations_t, opts["mAP_k"])
			mAPk_obj1_scores.append(mAPk_scores[0])
			mAPk_rel_scores.append(mAPk_scores[1])
			mAPk_obj2_scores.append(mAPk_scores[2])

	return mAPk_obj1_scores, mAPk_rel_scores, mAPk_obj2_scores


def train_batch(epoch, batch_idx, step, video_ids, videos_tensor, encoder, decoder, encoder_optimizer,
                decoder_optimizer, training_annotation, tf_mode=True, opts=None):
	videos_tensor = videos_tensor.to(device)
	# add 35 (no of objects) offset to separate <object> and <relationship> embeddings
	annotations_t = torch.LongTensor(
		[[training_annotation[item][0], training_annotation[item][1] + 35, training_annotation[item][2]]
		 for item in video_ids]
	).to(device)

	encoder.train()
	decoder.train()
	encoder.zero_grad()
	decoder.zero_grad()

	vid_imgs_encoded = []
	for i in range(videos_tensor.shape[0]):
		vid_imgs_encoded.append(encoder(videos_tensor[i]))

	vid_feats = torch.stack(vid_imgs_encoded, dim=0)  # vid_feats: (batch_size, n_frames, dim_vid)
	decoder_outputs, _ = decoder(vid_feats, target_variable=annotations_t, tf_mode=tf_mode)

	# de-offset <relationship> annotations when calculating loss since linear output has 35 nodes
	annotations_t[:, 1] = torch.sub(annotations_t[:, 1], 35)
	loss1 = F.cross_entropy(decoder_outputs[0], annotations_t[:, 0])
	loss2 = F.cross_entropy(decoder_outputs[1], annotations_t[:, 1])
	loss3 = F.cross_entropy(decoder_outputs[2], annotations_t[:, 2])
	total_loss = loss1 + loss2 + loss3
	losses_str = f"{total_loss:.6f} :: {loss1:.6f}, {loss2:.6f}, {loss3:.6f}"
	total_loss.backward()

	enc_lr = encoder_optimizer.param_groups[0]["lr"]
	dec_lr = decoder_optimizer.param_groups[0]["lr"]
	logging.info(f'LR update | enc_lr: {enc_lr:.6f}, dec_lr: {dec_lr:.6f}')

	torch.nn.utils.clip_grad_norm_(encoder.parameters(), opts["grad_clip"])
	torch.nn.utils.clip_grad_norm_(decoder.parameters(), opts["grad_clip"])
	encoder_optimizer.step()
	decoder_optimizer.step()

	seq_probs_list = [F.log_softmax(decoder_outputs[i], 1) for i in range(3)]
	topk_preds = [torch.topk(item, k=opts["mAP_k"], dim=1)[1] for item in seq_probs_list]
	mAPk_scores = utils.calculate_mapk_batch(topk_preds, annotations_t, opts["mAP_k"])
	mAPk_scores_str = f'{np.mean(mAPk_scores[0]):.3f}, {np.mean(mAPk_scores[1]):.3f}, {np.mean(mAPk_scores[2]):.3f}'
	logging.info(f'Step update | epoch: {epoch}, batch_idx: {batch_idx}, '
	             f'step: {step}, step_mAP@{opts["mAP_k"]}: {mAPk_scores_str}, losses: {losses_str}')

	step += 1
	return seq_probs_list, mAPk_scores, losses_str


def train(dataloader, encoder, decoder, encoder_optimizer, enc_lr_scheduler, decoder_optimizer, dec_lr_scheduler, opts):
	# load annotations labels
	with open(opts["train_annotation_path"]) as f:
		training_annotation = json.load(f)

	logging.info(f"Starting training at {opts['start_epoch']} epochs and will run for {opts['end_epoch']} epochs "
	             f"using device: {device}")

	for epoch in range(opts["start_epoch"], opts["end_epoch"]):
		step = 0
		losses_str = None

		# disable teacher forcing after epoch milestone has elapsed
		tf_mode = False if epoch > opts["disable_tf_after_epoch"] else True

		for batch_idx, (video_ids, videos_tensor) in enumerate(dataloader):
			_, _, losses_str = train_batch(epoch=epoch, batch_idx=batch_idx, step=step, video_ids=video_ids,
			                               videos_tensor=videos_tensor, encoder=encoder, decoder=decoder,
			                               encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer,
			                               training_annotation=training_annotation, tf_mode=tf_mode, opts=opts)

		# update step in scheduler per epoch
		enc_lr_scheduler.step()
		dec_lr_scheduler.step()

		# calculate overall MAP@k
		obj1_scores, rel_scores, obj2_scores = calculate_training_mAPk(dataloader, encoder, decoder,
		                                                               training_annotation, opts)
		mAPk_scores_str = f'{np.mean(obj1_scores):.3f}, {np.mean(rel_scores):.3f}, {np.mean(obj2_scores):.3f}'
		logging.info(f'Epoch update | epoch: {epoch}, epoch_mAP@{opts["mAP_k"]}: {mAPk_scores_str}')

		if epoch % opts["save_checkpoint_every"] == 0:
			save_file_path = os.path.join(opts["checkpoint_path"], f"model_{epoch}.pth")
			save_checkpoint({
				'epoch': epoch,
				'encoder_state_dict': encoder.state_dict(),
				'decoder_state_dict': decoder.state_dict(),
				'encoder_optimizer': encoder_optimizer.state_dict(),
				'decoder_optimizer': decoder_optimizer.state_dict()
			}, filename=save_file_path)
			logging.info(f"Model saved to {save_file_path}")

			model_info_path = os.path.join(opts["checkpoint_path"], 'score.txt')
			with open(model_info_path, 'a') as fh:
				time_str = time.strftime("%Y-%m-%d %H:%M:%S")
				enc_lr = encoder_optimizer.param_groups[0]["lr"]
				dec_lr = decoder_optimizer.param_groups[0]["lr"]
				fh.write(f"{time_str} | epoch: {epoch}, loss: {losses_str}, mAPk_scores: {mAPk_scores_str} "
				         f"| enc_lr: {enc_lr:.6f}, dec_lr: {dec_lr:.6f} \n")

	logging.info("Training completed")


def main(opts):
	encoder, decoder = model_provider(opts)

	encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=opts["weight_decay"])
	enc_exp_lr_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer, step_size=3,
	                                                 gamma=0.9)

	decoder_optimizer = optim.Adam(decoder.parameters(), lr=opts["learning_rate"], weight_decay=opts["weight_decay"])
	dec_exp_lr_scheduler = optim.lr_scheduler.StepLR(decoder_optimizer, step_size=opts["learning_rate_decay_every"],
	                                                 gamma=opts["learning_rate_decay_rate"])

	# load saved model if resume opt is set
	if opts["resume"]:
		if os.path.isfile(opts["resume"]):
			logging.info(f'loading checkpoint {opts["resume"]}')
			checkpoint = torch.load(opts["resume"])
			encoder.load_state_dict(checkpoint['encoder_state_dict'], map_location=device)
			decoder.load_state_dict(checkpoint['decoder_state_dict'], map_location=device)
			logging.info(f'loaded checkpoint {opts["resume"]}')
		else:
			raise RuntimeError(f'no checkpoint found at {opts["resume"]}')

	vrdataset = VRDataset(img_root=opts["train_dataset_path"], len=opts["train_dataset_size"],
	                      transform=data_transformations(opts))
	dataloader = DataLoader(vrdataset, batch_size=opts["batch_size"], shuffle=opts["shuffle"],
	                        num_workers=opts["num_workers"])
	train(dataloader, encoder, decoder, encoder_optimizer, enc_exp_lr_scheduler,
	      decoder_optimizer, dec_exp_lr_scheduler, opts)


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
