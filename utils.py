import logging
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')


def image_show(image):
	image = torchvision.utils.make_grid(image)
	image = image / 2 + 0.5  # unnormalize
	plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
	plt.show()


def save_checkpoint(state, filename='checkpoint.pth'):
	torch.save(state, filename)


def load_pretrained(model, filename, optimizer=None):
	if os.path.isfile(filename):
		print(f"loading checkpoint from file: {filename}")
		checkpoint = torch.load(filename)
		model.load_state_dict(checkpoint['state_dict'])
		if optimizer is not None:
			optimizer.load_state_dict(checkpoint['optimizer'])
			return model, optimizer, checkpoint['epoch']
		else:
			return model, checkpoint['epoch']
	else:
		print(f"Error! no checkpoint found file:{filename}")


def init_hidden(batch_size, n_frames, n_units):
	"""
	the weights are of the form (batch_size, n_units)
	note that batch_first=True does not affect the shape of hidden states
	:param batch_size:
	:param n_frames:
	:param n_units:
	:return:
	"""
	hidden_a = torch.randn(n_frames, batch_size, n_units)
	hidden_b = torch.randn(n_frames, batch_size, n_units)

	hidden_a = Variable(hidden_a).to(device)
	hidden_b = Variable(hidden_b).to(device)

	return hidden_a, hidden_b


def apk(actual, predicted, k=5):
	"""
	Computes the average precision at k.
	This function computes the average prescision at k between two lists of
	items.
	Parameters
	----------
	actual : list
			 A list of elements that are to be predicted (order doesn't matter)
	predicted : list
				A list of predicted elements (order does matter)
	k : int, optional
		The maximum number of predicted elements
	Returns
	-------
	score : double
			The average precision at k over the input lists
	"""
	if len(predicted) > k:
		predicted = predicted[:k]

	if not isinstance(actual, list):
		actual = [actual]

	score = 0.0
	num_hits = 0.0

	for i, p in enumerate(predicted):
		if p in actual and p not in predicted[:i]:
			num_hits += 1.0
			score += num_hits / (i + 1.0)

	if not actual:
		return 0.0

	return score / min(len(actual), k)


def mapk(actual, predicted, k=5):
	"""
	Computes the mean average precision at k.
	This function computes the mean average prescision at k between two lists
	of lists of items.
	Parameters
	----------
	actual : list
			 A list of lists of elements that are to be predicted
			 (order doesn't matter in the lists)
	predicted : list
				A list of lists of predicted elements
				(order matters in the lists)
	k : int, optional
		The maximum number of predicted elements
	Returns
	-------
	score : double
			The mean average precision at k over the input lists
	"""
	return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def calculate_mapk_batch(topk_preds, annotations, k=5):
	mAPk_scores = []
	for i in range(3):
		actual_list = annotations[:, i]
		predicted_list = topk_preds[i].tolist()
		mAPk = mapk(actual_list, predicted_list, k)
		mAPk_scores.append(mAPk)

	return mAPk_scores


def calculate_training_mAPk(dataloader, model, train_ann_dict, opts=None):
	mAPk_obj1_scores = []
	mAPk_rel_scores = []
	mAPk_obj2_scores = []
	for batch_idx, (video_ids, vid_tensor, opf_tensor) in enumerate(dataloader):
		vid_tensor = vid_tensor.to(device)
		opf_tensor = opf_tensor.to(device)
		batch_ann_t = [
			[train_ann_dict[item][0], train_ann_dict[item][1], train_ann_dict[item][2]] for item in video_ids
		]

		model.eval()
		with torch.no_grad():
			_, topk_preds_list = model(x_vid=vid_tensor, x_opf=opf_tensor, target_y=None, top_k=opts["mAP_k"])

			# calculate mean average precision
			mAPk_scores = calculate_mapk_batch(topk_preds_list, batch_ann_t, opts["mAP_k"])
			mAPk_obj1_scores.append(mAPk_scores[0])
			mAPk_rel_scores.append(mAPk_scores[1])
			mAPk_obj2_scores.append(mAPk_scores[2])

	return np.mean(mAPk_obj1_scores), np.mean(mAPk_rel_scores), np.mean(mAPk_obj2_scores)


def print_device(cli_opts):
	# https://discuss.pytorch.org/t/cuda-visible-devices-make-gpu-disappear/21439
	os.environ['CUDA_VISIBLE_DEVICES'] = cli_opts["gpu"]
	if cli_opts["gpu"]:
		logging.info(f'__CUDNN VERSION: {torch.backends.cudnn.version()}')
		logging.info(f'__Number CUDA Devices: {torch.cuda.device_count()}')
		subprocess.call(
			["nvidia-smi", "--format=csv",
			 "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
		logging.info(f'Available devices {torch.cuda.device_count()}')
		logging.info(f'Current CUDA Device: GPU {torch.cuda.current_device()}')
		logging.info(f'Current CUDA Device Name: {torch.cuda.get_device_name(int(cli_opts["gpu"]))}')
	return None
