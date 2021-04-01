import logging
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

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


def calculate_mapk_batch(topk_preds, annotations_t, k=5):
	mAPk_scores = []
	for i in range(3):
		actual_list = annotations_t[:, i].tolist()
		predicted_list = topk_preds[i].tolist()
		mAPk = mapk(actual_list, predicted_list, k)
		mAPk_scores.append(mAPk)

	return mAPk_scores


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
