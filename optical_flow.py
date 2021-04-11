import logging
import os

import cv2
from torch.utils.data import DataLoader

from dataset import ImageDataset
from model_config import model_options
from optical_flow_provider import *

logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')


def to_optical_flow_images(video_frames, opts):
	of_images = []
	of = OpticalFlowProvider(opts['optical_flow_type'])
	of.set1stFrame(video_frames[0].squeeze().numpy())
	for frame_idx in range(len(video_frames)):
		video_frame = video_frames[frame_idx].squeeze().numpy()
		of_images.append(of.apply(video_frame))
	return of_images


def padded_digit(idx):
	idx = str(idx)
	while len(idx) < 6:
		idx = "0" + idx
	return idx


def generate_optical_flow_images(opts, data_split):
	if data_split == 'train':
		dataset_path = opts['train_dataset_path']
		dataset_size = opts['train_dataset_size']
		optical_flow_dataset_path = opts['optical_flow_train_dataset_path']
	elif data_split == 'test':
		dataset_path = opts['test_dataset_path']
		dataset_size = opts['test_dataset_size']
		optical_flow_dataset_path = opts['optical_flow_test_dataset_path']
	else:
		raise RuntimeError(f"Invalid data_split: {data_split}")

	if not os.path.isdir(optical_flow_dataset_path):
		os.mkdir(optical_flow_dataset_path)
	if not os.path.isdir(os.path.join(optical_flow_dataset_path, opts['optical_flow_type'])):
		os.mkdir(os.path.join(optical_flow_dataset_path, opts['optical_flow_type']))

	imageDataset = ImageDataset(img_root=dataset_path, n_samples=dataset_size)
	dataloader = DataLoader(imageDataset, batch_size=1)

	for batch_idx, (video_ids, video_frames) in enumerate(dataloader):
		video_folder_path = os.path.join(optical_flow_dataset_path, opts['optical_flow_type'], video_ids[0])

		if not os.path.isdir(video_folder_path):
			os.mkdir(video_folder_path)

		of_images = to_optical_flow_images(video_frames, opts)
		for idx, frame in enumerate(of_images):
			cv2.imwrite(os.path.join(video_folder_path, padded_digit(idx + 1) + '_0.jpg'), frame[:, :, 0])
			cv2.imwrite(os.path.join(video_folder_path, padded_digit(idx + 1) + '_1.jpg'), frame[:, :, 1])
		logging.info(f'{opts["optical_flow_type"]} optical flow image generated on video_ids: {video_ids}')


if __name__ == '__main__':
	cli_opts = vars(model_options())
	generate_optical_flow_images(cli_opts, data_split=cli_opts["data_split"])
