import os
import cv2
import logging

from optical_flow_generator import *
from model_config import model_options
from dataset import ImageDataset
from torch.utils.data import DataLoader

logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')


def padded_digit(idx):
	idx = str(idx)
	while len(idx) < 6:
		idx = "0" + idx
	return idx

def generate_optical_flow_images(opts, mode):
	dataset_path = None
	dataset_len = None
	optical_flow_dataset_path = None
	if mode == 'train':
		dataset_path = opts['train_dataset_path']
		dataset_len = opts['train_dataset_len']
		optical_flow_dataset_path = opts['optical_flow_train_dataset_path']
	else:
		dataset_path = opts['test_dataset_path']
		dataset_len = opts['test_dataset_len']
		optical_flow_dataset_path = opts['optical_flow_test_dataset_path']

	if not os.path.isdir(optical_flow_dataset_path):
		os.mkdir(optical_flow_dataset_path)
	if not os.path.isdir(os.path.join(optical_flow_dataset_path, opts['optical_flow_type'])):
		os.mkdir(os.path.join(optical_flow_dataset_path, opts['optical_flow_type']))

	imageDataset = ImageDataset(img_root=dataset_path, len=dataset_len)
	dataloader = DataLoader(imageDataset, batch_size=1, num_workers=opts["num_workers"])

	for batch_idx, (video_ids, video_frames) in enumerate(dataloader):
		video_folder_path = os.path.join(optical_flow_dataset_path, opts['optical_flow_type'], video_ids[0])
		
		if os.path.isdir(video_folder_path): 
			# skip optical flow feature generation
			if len(os.listdir(video_folder_path)) == len(video_frames):
				logging.info(f'{opts["optical_flow_type"]} optical flow skip image generation on video {",".join(video_ids)}...')
				continue
		else:
			os.mkdir(video_folder_path)

		of_images = to_optical_flow_images(video_frames, opts)
		for frame_idx in range(len(of_images)):
			cv2.imwrite(os.path.join(video_folder_path, padded_digit(frame_idx + 1) + '.jpg'), of_images[frame_idx])

		logging.info(f'{opts["optical_flow_type"]} optical flow image generated on video {",".join(video_ids)}...')


if __name__ == '__main__':
	opts = vars(model_options())
	generate_optical_flow_images(opts, mode='train')