import os

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset


class VRDataset(Dataset):

	def __init__(self, img_root, len, transform=None):
		self.root_dir = img_root
		self.len = len
		self.transform = transform

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		dir_name = str(idx)
		while len(dir_name) < 6:
			dir_name = "0" + dir_name

		vid_path = os.path.join(self.root_dir, dir_name)
		frame_names = sorted(os.listdir(vid_path))
		images = [Image.open(os.path.join(vid_path, frame_name)) for frame_name in frame_names]
		if self.transform:
			images = [self.transform(image) for image in images]

		return dir_name, torch.stack(images)


class ImageDataset(Dataset):
	def __init__(self, img_root, len):
		self.root_dir = img_root
		self.len = len

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		dir_name = str(idx)
		while len(dir_name) < 6:
			dir_name = "0" + dir_name

		vid_path = os.path.join(self.root_dir, dir_name)
		frame_names = sorted(os.listdir(vid_path))
		images = [cv2.imread(os.path.join(vid_path, frame_name)) for frame_name in frame_names]
		return dir_name, images
