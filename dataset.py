import os
import torch
import cv2 

from PIL import Image
from torch.utils.data import Dataset


class VRDataset(Dataset):

	def __init__(self, img_root, img_root_alternate, len, transform=None, transform_alternate=None):
		self.root_dir = img_root
		self.root_dir_alternate = img_root_alternate
		self.len = len
		self.transform = transform
		self.transform_alternate = transform_alternate

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

		vid_path_alternate = os.path.join(self.root_dir_alternate, dir_name)
		frame_names_alternate = sorted(os.listdir(vid_path_alternate))
		images_alternate = [Image.open(os.path.join(vid_path_alternate, frame_name_alternate)) for frame_name_alternate in frame_names_alternate]
		if self.transform_alternate:		
			images_alternate = [self.transform_alternate(image_alternate) for image_alternate in images_alternate]

		return dir_name, torch.stack(images), torch.stack(images_alternate)


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
