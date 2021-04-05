import os

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset


class VRDataset(Dataset):

	def __init__(self, vid_root, opf_root, n_samples, transform_vid=None, transform_opf=None):
		self.root_dir_vid = vid_root
		self.root_dir_opf = opf_root
		self.n_samples = n_samples
		self.transform_vid = transform_vid
		self.transform_opf = transform_opf

	def __len__(self):
		return self.n_samples

	def __getitem__(self, idx):
		dir_name = str(idx)
		while len(dir_name) < 6:
			dir_name = "0" + dir_name

		vid_path = os.path.join(self.root_dir_vid, dir_name)
		frame_names_vid = sorted(os.listdir(vid_path))
		images_vid = [Image.open(os.path.join(vid_path, frame_vid)) for frame_vid in frame_names_vid]
		if self.transform_vid:
			images_vid = [self.transform_vid(image_vid) for image_vid in images_vid]

		opf_path = os.path.join(self.root_dir_opf, dir_name)
		frame_names_opf = sorted(os.listdir(opf_path))
		images_opf = [Image.open(os.path.join(opf_path, frame_opf)) for frame_opf in frame_names_opf]
		if self.transform_opf:
			images_opf = [self.transform_opf(image_opf) for image_opf in images_opf]

		return dir_name, torch.stack(images_vid), torch.stack(images_opf)


class ImageDataset(Dataset):
	def __init__(self, img_root, n_samples):
		self.root_dir = img_root
		self.n_samples = n_samples

	def __len__(self):
		return self.n_samples

	def __getitem__(self, idx):
		dir_name = str(idx)
		while len(dir_name) < 6:
			dir_name = "0" + dir_name

		vid_path = os.path.join(self.root_dir, dir_name)
		frame_names = sorted(os.listdir(vid_path))
		images = [cv2.imread(os.path.join(vid_path, frame_name)) for frame_name in frame_names]
		return dir_name, images
