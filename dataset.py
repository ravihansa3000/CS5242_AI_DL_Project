import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rgb_frames(img_dir, vid, start, num):
	frames = []
	for i in range(start, start + num):
		img = cv2.imread(os.path.join(img_dir, vid, str(i).zfill(6) + '.jpg'))[:, :, [2, 1, 0]]
		w, h, c = img.shape
		if w < 226 or h < 226:
			d = 226. - min(w, h)
			sc = 1 + d / min(w, h)
			img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
		img = (img / 255.) * 2 - 1
		frames.append(img)
	return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
	frames = []
	for i in range(start, start + num):
		imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
		imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

		w, h = imgx.shape
		if w < 224 or h < 224:
			d = 224. - min(w, h)
			sc = 1 + d / min(w, h)
			imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
			imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

		imgx = (imgx / 255.) * 2 - 1
		imgy = (imgy / 255.) * 2 - 1
		img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
		frames.append(img)
	return np.asarray(frames, dtype=np.float32)


def video_to_tensor(pic):
	"""Convert a ``numpy.ndarray`` to tensor.
	Converts a numpy.ndarray (T x H x W x C)
	to a torch.FloatTensor of shape (C x T x H x W)

	Args:
		 pic (numpy.ndarray): Video to be converted to tensor.
	Returns:
		 Tensor: Converted video.
	"""
	return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


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
		vid_dir_name = str(idx).zfill(6)
		vid_path = os.path.join(self.root_dir_vid, vid_dir_name)
		frame_names_vid = sorted(os.listdir(vid_path))
		images_vid = [Image.open(os.path.join(vid_path, frame_vid)) for frame_vid in frame_names_vid]
		if self.transform_vid:
			images_vid = [self.transform_vid(image_vid) for image_vid in images_vid]

		images_opf = load_rgb_frames(self.root_dir_vid, vid_dir_name, 1, 30)
		if self.transform_opf:
			images_opf = self.transform_opf(images_opf)

		return vid_dir_name, torch.stack(images_vid), video_to_tensor(images_opf)


class ImageDataset(Dataset):
	def __init__(self, img_root, n_samples):
		self.root_dir = img_root
		self.n_samples = n_samples

	def __len__(self):
		return self.n_samples

	def __getitem__(self, idx):
		dir_name = str(idx).zfill(6)
		vid_path = os.path.join(self.root_dir, dir_name)
		frame_names = sorted(os.listdir(vid_path))
		images = [cv2.imread(os.path.join(vid_path, frame_name)) for frame_name in frame_names]
		return dir_name, images
