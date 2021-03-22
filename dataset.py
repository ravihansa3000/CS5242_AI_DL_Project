from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch, os
import numpy as np
from PIL import Image


class VRDataset(Dataset):

	def __init__(self, img_root, transform=None):
		self.root_dir = img_root
		self.transform = transform

	def __len__(self):
		return 447  # temoporarily hardcoded.

	def __getitem__(self, idx):
		dir_name = str(idx)
		while len(dir_name) < 6:
			dir_name = "0" + dir_name

		vid_path = os.path.join(self.root_dir, dir_name)
		frame_names = sorted(os.listdir(vid_path))
		images = [Image.open(os.path.join(vid_path, frame_name)) for frame_name in frame_names]
		if self.transform:
			images = [self.transform(image) for image in images]

		ret = {
			'video_id': dir_name,
			'frames': images
		}
		return ret
