import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import torchvision


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
