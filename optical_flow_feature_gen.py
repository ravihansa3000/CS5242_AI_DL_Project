import os
import cv2

from optical_flow import *
from model_config import model_options
from dataset import ImageDataset
from torch.utils.data import DataLoader

def padded_digit(idx):
	idx = str(idx)
	while len(idx) < 6:
		idx = "0" + idx
	return idx

def main(opts):
	imageDataset = ImageDataset(img_root=opts["train_dataset_path"], len=119)
	dataloader = DataLoader(imageDataset, batch_size=1, num_workers=opts["num_workers"])

	for batch_idx, (video_ids, video_frames) in enumerate(dataloader):
		video_folder_path = os.path.join(opts["optical_flow_dataset_path"], opts['optical_flow_type'], video_ids[0])
		# skip unneeded the optical flow feature generation
		if os.path.isdir(video_folder_path) and len(os.listdir(video_folder_path)) == len(video_frames):
			continue
		else:
			os.mkdir(video_folder_path)
		op_flow = OpticalFlowProvider(opts['optical_flow_type'])
		op_flow.set1stFrame(video_frames[0].squeeze().numpy())
		for frame_idx in range(len(video_frames)):
			video_frame = video_frames[frame_idx].squeeze().numpy()
			op_flow_img = op_flow.apply(video_frame)
			op_flow.set1stFrame(video_frame)
			cv2.imwrite(os.path.join(video_folder_path, padded_digit(frame_idx + 1) + '.jpg'), op_flow_img)


if __name__ == '__main__':
	opts = vars(model_options())
	if not os.path.isdir(opts["optical_flow_dataset_path"]):
		os.mkdir(opts["optical_flow_dataset_path"])
	if not os.path.isdir(os.path.join(opts["optical_flow_dataset_path"], opts['optical_flow_type'])):
		os.mkdir(os.path.join(opts["optical_flow_dataset_path"], opts['optical_flow_type']))
	main(opts)