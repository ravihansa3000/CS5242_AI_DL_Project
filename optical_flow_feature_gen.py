import cv2

from optical_flow import *
from model_config import model_options
from dataset import ImageDataset
from torch.utils.data import DataLoader

def image_name(idx):
	idx = str(idx)
	while len(idx) < 6:
		idx = "0" + idx
	return idx + ".jpg"

def transform_video_frame(video_frame):
	return video_frame.squeeze().numpy()

def main(opts):
	imageDataset = ImageDataset(img_root=opts["train_dataset_path"], len=119)
	dataloader = DataLoader(imageDataset, batch_size=1, num_workers=opts["num_workers"])

	for batch_idx, (video_ids, video_frames) in enumerate(dataloader):
		op_flow = optical_flow_provider('lucas_kanade')
		op_flow.set1stFrame(transform_video_frame(video_frames[0]))

		for frame_idx in range(len(video_frames) - 1):
			processed_img = op_flow.apply(transform_video_frame(video_frames[frame_idx + 1]))
			op_flow.set1stFrame(transform_video_frame(video_frames[frame_idx + 1]))
			cv2.imwrite(image_name(frame_idx), processed_img)
			
		return

if __name__ == '__main__':
	opts = vars(model_options())
	main(opts)