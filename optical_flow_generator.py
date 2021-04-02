from optical_flow_provider import *

def to_optical_flow_images(video_frames, opts):
	of_images = []
	of = OpticalFlowProvider(opts['optical_flow_type'])
	of.set1stFrame(video_frames[0].squeeze().numpy())
	for frame_idx in range(len(video_frames)):
		video_frame = video_frames[frame_idx].squeeze().numpy()
		of_images.append(of.apply(video_frame))
		of.set1stFrame(video_frame)
	return of_images