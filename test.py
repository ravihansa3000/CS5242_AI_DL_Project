import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import utils
from dataset import VRDataset
from model_config import model_options, model_provider, data_transformations

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')


def test(dataloader, encoder, decoder, opts):
	preds_res_list = []
	preds_ann_list = []
	obj1_scores = []
	rel_scores = []
	obj2_scores = []
	annotations_t = None

	# load annotations labels if testing training data
	with open(opts["train_annotation_path"]) as f:
		train_ann = json.load(f)

	# load annotations entities
	with open("data/object1_object2.json") as f:
		object_types = json.load(f)
		obj_idx_map = dict((v, k) for k, v in object_types.items())

	with open("data/relationship.json") as f:
		rel_types = json.load(f)
		rel_idx_map = dict((v, k) for k, v in rel_types.items())

	for batch_idx, (video_ids, videos_tensor) in enumerate(dataloader):
		videos_tensor = videos_tensor.to(device)
		if opts["data_split"] == "train":
			annotations_t = torch.LongTensor(
				[[train_ann[item][0], train_ann[item][1], train_ann[item][2]]
				 for item in video_ids]
			).to(device)

		encoder.eval()
		decoder.eval()

		with torch.no_grad():
			vid_imgs_encoded = []
			for i in range(videos_tensor.shape[0]):
				vid_imgs_encoded.append(encoder(videos_tensor[i]))

			vid_feats = torch.stack(vid_imgs_encoded, dim=0)  # vid_feats: (batch_size, n_frames, dim_vid)
			_, topk_preds = decoder(
				vid_feats,
				target_variable=annotations_t,
				tf_mode=False,
				top_k=opts["mAP_k"]
			)

			# accumulate predictions in <id>:<label> format
			for i in range(opts["batch_size"]):
				sample_obj1 = topk_preds[0][i].tolist()
				sample_rel = topk_preds[1][i].tolist()
				sample_obj2 = topk_preds[2][i].tolist()

				obj1_str = " ".join(str(x) for x in sample_obj1)
				rel_str = " ".join(str(x) for x in sample_rel)
				obj2_str = " ".join(str(x) for x in sample_obj2)

				preds_res_list.append(obj1_str)
				preds_res_list.append(rel_str)
				preds_res_list.append(obj2_str)

				obj1_ann = " ".join(obj_idx_map[x] for x in sample_obj1)
				rel_ann = " ".join(rel_idx_map[x] for x in sample_rel)
				obj2_ann = " ".join(obj_idx_map[x] for x in sample_obj2)

				video_item = video_ids[i]
				ground_truth = f'{train_ann[video_item][0]}--{train_ann[video_item][1]}--{train_ann[video_item][2]}'
				ground_truth_ann = f'{obj_idx_map[train_ann[video_item][0]]}--{rel_idx_map[train_ann[video_item][1]]}' \
				                   f'--{obj_idx_map[train_ann[video_item][2]]}'

				preds_ann_list.append([obj1_str, rel_str, obj2_str,
				                       obj1_ann, rel_ann, obj2_ann,
				                       ground_truth, ground_truth_ann])

			# calculate mean average precision if testing training data
			if opts["data_split"] == "train":
				mAPk_scores = utils.calculate_mapk_batch(topk_preds, annotations_t, opts["mAP_k"])
				obj1_scores.append(mAPk_scores[0])
				rel_scores.append(mAPk_scores[1])
				obj2_scores.append(mAPk_scores[2])
				mAPk_scores_str = f'{np.mean(obj1_scores):.3f}, {np.mean(rel_scores):.3f}, {np.mean(obj2_scores):.3f}'
				logging.info(f'Inferences generated for batch_idx: {batch_idx}, mAPk_scores: {mAPk_scores_str}')
			else:
				logging.info(f'Inferences generated for batch_idx: {batch_idx}')

	preds_res_df = pd.DataFrame(preds_res_list, columns=['label'])
	preds_ann_df = pd.DataFrame(preds_ann_list,
	                            columns=['object1', 'relationship', 'object2',
	                                     'obj1_ann', 'rel_ann', 'obj2_ann',
	                                     'ground_truth', 'ground_truth_ann'])
	preds_res_df.index.name = preds_ann_df.index.name = 'ID'

	preds_res_df.to_csv('model_run_data/preds_res.csv', index_label="ID")
	preds_ann_df.to_csv('model_run_data/preds_ann.csv', index_label="ID")


def main(opts):
	encoder, decoder = model_provider(opts)

	logging.info(json.dumps(cli_opts, indent=4))
	logging.info(f'__Python VERSION: {sys.version}')
	logging.info(f'__pyTorch VERSION: {torch.__version__}')

	# load trained model
	if os.path.isfile(opts["trained_model"]):
		logging.info(f'loading trained model {opts["trained_model"]}')
		encoder.load_state_dict(torch.load(opts['trained_model'], map_location=device)['encoder_state_dict'])
		decoder.load_state_dict(torch.load(opts['trained_model'], map_location=device)['decoder_state_dict'])
		logging.info(f'loaded trained model {opts["trained_model"]}')
	else:
		raise RuntimeError(f'no trained model found at {opts["trained_model"]}')

	vrdataset = VRDataset(img_root=opts["test_dataset_path"], len=opts["test_dataset_size"],
	                      transform=data_transformations(opts))
	dataloader = DataLoader(vrdataset, batch_size=opts["batch_size"], num_workers=opts["num_workers"], shuffle=False)

	test(dataloader, encoder, decoder, opts)


if __name__ == '__main__':
	cli_opts = vars(model_options())
	main(cli_opts)
