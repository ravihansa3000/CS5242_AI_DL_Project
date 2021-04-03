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


def test(dataloader, model, opts):
	# load annotations labels if testing training data
	with open(opts["train_annotation_path"]) as f:
		train_ann_dict = json.load(f)

	# load annotations entities
	with open("data/object1_object2.json") as f:
		object_types = json.load(f)
		obj_idx_map = dict((v, k) for k, v in object_types.items())

	with open("data/relationship.json") as f:
		rel_types = json.load(f)
		rel_idx_map = dict((v, k) for k, v in rel_types.items())

	obj1_scores = []
	rel_scores = []
	obj2_scores = []
	preds_sub_list = []  # store results for final submission csv
	preds_ann_list = []  # store results with ground truth labels of train data (when testing train or eval data)

	for batch_idx, (video_ids, videos_tensor) in enumerate(dataloader):
		videos_tensor = videos_tensor.to(device)
		model.eval()
		with torch.no_grad():
			_, topk_preds = model(x=videos_tensor, top_k=opts["mAP_k"])

			# calculate mean average precision if testing training data
			if opts["data_split"] in ["train", "eval"]:
				batch_ann_t = torch.LongTensor([
					[train_ann_dict[item][0], train_ann_dict[item][1], train_ann_dict[item][2]]
					for item in video_ids
				]).to(device)
				mAPk_scores_batch = utils.calculate_mapk_batch(topk_preds, batch_ann_t, opts["mAP_k"])
				obj1_scores.append(mAPk_scores_batch[0])
				rel_scores.append(mAPk_scores_batch[1])
				obj2_scores.append(mAPk_scores_batch[2])

		# organize predictions per sample in <id>:<label> format
		for vid_idx in range(len(video_ids)):
			sample_obj1_k = topk_preds[0][vid_idx].tolist()
			sample_rel_k = topk_preds[1][vid_idx].tolist()
			sample_obj2_k = topk_preds[2][vid_idx].tolist()

			obj1_k_str = " ".join(str(x) for x in sample_obj1_k)
			rel_k_str = " ".join(str(x) for x in sample_rel_k)
			obj2_k_str = " ".join(str(x) for x in sample_obj2_k)

			preds_sub_list.append(obj1_k_str)
			preds_sub_list.append(rel_k_str)
			preds_sub_list.append(obj2_k_str)

			obj1_ann_str = " ".join(obj_idx_map[x] for x in sample_obj1_k)
			rel_ann_str = " ".join(rel_idx_map[x] for x in sample_rel_k)
			obj2_ann_str = " ".join(obj_idx_map[x] for x in sample_obj2_k)

			if opts["data_split"] in ["train", "eval"]:
				sample_ann_l = batch_ann_t[vid_idx].tolist()
				ground_truth = f'{sample_ann_l[0]}--{sample_ann_l[1]}--{sample_ann_l[2]}'
				ground_truth_ann = f'{obj_idx_map[sample_ann_l[0]]}--{rel_idx_map[sample_ann_l[1]]}' \
				                   f'--{obj_idx_map[sample_ann_l[2]]}'

				preds_ann_list.append([obj1_k_str, rel_k_str, obj2_k_str, obj1_ann_str, rel_ann_str, obj2_ann_str,
				                       ground_truth, ground_truth_ann])
			else:
				preds_ann_list.append([obj1_k_str, rel_k_str, obj2_k_str, obj1_ann_str, rel_ann_str, obj2_ann_str])

		logging.info(f'Inferences generated for video_ids: {video_ids}')

	logging.info(f'Inferences generated for all test data')
	if opts["data_split"] in ["train", "eval"]:
		mAPk_scores_str = f'{np.mean(obj1_scores):.3f}, {np.mean(rel_scores):.3f}, {np.mean(obj2_scores):.3f}'
		logging.info(f'{opts["data_split"]} mAPk_scores: {mAPk_scores_str}')
		preds_ann_df_cols = ['object1', 'relationship', 'object2', 'obj1_ann', 'rel_ann', 'obj2_ann',
		                     'ground_truth', 'ground_truth_ann']
	else:
		preds_ann_df_cols = ['object1', 'relationship', 'object2', 'obj1_ann', 'rel_ann', 'obj2_ann']

	preds_sub_df = pd.DataFrame(preds_sub_list, columns=['label'])
	preds_ann_df = pd.DataFrame(preds_ann_list, columns=preds_ann_df_cols)
	preds_sub_df.index.name = preds_ann_df.index.name = 'ID'
	preds_sub_df.to_csv('model_run_data/preds_sub.csv', index_label="ID")
	preds_ann_df.to_csv('model_run_data/preds_ann.csv', index_label="ID")


def main(opts):
	model = model_provider(opts)

	logging.info(json.dumps(cli_opts, indent=4))
	logging.info(f'__Python VERSION: {sys.version}')
	logging.info(f'__pyTorch VERSION: {torch.__version__}')

	if os.path.isfile(opts["trained_model"]):
		logging.info(f'loading trained model {opts["trained_model"]}')
		model.load_state_dict(torch.load(opts['trained_model'], map_location=device)['state_dict'])
		logging.info(f"model: {model}")
		logging.info(f'loaded trained model {opts["trained_model"]}')
	else:
		raise RuntimeError(f'no trained model found at {opts["trained_model"]}')

	if opts["data_split"] in ["train", "eval"]:
		opts["test_dataset_path"] = opts["train_dataset_path"]

	vrdataset = VRDataset(img_root=opts["test_dataset_path"], len=opts["test_dataset_size"],
	                      transform=data_transformations(opts, mode='test'))
	dataloader = DataLoader(vrdataset, batch_size=opts["batch_size"], shuffle=False, num_workers=opts["num_workers"])

	test(dataloader, model, opts)
	logging.info("Testing completed")


if __name__ == '__main__':
	cli_opts = vars(model_options())
	main(cli_opts)
