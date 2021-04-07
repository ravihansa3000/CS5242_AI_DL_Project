import json
import sys

import pandas as pd
import torch

import utils
from dataset import VRDataset
from model_config import model_provider, data_transformations_vid, data_transformations_opf
from optical_flow import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')


def get_label_from_map(label_map, indices):
	if not isinstance(indices, list):
		indices = [indices]

	res = []
	for idx in indices:
		res.append(label_map[idx])
	return ",".join(res)


def eval(dataloader, model, opts):
	# load annotations labels if testing training data
	with open("data/test_annotation.json") as f:
		test_ann_dict = json.load(f)

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

	for batch_idx, (video_ids, vid_tensor, opf_tensor) in enumerate(dataloader):
		vid_tensor = vid_tensor.to(device)
		opf_tensor = opf_tensor.to(device)
		model.eval()
		with torch.no_grad():
			_, topk_preds = model(x_vid=vid_tensor, x_opf=opf_tensor, top_k=opts["mAP_k"])

			# calculate mean average precision
			batch_ann_arr = np.array([
				[test_ann_dict[item][0], test_ann_dict[item][1], test_ann_dict[item][2]] for item in video_ids
			])
			mAPk_scores_batch = utils.calculate_mapk_batch(topk_preds, batch_ann_arr, opts["mAP_k"])
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

			sample_ann_l = batch_ann_arr[vid_idx]
			ground_truth = f'{sample_ann_l[0]} | {sample_ann_l[1]} | {sample_ann_l[2]}'
			ground_truth_ann = f'{get_label_from_map(obj_idx_map, sample_ann_l[0])} | ' \
			                   f'{get_label_from_map(rel_idx_map, sample_ann_l[1])} | ' \
			                   f'{get_label_from_map(obj_idx_map, sample_ann_l[2])}'

			preds_ann_list.append([
				obj1_k_str, rel_k_str, obj2_k_str,
				obj1_ann_str, rel_ann_str, obj2_ann_str,
				ground_truth, ground_truth_ann
			])

	mAPk_scores_str = f'{np.mean(obj1_scores):.3f}, {np.mean(rel_scores):.3f}, {np.mean(obj2_scores):.3f}'
	logging.info(f'eval mAPk_scores: {mAPk_scores_str}')
	preds_ann_df_cols = [
		'object1', 'relationship', 'object2',
		'obj1_ann', 'rel_ann', 'obj2_ann',
		'ground_truth', 'ground_truth_ann'
	]

	preds_sub_df = pd.DataFrame(preds_sub_list, columns=['label'])
	preds_ann_df = pd.DataFrame(preds_ann_list, columns=preds_ann_df_cols)
	preds_sub_df.index.name = preds_ann_df.index.name = 'ID'
	preds_sub_df.to_csv('model_run_data/eval_preds_sub.csv', index_label="ID")
	preds_ann_df.to_csv('model_run_data/eval_preds_ann.csv', index_label="ID")


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

	vrdataset = VRDataset(
		vid_root=opts["test_dataset_path"],
		opf_root=os.path.join(opts['optical_flow_test_dataset_path'], opts['optical_flow_type']),
		n_samples=opts["test_dataset_size"],
		transform_vid=data_transformations_vid(opts, mode='test'),
		transform_opf=data_transformations_opf(opts)
	)
	dataloader = DataLoader(vrdataset, batch_size=opts["batch_size"], shuffle=False, num_workers=opts["num_workers"])

	eval(dataloader, model, opts)
	logging.info("Evaluation completed")


if __name__ == '__main__':
	cli_opts = vars(model_options())
	main(cli_opts)
