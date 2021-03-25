import sys
import json
import os
import argparse
import time
import subprocess
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from S2VTModel import S2VTModel
from dataset import VRDataset
from utils import save_checkpoint
from model_config import model_options, model_provider, data_transformations

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')


def test(dataloader, model):
	preds = []
	for batch_idx, (video_ids, videos_tensor) in enumerate(dataloader):
		model.eval()
		videos_tensor = videos_tensor.to(device)
		with torch.no_grad():
			output, _ = model(x=videos_tensor)
			_, indices = torch.topk(output, k=5, dim=1)
			preds.append(indices.tolist())

	preds_df = pd.DataFrame(preds, columns=['object1', 'relationship', 'object2'])
	preds_df.index = np.arange(1, len(preds_df) + 1)
	preds_res = []
	for pred in preds_df.stack():
		preds_res.append(" ".join(map(str, pred)))
	preds_res_df = pd.DataFrame(preds_res, columns=['label'])
	preds_res_df.index.name = 'ID'
	preds_res_df.index = np.arange(1, len(preds_res_df) + 1)
	return preds_df, preds_res_df

def main(opts):
	model = model_provider(opts)
	vrdataset = VRDataset(img_root=opts["test_dataset_path"], len=119, transform=data_transformations(opts))
	dataloader = DataLoader(vrdataset, batch_size=1, num_workers=opts["num_workers"])
	if os.path.isfile(opts["trained_model"]):
		logging.info(f'loading trained model {opts["trained_model"]}')
		model.load_state_dict(torch.load(opts['trained_model'], map_location=device)['state_dict'])
		logging.info(f'loaded trained model {opts["trained_model"]}')
		_, preds_res_df = test(dataloader, model)
		preds_res_df.to_csv('preds.csv', index_label="ID")
	else:
		logging.info(f'no trained model found at {opts["trained_model"]}')   

if __name__ == '__main__':
	opts = vars(model_options())
	main(opts)