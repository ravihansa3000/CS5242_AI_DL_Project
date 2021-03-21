import json
import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import S2VTModel


def train(loader, model, optimizer, lr_scheduler, opt):
    model.train()

    for epoch in range(opt["epochs"]):
        lr_scheduler.step()

        iteration = 0

        for data in loader:
            pass


def main(opt):
    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)

    model = S2VTModel(
        opt["vocab_size"],
        opt["max_len"],
        opt["dim_hidden"],
        opt["dim_word"],
        opt['dim_vid'],
        rnn_cell=opt['rnn_type'],
        n_layers=opt['num_layers'],
        rnn_dropout_p=opt["rnn_dropout_p"])

    model = model.cuda()

    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])

    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(dataloader, model, optimizer, exp_lr_scheduler, opt)
