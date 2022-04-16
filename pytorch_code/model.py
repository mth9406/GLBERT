#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from layers import * 

class GLBert4Rec(nn.Module):

    def __init__(self, opt, n_items):
        super().__init__()
        self.n_items = n_items # the number of items
        self.N = opt.N # the number of layers to be repeated..
        self.hidden_dim = opt.hidden_dim
        self.num_head = opt.num_head
        self.inner_dim = opt.inner_dim
        self.max_length = opt.max_length
        self.embedding = nn.Embedding(num_embeddings= n_items, embedding_dim= opt.hidden_dim, padding_idx= 0)
        self.pos_embedding = nn.Embedding(opt.max_length, opt.hidden_dim)
        self.graph_in_conv_layers = nn.ModuleList(
            [nn.Linear(opt.hidden_dim, opt.hidden_dim) for _ in range(opt.N)]
        )
        self.graph_out_conv_layers = nn.ModuleList(
            [nn.Linear(opt.hidden_dim, opt.hidden_dim) for _ in range(opt.N)]
        )
        self.graph_in_out_mix_conv_layers = nn.ModuleList(
            [nn.Linear(2*opt.hidden_dim, opt.hidden_dim) for _ in range(opt.N)]
        )
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(opt.hidden_dim, opt.num_head, opt.inner_dim) for _ in range(opt.N)]
        )
        self.projection = nn.Sequential(
            nn.Linear(opt.hidden_dim, opt.hidden_dim),
            nn.Linear(opt.hidden_dim, n_items)
        )
        self.dropout = nn.Dropout(0.1)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.batch_size = opt.batchSize
    

    def forward(self, input, A):
        device= input.device
        bs, item_len = input.shape[:2] # 0, 1
        mask = makeMask(input, option = 'padding').to(device)
        pos = torch.arange(0, item_len).unsqueeze(0).repeat(bs, 1).to(device)
        # (bs, item_len)
        # [[0, 1, 2, ..., item_len-1],...,[0, 1, 2, ..., item_len-1]]

        # Embedding layer
        output = self.dropout(self.embedding(input) + self.pos_embedding(pos))
        # (bs, item_len, hidden_dim)
        # Encoder layers
        for graph_in_conv_layer, graph_out_conv_layer, graph_in_out_mix_conv_layer, enc_layer \
            in zip(self.graph_in_conv_layers, self.graph_out_conv_layers, self.graph_in_out_mix_conv_layers, self.enc_layers):
            output_in = torch.matmul(A[:, :, :A.shape[1]], graph_in_conv_layer(output))
            output_out = torch.matmul(A[:, :, A.shape[1]:2*A.shape[1]], graph_out_conv_layer(output))
            output = graph_in_out_mix_conv_layer(torch.cat([output_in, output_out], 2))
            output = enc_layer(output, mask)
        # (bs, item_len, hidden_dim)
        output = output[:, -1, :] # (bs, hidden_dim)
        output = self.projection(output)
        return output

    # def compute_scores(self, hidden, mask):
    #     scores =  self.projection(hidden[:, -1, :])
    #     return scores

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    # get = lambda i: hidden[i][alias_inputs[i]]
    # seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, hidden

def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
