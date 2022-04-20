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
        # self.graph_in_conv_layers = nn.ModuleList(
        #     [nn.Linear(opt.hidden_dim, opt.hidden_dim) for _ in range(opt.N)]
        # )
        # self.graph_out_conv_layers = nn.ModuleList(
        #     [nn.Linear(opt.hidden_dim, opt.hidden_dim) for _ in range(opt.N)]
        # )
        self.graph_in_out_mix_conv_layers = nn.ModuleList(
            [nn.Linear(2*opt.hidden_dim, opt.hidden_dim) for _ in range(opt.N)]
        )
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(opt.hidden_dim, opt.num_head, opt.inner_dim) for _ in range(opt.N)]
        )
        self.projection_att = nn.Linear(opt.hidden_dim, 1)
        # to obtain a global attention

        self.projection_sess = nn.Linear(2*opt.hidden_dim, opt.hidden_dim)
        # to obtain a session representation
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

        # Encoder layers
        # for graph_in_conv_layer, graph_out_conv_layer, graph_in_out_mix_conv_layer, enc_layer \
        #     in zip(self.graph_in_conv_layers, self.graph_out_conv_layers, self.graph_in_out_mix_conv_layers, self.enc_layers):
        #     output_in = torch.matmul(A[:, :, :A.shape[1]], graph_in_conv_layer(output))
        #     output_out = torch.matmul(A[:, :, A.shape[1]:2*A.shape[1]], graph_out_conv_layer(output))
        #     output = graph_in_out_mix_conv_layer(torch.cat([output_in, output_out], 2))
        #     # (bs, item_len, hidden_dim)
        #     output = enc_layer(output)
        # output_local = output[:, -1, :] 
        for graph_in_out_mix_conv_layer, enc_layer \
            in zip(self.graph_in_out_mix_conv_layers, self.enc_layers):
            output_in = torch.matmul(A[:, :, :A.shape[1]], output)
            output_out = torch.matmul(A[:, :, A.shape[1]:2*A.shape[1]], output)
            output = graph_in_out_mix_conv_layer(torch.cat([output_in, output_out], 2))
            # (bs, item_len, hidden_dim)
            output = enc_layer(output)
        output_local = output[:, -1, :] 
        # (bs, hidden_dim)
        # to represent user's current interest

        att = self.projection_att(output)
        att = att / att.sum(dim= 1, keepdim= True)
        # (bs, item_len, 1)
        output = output.permute(0,2,1)
        # (bs, hidden_dim, item_len)
        output = (output @ att).squeeze() 
        # (bs, hidden_dim)

        # output_global
        output = torch.cat([output_local, output], dim= 1)  
        del output_local
        # (bs, 2*hidden_dim)      

        output = self.projection_sess(output)
        # seesion embedding
        # (bs, hidden_dim)
        output = output.unsqueeze(1)
        # (bs, 1, hidden_dim)
        # obtain scores
        output = output@(self.embedding.weight[1:].T)
        # (bs, 1, hidden_dim) @ (hidden_dim, n_items)
        # (bs, 1, V)
        output = output.squeeze()
        # (bs, n_items)

        return output
    # def compute_scores(self, hidden, mask):
    #     scores =  self.projection(hidden[:, -1, :])
    #     return scores

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data, device):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = (torch.Tensor(alias_inputs).long()).to(device)
    items = (torch.Tensor(items).long()).to(device)
    A = (torch.Tensor(A).float()).to(device)
    mask = (torch.Tensor(mask).long()).to(device)
    hidden = model(items, A)
    # get = lambda i: hidden[i][alias_inputs[i]]
    # seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, hidden

def train_test(model, train_data, test_data, device):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data, device)
        targets = (torch.Tensor(targets).long()).to(device)
        loss = model.loss_function(scores, targets - 1)
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
        targets, scores = forward(model, i, test_data, device)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
