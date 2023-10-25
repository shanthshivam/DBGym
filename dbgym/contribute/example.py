"""
resnet.py
Residual Network module.
"""

import copy
import time

import torch
from torch import Tensor, nn
from yacs.config import CfgNode

from dbgym.db import Tabular
from dbgym.loss import compute_loss
from dbgym.register import register


class ResNet(nn.Module):
    """
    Residual Network module.
    """

    def __init__(self, cfg: CfgNode, data: Tabular):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        output_dim = cfg.model.output_dim
        self.layer = cfg.model.layer

        max_values = torch.max(data.x_d, dim=0).values
        self.embeddings = nn.ModuleList([
            nn.Embedding(max_values[i] + 1, hidden_dim)
            for i in range(data.x_d.size(1))
        ])
        self.linear = nn.Linear(data.x_c.shape[1], hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(0, self.layer - 1):
            norm = nn.LayerNorm(hidden_dim)
            layer1 = nn.Linear(hidden_dim, hidden_dim)
            layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layers.append(nn.ModuleList([norm, layer1, layer2]))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, data: Tabular) -> Tensor:
        """
        Forward propagation function

        Args:
        - data: tabular data

        Returns:
        - out: output data
        """

        x_d = [
            self.embeddings[j](data.x_d[:, j]) for j in range(data.x_d.size(1))
        ]
        x_d = torch.sum(torch.stack(x_d, dim=2), dim=2)
        x = x_d + self.linear(data.x_c)

        for i in range(0, self.layer - 1):
            z = self.layers[i][0](x)
            z = self.layers[i][1](z)
            z = nn.functional.relu(z)
            z = self.layers[i][2](z)
            z = nn.functional.relu(z)
            x = x + z

        x = self.layers[-1](x)
        return x


register('tabular_model', 'ResNet', ResNet)


def train(dataset, model, optimizer, scheduler, cfg: CfgNode):
    '''
    The training function
    '''

    data = dataset.to(torch.device(cfg.device))
    y = data.y
    mask = dataset.mask

    # statistics
    s = {}
    if cfg.model.output_dim > 1:
        s['metric'] = 'Accuracy'
    elif cfg.model.output_dim == 1:
        s['metric'] = 'Mean Squared Error'
    # flag
    f = cfg.model.output_dim > 1

    for epoch in range(cfg.train.epoch):
        model.train()
        optimizer.zero_grad()
        output = model(data)
        target = y.squeeze()
        result = {}
        losses = {}
        loss, score = compute_loss(cfg, output[mask['train']],
                                   target[mask['train']])
        loss.backward()
        val_loss, val_score = compute_loss(cfg, output[mask['valid']],
                                           target[mask['valid']])
        test_loss, test_score = compute_loss(cfg, output[mask['test']],
                                             target[mask['test']])
        optimizer.step()
        scheduler.step()

        if epoch == 0 or (val_score - s['valid']) * (f * 2 - 1) > 0:
            s['best_epoch'] = epoch
            s['train_loss'] = loss.item()
            s['train'] = score
            s['valid_loss'] = val_loss.item()
            s['valid'] = val_score
            s['test_loss'] = test_loss.item()
            s['test'] = test_score
            s['model'] = copy.deepcopy(model)
            s['pred'] = output[mask['all']].detach()
            if f:
                s['pred'] = s['pred'].argmax(dim=1)

        result['Train'] = score
        losses['Train'] = loss.item()
        result['Valid'] = val_score
        losses['Valid'] = val_loss.item()
        result['Test'] = test_score
        losses['Test'] = test_loss.item()

    return s


register('train', 'train_simple', train)


def add_new_config(config: CfgNode):
    # example argument
    config.example_arg = 'example'
    # example argument group
    config.example_group = CfgNode()
    # then argument can be specified within the group
    config.example_group.example_arg = 'example'


register('config', 'example_config', add_new_config)
