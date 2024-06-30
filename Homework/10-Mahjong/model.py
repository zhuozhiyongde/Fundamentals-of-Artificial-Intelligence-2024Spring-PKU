#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   model.py
# @Time    :   2024/06/30 18:40:46
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code

"""
model.py: 用于定义神经网络模型
"""

import torch
import torch.nn.functional as F
from torch import nn


class SelfModel(nn.Module):
    def __init__(self, hidden=128, obs_dim=56):
        nn.Module.__init__(self)
        self.hidden = hidden
        self.obs_dim = obs_dim
        # self.vec_dim = vec_dim
        self._input_layer = nn.Sequential(
            nn.Conv2d(obs_dim, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        num_blocks = 20
        self._hidden_layers = nn.ModuleList(
            [self.res_block(self.hidden) for _ in range(num_blocks)]
        )

        self._output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 9, 256),
            nn.GELU(),
            nn.Linear(256, 235),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def res_block(self, hidden=256):
        return nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
        )

    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()

        x = self._input_layer(obs)
        for block in self._hidden_layers:
            x = x + block(x)
            x = F.gelu(x)
        x = self._output_layer(x)

        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return x + inf_mask


class SelfVecModel(nn.Module):
    def __init__(self, obs_dim, vec_dim, hidden=128, num_blocks=20):
        nn.Module.__init__(self)
        self.hidden = hidden
        self.obs_dim = obs_dim
        self.vec_dim = vec_dim
        self._input_layer = nn.Sequential(
            nn.Conv2d(obs_dim, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self._hidden_layers = nn.ModuleList(
            [self.res_block(self.hidden) for _ in range(num_blocks)]
        )

        down_sample_ratio = ((hidden * 4 * 9 + vec_dim) / 235) ** (1 / 3)
        down_sample_dim_1 = int((hidden * 4 * 9 + vec_dim) / down_sample_ratio)
        down_sample_dim_1 = down_sample_dim_1 // 8 * 8
        down_sample_dim_2 = int(down_sample_dim_1 / down_sample_ratio)
        down_sample_dim_2 = down_sample_dim_2 // 8 * 8

        self._output_layer = nn.Sequential(
            nn.Linear(hidden * 4 * 9 + vec_dim, down_sample_dim_1),
            nn.GELU(),
            nn.Linear(down_sample_dim_1, down_sample_dim_2),
            nn.GELU(),
            nn.Linear(down_sample_dim_2, 235),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def res_block(self, hidden=256):
        return nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
        )

    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        vec = input_dict["obs"]["vec"].float()
        x = self._input_layer(obs)
        for block in self._hidden_layers:
            x = x + block(x)
            x = F.gelu(x)
        # 展平, 128*4*9 = 4608
        x = torch.flatten(x, start_dim=1)
        # 链接 x 和 vec, 4608 + 117 = 4725
        x = torch.cat([x, vec], dim=1)
        # FC
        x = self._output_layer(x)

        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return x + inf_mask
