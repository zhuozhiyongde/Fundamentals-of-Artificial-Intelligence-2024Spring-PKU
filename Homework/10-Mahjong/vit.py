#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   vit.py
# @Time    :   2024/06/30 18:41:31
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code

"""
vit.py: 尝试使用 ViT 模型进行训练
ref: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


# 辅助函数
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# 定义 FeedForward 类
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# 定义 Attention 类
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# 定义 Transformer 类
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# 定义 SelfVitModel 类
class SelfVitModel(nn.Module):
    def __init__(
        self,
        patches_num=56,
        num_classes=235,
        dim=768,
        depth=8,
        heads=8,
        mlp_dim=512,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super().__init__()

        # 定义patches的数量
        self.patches_num = patches_num
        patch_dim = 4 * 9  # 每个patch的维度是4x9

        # 将patches映射到指定的维度
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b n c1 c2 -> b n (c1 c2)", c1=4, c2=9),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, patches_num + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer模块
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 分类头
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, input_dict):
        # 获取特征图
        x = input_dict["obs"]["observation"].float()
        # print(x.size()) # 1024 56 4 9

        # 将特征图映射到patches
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # 添加分类token
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        # 通过Transformer
        x = self.transformer(x)

        # 使用分类token进行分类
        x = x[:, 0]
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return self.mlp_head(x) + inf_mask
