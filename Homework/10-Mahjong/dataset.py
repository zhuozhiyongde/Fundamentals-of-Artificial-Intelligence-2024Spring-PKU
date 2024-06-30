#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   dataset.py
# @Time    :   2024/06/30 18:39:23
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code


"""
dataset.py: 将预处理好的数据集一次性加载到内存中，按全局索引获取状态动作对。
args:
- lazy: 是否懒加载
- augment: 是否随机数据增强
"""

import json
from bisect import bisect_right

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class MahjongGBDataset(Dataset):
    def __init__(
        self, folder="data", begin=0, end=1, tqdm_disable=0, augment=False, lazy=False
    ):
        with open("data/data-count-back.json") as f:
            self.match_samples = json.load(f)
        # 总比赛数
        self.total_matches = len(self.match_samples)
        # 总步数
        self.total_samples = sum(self.match_samples)
        # 切片采样
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin : self.end]
        # 重新计算总比赛数和总步数
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.folder = folder
        self.augment = augment
        self.lazy = lazy
        # 将各个局各自的步数改为总步数偏移
        # Example: [23, 45, 64] -> [0, 23, 68]
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        # cache 是一个字典，存储了所有的数据
        self.cache = {"obs": [], "mask": [], "vec": [], "act": []}

        if self.augment:
            with open("data/cannot_enhance_matches.json") as f:
                cannot_enhance_matches = json.load(f)
                self.cannot_enhance_matches = cannot_enhance_matches
        if self.lazy:
            return

        # 正常加载数据到内存中
        for i in tqdm(
            range(self.matches),
            desc="loading data".ljust(20),
            bar_format="{l_bar}{bar:40}{r_bar}",
            disable=bool(tqdm_disable),
        ):
            d = np.load("%s/%d.npz" % (self.folder, i + self.begin))
            for k in d:
                self.cache[k].append(d[k])

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]

        if self.lazy:
            match_data = np.load("%s/%d.npz" % (self.folder, match_id + self.begin))
            obs = match_data["obs"][sample_id]
            mask = match_data["mask"][sample_id]
            vec = match_data["vec"][sample_id]
            act = match_data["act"][sample_id]

        else:
            obs = self.cache["obs"][match_id][sample_id]
            mask = self.cache["mask"][match_id][sample_id]
            vec = self.cache["vec"][match_id][sample_id]
            act = self.cache["act"][match_id][sample_id]

        if self.augment and match_id not in self.cannot_enhance_matches:
            perm = np.random.permutation(3)
            a = np.array([0, 1])
            b = np.concatenate(
                [np.arange(0, 27).reshape(3, -1)[perm].flatten(), np.arange(27, 34)]
            )
            enhanced_vec_index = np.concatenate(
                [np.arange(0, 13), b + 13, [47], b + 48, [82], b + 83]
            )
            b = np.tile(b, (5, 1))
            c = (np.arange(0, 5) * 34 + 2).reshape(-1, 1)
            b += c
            b = b.flatten()
            e = np.arange(172, 235).reshape(3, -1)[perm].flatten()
            enhanced_mask_index = np.concatenate([a, b, e])
            enhanced_act = np.where(enhanced_mask_index == act)[0][0]

            return (
                obs[:, [*perm, -1]],
                mask[enhanced_mask_index],
                vec[enhanced_vec_index],
                enhanced_act,
            )

        return (
            obs,
            mask,
            vec,
            act,
        )
