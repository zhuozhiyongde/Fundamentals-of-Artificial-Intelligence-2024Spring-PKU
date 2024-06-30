"""
将预处理好的数据集一次性加载到内存中，按全局索引获取状态动作对。

提示：可以修改这个类，从而支持：
1. 延迟加载数据（特征复杂后数据量特别大，没法装进内存）；
2. 数据增强。
"""

from torch.utils.data import Dataset
import numpy as np
from bisect import bisect_right

class MahjongGBDataset(Dataset):
    
    def __init__(self, begin = 0, end = 1, augment = False):
        import json
        with open('data/count.json') as f:
            self.match_samples = json.load(f)
        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin : self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.augment = augment
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        self.cache = {'obs': [], 'mask': [], 'act': []}
        for i in range(self.matches):
            if i % 128 == 0: print('loading', i)
            d = np.load('data/%d.npz' % (i + self.begin))
            for k in d:
                self.cache[k].append(d[k])
    
    def __len__(self):
        return self.samples
    
    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        return self.cache['obs'][match_id][sample_id], self.cache['mask'][match_id][sample_id], self.cache['act'][match_id][sample_id]