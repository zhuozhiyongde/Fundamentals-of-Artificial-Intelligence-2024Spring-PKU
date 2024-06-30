#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   valid.py
# @Time    :   2024/06/30 18:34:46
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code

"""
valid.py: 筛选出不能增强的绿一色、推不倒番型的对局，保存到 cannot_enhance_matches.json中
"""

import json
import re

match_id = -1
cannot_enhance_matches = []
with open("data/data.txt", "r") as f:
    for line in f:
        if re.match("Match", line):
            match_id += 1
        if re.search("绿一色|推不倒", line):
            cannot_enhance_matches.append(match_id)

print("[Detected] ", len(cannot_enhance_matches))
with open("data/cannot_enhance_matches.json", "w") as f:
    json.dump(cannot_enhance_matches, f)
