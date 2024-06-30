#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   preprocess.py
# @Time    :   2024/06/30 18:34:35
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code

"""
preprocess.py: 数据预处理，对于每一局数据，建四个FeatureAgent（作为四个玩家视角），将对局中的事件传给相应的agent，整理出每个人每个决策点下的特征表示以及实际选择的动作，保存到data文件夹下。
"""

import json
import os
import re

import numpy as np
from feature import FeatureAgent


def filterData():
    global obs
    global actions
    newobs = [[] for i in range(4)]
    newactions = [[] for i in range(4)]
    for i in range(4):
        for j, o in enumerate(obs[i]):
            if (
                o["action_mask"].sum() > 1
            ):  # ignore states with single valid action (Pass)
                newobs[i].append(o)
                newactions[i].append(actions[i][j])
    obs = newobs
    actions = newactions


def saveData():
    assert [len(x) for x in obs] == [
        len(x) for x in actions
    ], "obs actions not matching!"
    l.append(sum([len(x) for x in obs]))
    np.savez(
        "%s/%d.npz" % (output_dir, matchid),
        obs=np.stack([x["observation"] for i in range(4) for x in obs[i]]).astype(
            np.int8
        ),
        mask=np.stack([x["action_mask"] for i in range(4) for x in obs[i]]).astype(
            np.int8
        ),
        vec=np.stack([x["vec"] for i in range(4) for x in obs[i]]).astype(np.float16),
        act=np.array([x for i in range(4) for x in actions[i]]),
    )
    for x in obs:
        x.clear()
    for x in actions:
        x.clear()


def process_data(file_path, start_line, end_line, offset, cpu_id):
    global obs, actions, matchid, l
    obs = [[] for i in range(4)]
    actions = [[] for i in range(4)]
    matchid = offset - 1
    l = []

    with open(file_path, encoding="UTF-8") as f:
        for _ in range(start_line):
            f.readline()  # Skip lines until start_line
        line = f.readline()
        assert line.startswith("Match"), "Not a match start line"
        line_number = start_line
        # print(start_line)
        if end_line is None:
            end_line = 1e9
        # print(line)
        while line and line_number <= end_line:
            t = line.split()
            if len(t) == 0:
                line = f.readline()
                line_number += 1
                continue
            if t[0] == "Match":
                matchid += 1
                agents = [FeatureAgent(i) for i in range(4)]
            elif t[0] == "Wind":
                for agent in agents:
                    agent.request2obs(line)
            elif t[0] == "Player":
                p = int(t[1])
                if t[2] == "Deal":
                    agents[p].request2obs(" ".join(t[2:]))
                elif t[2] == "Draw":
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs(" ".join(t[2:])))
                            actions[p].append(0)
                        else:
                            agents[i].request2obs(" ".join(t[:3]))
                elif t[2] == "Play":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action(" ".join(t[2:])))
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs(line)
                        else:
                            obs[i].append(agents[i].request2obs(line))
                            actions[i].append(0)
                    curTile = t[3]
                elif t[2] == "Chi":
                    actions[p].pop()
                    actions[p].append(
                        agents[p].response2action("Chi %s %s" % (curTile, t[3]))
                    )
                    for i in range(4):
                        if i == p:
                            obs[p].append(
                                agents[p].request2obs("Player %d Chi %s" % (p, t[3]))
                            )
                            actions[p].append(0)
                        else:
                            agents[i].request2obs("Player %d Chi %s" % (p, t[3]))
                elif t[2] == "Peng":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("Peng %s" % t[3]))
                    for i in range(4):
                        if i == p:
                            obs[p].append(
                                agents[p].request2obs("Player %d Peng %s" % (p, t[3]))
                            )
                            actions[p].append(0)
                        else:
                            agents[i].request2obs("Player %d Peng %s" % (p, t[3]))
                elif t[2] == "Gang":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("Gang %s" % t[3]))
                    for i in range(4):
                        agents[i].request2obs("Player %d Gang %s" % (p, t[3]))
                elif t[2] == "AnGang":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("AnGang %s" % t[3]))
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs("Player %d AnGang %s" % (p, t[3]))
                        else:
                            agents[i].request2obs("Player %d AnGang" % p)
                elif t[2] == "BuGang":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("BuGang %s" % t[3]))
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs("Player %d BuGang %s" % (p, t[3]))
                        else:
                            obs[i].append(
                                agents[i].request2obs("Player %d BuGang %s" % (p, t[3]))
                            )
                            actions[i].append(0)
                elif t[2] == "Hu":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("Hu"))
                # Deal with Ignore clause
                if t[2] in ["Peng", "Gang", "Hu"]:
                    for k in range(5, 15, 5):
                        if len(t) > k:
                            p = int(t[k + 1])
                            if t[k + 2] == "Chi":
                                actions[p].pop()
                                actions[p].append(
                                    agents[p].response2action(
                                        "Chi %s %s" % (curTile, t[k + 3])
                                    )
                                )
                            elif t[k + 2] == "Peng":
                                actions[p].pop()
                                actions[p].append(
                                    agents[p].response2action("Peng %s" % t[k + 3])
                                )
                            elif t[k + 2] == "Gang":
                                actions[p].pop()
                                actions[p].append(
                                    agents[p].response2action("Gang %s" % t[k + 3])
                                )
                            elif t[k + 2] == "Hu":
                                actions[p].pop()
                                actions[p].append(agents[p].response2action("Hu"))
                        else:
                            break

            elif t[0] == "Score":
                filterData()
                saveData()
            line = f.readline()
            line_number += 1
    with open(f"{output_dir}/count-{cpu_id}.json", "w") as f:
        json.dump(l, f)


output_dir = "data-output"

# python preprocess.py {file_path} {start_line} {end_line} {offset} {cpu_id}
if __name__ == "__main__":
    import sys

    file_path = sys.argv[1]
    start_line = int(sys.argv[2])
    if sys.argv[3] == "None":
        end_line = None
    else:
        end_line = int(sys.argv[3])
    offset = int(sys.argv[4])
    cpu_id = int(sys.argv[5])
    output_dir = sys.argv[6]
    process_data(file_path, start_line, end_line, offset, cpu_id)
