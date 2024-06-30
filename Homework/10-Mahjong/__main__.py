#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   __main__.py
# @Time    :   2024/06/30 22:50:29
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code

"""
__main__.py: 与Botzone交互的Bot代码。
将Botzone的输入整理后交给Agent类处理，得到状态特征作为网络输入，网络输出的动作再转成字符串，进一步转成Botzone输出格式。
"""

# Agent part
import sys

# Botzone interaction
import numpy as np
import torch
from feature import FeatureAgent

# Model part
from model import SelfVecModel


def obs2response(model, obs):
    logits = model(
        {
            "is_training": False,
            "obs": {
                "observation": torch.from_numpy(np.expand_dims(obs["observation"], 0)),
                "vec": torch.from_numpy(np.expand_dims(obs["vec"], 0)),
                "action_mask": torch.from_numpy(np.expand_dims(obs["action_mask"], 0)),
            },
        }
    )
    action = logits.detach().numpy().flatten().argmax()
    response = agent.action2response(action)
    return response


if __name__ == "__main__":
    model = SelfVecModel(obs_dim=FeatureAgent.OBS_SIZE, vec_dim=FeatureAgent.VEC_SIZE)
    data_dir = "/data/vec-fix-act-128.pkl"
    # data_dir = "./vec-fix-act-128.pkl"
    model.load_state_dict(torch.load(data_dir, map_location=torch.device("cpu")))
    input()  # 1
    while True:
        request = input()
        while not request.strip():
            request = input()
        t = request.split()
        if t[0] == "0":
            seatWind = int(t[1])
            agent = FeatureAgent(seatWind)
            agent.request2obs("Wind %s" % t[2])
            print("PASS")
        elif t[0] == "1":
            agent.request2obs(" ".join(["Deal", *t[5:]]))
            print("PASS")
        elif t[0] == "2":
            obs = agent.request2obs("Draw %s" % t[1])
            response = obs2response(model, obs)
            t = response.split()
            if t[0] == "Hu":
                print("HU")
            elif t[0] == "Play":
                print("PLAY %s" % t[1])
            elif t[0] == "Gang":
                print("GANG %s" % t[1])
                angang = t[1]
            elif t[0] == "BuGang":
                print("BUGANG %s" % t[1])
        elif t[0] == "3":
            p = int(t[1])
            if t[2] == "DRAW":
                agent.request2obs("Player %d Draw" % p)
                zimo = True
                print("PASS")
            elif t[2] == "GANG":
                if p == seatWind and angang:
                    agent.request2obs("Player %d AnGang %s" % (p, angang))
                elif zimo:
                    agent.request2obs("Player %d AnGang" % p)
                else:
                    agent.request2obs("Player %d Gang" % p)
                print("PASS")
            elif t[2] == "BUGANG":
                obs = agent.request2obs("Player %d BuGang %s" % (p, t[3]))
                if p == seatWind:
                    print("PASS")
                else:
                    response = obs2response(model, obs)
                    if response == "Hu":
                        print("HU")
                    else:
                        print("PASS")
            else:
                zimo = False
                if t[2] == "CHI":
                    agent.request2obs("Player %d Chi %s" % (p, t[3]))
                elif t[2] == "PENG":
                    agent.request2obs("Player %d Peng" % p)
                obs = agent.request2obs("Player %d Play %s" % (p, t[-1]))
                if p == seatWind:
                    print("PASS")
                else:
                    response = obs2response(model, obs)
                    t = response.split()
                    if t[0] == "Hu":
                        print("HU")
                    elif t[0] == "Pass":
                        print("PASS")
                    elif t[0] == "Gang":
                        print("GANG")
                        angang = None
                    elif t[0] in ("Peng", "Chi"):
                        obs = agent.request2obs("Player %d " % seatWind + response)
                        response2 = obs2response(model, obs)
                        print(" ".join([t[0].upper(), *t[1:], response2.split()[-1]]))
                        agent.request2obs("Player %d Un" % seatWind + response)
        print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
        sys.stdout.flush()
