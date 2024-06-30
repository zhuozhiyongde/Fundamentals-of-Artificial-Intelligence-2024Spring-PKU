#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   agent.py
# @Time    :   2024/06/30 19:05:31
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code


"""
agent.py: 接口类，描述了与Botzone交互的智能体的行为。在 FeatureAgent 中进一步实现。
也即，按顺序接收一个玩家在对局中观察到的事件，并在每个决策点整理出状态特征；将网络输出的动作转为事件。
"""


class MahjongGBAgent:
    def __init__(self, seatWind):
        pass

    """
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
    Player N(me) Play XX
    Player N(me) BuGang XX
    Player N(not me) Peng
    Player N(not me) Chi XX
    Player N(me) UnPeng
    Player N(me) UnChi XX
    
    Player N Hu
    Huang
    Player N Invalid
    Draw XX
    Player N(not me) Play XX
    Player N(not me) BuGang XX
    Player N(me) Peng
    Player N(me) Chi XX
    """

    def request2obs(self, request):
        pass

    """
    Hu
    Play XX
    (An)Gang XX
    BuGang XX
    Gang
    Peng
    Chi XX
    Pass
    """

    def action2response(self, action):
        pass
