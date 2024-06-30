#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   feature.py
# @Time    :   2024/06/30 18:43:22
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code

"""
feature.py: 在模拟对局中按照麻将规则构建出每个决策点的所有可行动作以及特征表示，用于训练模型。
"""

from collections import defaultdict

import numpy as np
from agent import MahjongGBAgent
from MahjongGB import (
    HonorsAndKnittedTilesShanten,
    KnittedStraightShanten,
    MahjongFanCalculator,
    RegularShanten,
    SevenPairsShanten,
    ThirteenOrphansShanten,
)


class FeatureAgent(MahjongGBAgent):
    """
    observation: 60*4*9
        (men+quan+hand4)*4*9
        前两个维度表示门风和场风，后四个维度表示手牌信息。
        `OBS` 先是被组织为 60*36 的二维张量格式，随后 self._obs() 返回时，会被转换为 60*4*9 的三维张量格式。
        36个维度依次为 W(1-9, 万) T(1-9, 筒) B(1-9, 饼) [F(1-4, 风) J(1-3, 箭)]
        60这个通道数的详细说明见后续定义处。
        风向的记录依照如下规则：
        0 1 2 3 -> 东 南 西 北

    vec: 117
        定义见后。

    action_mask: 235
        235 = pass1 + hu1 + discard34 + gang34 + angang34 + bugang34 + peng34 + chi63(3*7*3)
        235 = 过 1 + 胡 1 + 弃牌 34 + 明杠 34 + 暗杠 34 + 补杠 34 + 碰牌 34 + 吃牌 63
        其中吃牌 63 = 花色万条饼3 * 中心牌二到八7 * 吃三张中的第几张3
    """

    class OFFSET_OBS:
        KE = 0  # +4, 玩家副露信息-刻子，4 玩家
        SHUN = 4  # +4, 玩家副露信息-顺子, 4 玩家
        GANG = 8  # +4, 玩家副露信息-杠子, 4 玩家
        ANGANG = 12  # +4, 玩家副露信息-暗杠, 4 玩家
        PLAY = 16  # +4, 玩家出牌信息, 4 玩家
        LAST = 20  # +1, 最后一张牌信息
        UNKNOWN = 21  # +1, 未知牌信息
        HAND = 22  # +1, 玩家手牌信息
        PREVALENT_WIND = 23  # +1, 场风信息
        SEAT_WIND = 24  # +1, 座风信息
        SHANTEN_PASS = 25  # +1, PASS 之后的上听/胡牌所需要的牌
        SHANTEN_PLAY = 26  # +34, PLAY 之后的上听/胡牌所需要的牌

    OBS_SIZE = 60

    class OFFSET_ACT:
        Pass = 0  # 过
        Hu = 1  # 胡
        Play = 2  # 打牌
        Gang = 36  # 明杠
        AnGang = 70  # 暗杠
        BuGang = 104  # 补杠
        Peng = 138  # 碰
        Chi = 172  # 吃

    ACT_SIZE = 235

    class OFFSET_VEC:
        PLAYERS_HAND = 0  # +4, 玩家手牌数量, 4 玩家
        UNKNOWN = 4  # +1, 未知牌数量
        PREVALENT_WIND = 5  # +1, 场风信息
        SEAT_WIND = 6  # +1, 座风信息
        REST = 7  # +4, 牌墙剩余数量, 4 玩家
        STEP = 11  # +1, 当前步数
        HU_PROB_PASS = 12  # +1, 已经听牌后，PASS 之后的胡牌概率
        HU_PROB_PLAY = 13  # +34, 已经听牌后，PLAY 之后的胡牌概率
        # 以下操作中，若操作不合法则为 10
        # 发生于手牌数 13 时
        SHAN_EXP_PASS = 47  # +1, PASS 后，来下一张牌后的上听数期望
        # 发生于手牌数 14 时
        SHAN_EXP_PLAY = 48  # +34, PLAY 后，来下一张牌后的上听数期望
        # 发生于手牌数 13 时
        SHAN_DIS_PASS = 82  # +1, PASS 后，当前牌型的最小上听距离
        # 发生于手牌数14时
        SHAN_DIS_PLAY = 83  # +34, PLAY 后，当前牌型的最小上听距离

    VEC_SIZE = 117

    # 牌名列表
    TILE_LIST = [
        *("W%d" % (i + 1) for i in range(9)),  # 万
        *("T%d" % (i + 1) for i in range(9)),  # 筒
        *("B%d" % (i + 1) for i in range(9)),  # 饼
        *("F%d" % (i + 1) for i in range(4)),  # 风
        *("J%d" % (i + 1) for i in range(3)),  # 箭
    ]

    # 牌名 -> 索引 的一个字典，这个索引的范围是 OBS 的第二个维度 36（实际上最后有空余）
    OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}
    OFFSET_TILE["CONCEALED"] = 34

    def __init__(self, seatWind):
        # 座风
        self.seatWind = seatWind
        # 记录每个玩家的吃、碰、杠牌的组合。
        self.packs = [[] for i in range(4)]
        # 记录每个玩家的出牌历史。
        self.history = [[] for i in range(4)]
        # 记录每个玩家剩余的牌墙数。
        self.tileWall = [21] * 4
        # 记录已经展示出来的牌的数量。
        self.shownTiles = defaultdict(int)
        # 记录已知的牌，包括自己手牌、副露的牌、打出的牌等。
        self.knownTiles = {c: 0 for c in self.TILE_LIST}

        # 记录自己的手牌
        self.hand = []

        # 环境标识
        # 标识是否已经到了最后一张牌。如果为True，表示已经到了最后一张牌。
        self.isWallLast = False
        # 标识是否涉及杠牌。如果为True，表示当前操作涉及杠牌。
        self.isAboutKong = False
        # 记录当前观测信息与玩家的手牌信息
        self.obs = np.zeros((self.OBS_SIZE, 36))
        self.vec = np.zeros(self.VEC_SIZE)

        # 更新观测信息中的座风信息
        self.obs[
            self.OFFSET_OBS.SEAT_WIND, self.OFFSET_TILE["F%d" % (seatWind + 1)]
        ] = 1
        # 初始化 UNKNOWN 信息
        self.obs[self.OFFSET_OBS.UNKNOWN, :34] = 4

        # 初始化 Vec 信息
        # 初始化 PLAYERS_HAND 信息
        self.vec[self.OFFSET_VEC.PLAYERS_HAND : self.OFFSET_VEC.PLAYERS_HAND + 4] = 13
        # 初始化 UNKNOWN 信息
        self.vec[self.OFFSET_OBS.UNKNOWN] = 136
        # 初始化座风信息
        self.vec[self.OFFSET_VEC.SEAT_WIND] = seatWind
        self.vec[self.OFFSET_VEC.REST : self.OFFSET_VEC.REST + 4] = 21
        # 初始化有关上听数期望、距离的信息
        self.vec[self.OFFSET_VEC.SHAN_EXP_PASS :] = 10

    def request2obs(self, request):
        """
        根据一行请求更新观测信息 self.obs / self.vec
        请求的类型可以多样化，下述任意一行均为有效：
        具体格式 ref: https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong
            Wind 0..3
            Deal XX XX ...
            Player N Draw
            Player N Gang
            Player N(me) AnGang XX
            Player N(me) Play XX
            Player N(me) BuGang XX
            Player N(not me) Peng
            Player N(not me) Chi XX
            Player N(not me) AnGang

            Player N Hu
            Huang
            Player N Invalid
            Draw XX
            Player N(not me) Play XX
            Player N(not me) BuGang XX
            Player N(me) Peng
            Player N(me) Chi XX
        """
        t = request.split()
        if t[0] == "Wind":
            """
            Wind [0~3]
            """
            # 存储当前场风
            self.prevalentWind = int(t[1])
            # 更新观测信息中的场风信息
            self.obs[
                self.OFFSET_OBS.PREVALENT_WIND,
                self.OFFSET_TILE["F%d" % (self.prevalentWind + 1)],
            ] = 1
            self.vec[self.OFFSET_VEC.PREVALENT_WIND] = self.prevalentWind
            return
        # 发牌 / 起始摸牌
        if t[0] == "Deal":
            """
            Deal 牌1 牌2 ...
            """
            self.hand = t[1:]
            # 下面这个函数会根据 self.hand 更新 self.obs
            self._update_known_tile(self.hand)
            self._hand_embedding_update()
            self._shanten_embedding_update()
            return
        # 荒庄
        if t[0] == "Huang":
            """
            Huang
            """
            # 无可用操作
            self.valid = []
            # 返回信息
            return self._obs()
        # 摸牌
        if t[0] == "Draw":
            """
            Draw 牌1
            """
            # 可选动作： Hu, Play, AnGang, BuGang

            # 减少牌墙数量
            self.tileWall[0] -= 1
            # 更新 vec 信息
            self.vec[self.OFFSET_VEC.REST] = self.tileWall[0]
            # 更新当前步数
            self.vec[self.OFFSET_VEC.STEP] += 1
            # 检查是否是最后一张牌
            self.isWallLast = self.tileWall[1] == 0
            # 获取摸到的牌
            tile = t[1]

            # 更新已知牌信息
            self._update_known_tile([tile])

            # 初始置空可用操作空间
            self.valid = []

            # 检查是否能胡牌
            if self._check_mahjong(
                tile, isSelfDrawn=True, isAboutKong=self.isAboutKong
            ):
                self.valid.append(self.OFFSET_ACT.Hu)

            # 将摸到的牌添加到玩家手牌中，更新手牌
            self.hand.append(tile)
            self._hand_embedding_update()
            self._shanten_embedding_update()
            # 重置杠相关标志
            self.isAboutKong = False

            # 生成动作空间
            for tile in set(self.hand):  # set 可以去重
                # 生成打出牌的有效动作，添加 Play 牌X 的动作索引
                self.valid.append(self.OFFSET_ACT.Play + self.OFFSET_TILE[tile])
                # 检查暗杠的可能性，添加 AnGang 牌X 的动作索引
                if (
                    self.hand.count(tile) == 4
                    and not self.isWallLast
                    and self.tileWall[0] > 0
                ):
                    self.valid.append(self.OFFSET_ACT.AnGang + self.OFFSET_TILE[tile])

            # 检查补杠的可能性
            if not self.isWallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == "PENG" and tile in self.hand:
                        self.valid.append(
                            self.OFFSET_ACT.BuGang + self.OFFSET_TILE[tile]
                        )
            return self._obs()

        # 往后的分支都是根据 Player N XX 这种格式的请求来处理的
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        # 获取玩家编号
        p = (int(t[1]) + 4 - self.seatWind) % 4

        # Player N Draw
        if t[2] == "Draw":
            """
            玩家 p 摸牌
            """
            # 减少玩家 p 的牌墙数量
            self.tileWall[p] -= 1
            # 更新 vec 信息
            self.vec[self.OFFSET_VEC.REST + p] = self.tileWall[p]

            # 检查是否是最后一张牌
            self.isWallLast = self.tileWall[(p + 1) % 4] == 0
            # 更新 PLAYER p 的手牌数量信息
            self.vec[self.OFFSET_VEC.PLAYERS_HAND + p] += 1
            return

        # Player N Invalid
        if t[2] == "Invalid":
            """
            玩家 p 的操作无效
            """
            # 初始置空可用操作空间
            self.valid = []
            return self._obs()

        # Player N Hu
        if t[2] == "Hu":
            """
            玩家 p 胡牌
            """
            # 初始置空可用操作空间
            self.valid = []
            return self._obs()

        # Player N Play XX
        if t[2] == "Play":
            """
            玩家 p 打出一张牌 XX
            """
            # 获取打出的牌
            self.tileFrom = p
            self.curTile = t[3]
            # 更新打出的牌数量
            self.shownTiles[self.curTile] += 1
            # 记录全局信息
            # 时序衰减
            self.obs[self.OFFSET_OBS.PLAY + p, :] *= 0.9
            self.obs[self.OFFSET_OBS.PLAY + p, self.OFFSET_TILE[self.curTile]] += 1

            if p == 3:
                self.obs[self.OFFSET_OBS.LAST, :] = 0
                self.obs[self.OFFSET_OBS.LAST, self.OFFSET_TILE[self.curTile]] = 1

            # 将打出的牌记录在历史记录中
            self.history[p].append(self.curTile)

            # 如果是自己打出的牌
            if p == 0:
                # 从手牌中移除打出的牌，更新手牌
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # 更新 UNKNOWN 信息
                self._update_known_tile([self.curTile])
                # 可选动作：Hu, Gang, Peng, Chi, Pass
                # 更新玩家 p 的手牌数量信息
                self.vec[self.OFFSET_VEC.PLAYERS_HAND + p] -= 1
                self.valid = []
                # 检查是否能胡牌
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT.Hu)
                # 检查是否是最后一张牌
                if not self.isWallLast:
                    # 检查碰和杠的可能性
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(
                            self.OFFSET_ACT.Peng + self.OFFSET_TILE[self.curTile]
                        )
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(
                                self.OFFSET_ACT.Gang + self.OFFSET_TILE[self.curTile]
                            )
                    # 检查吃的可能性
                    color = self.curTile[0]
                    if p == 3 and color in "WTB":
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3):
                            tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(
                                self.OFFSET_ACT.Chi
                                + "WTB".index(color) * 21
                                + (num - 3) * 3
                                + 2
                            )

                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(
                                self.OFFSET_ACT.Chi
                                + "WTB".index(color) * 21
                                + (num - 2) * 3
                                + 1
                            )

                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(
                                self.OFFSET_ACT.Chi
                                + "WTB".index(color) * 21
                                + (num - 1) * 3
                            )

                # 添加 Pass 动作
                self.valid.append(self.OFFSET_ACT.Pass)
                self._shanten_embedding_update()
                return self._obs()

        # Player N Chi XX
        if t[2] == "Chi":
            """
            玩家 p 吃牌 XX
            这个 XX 是顺子中心牌
            """
            # 获取吃的牌
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            # 更新吃牌的记录
            self.packs[p].append(("CHI", tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1

            for i in range(-1, 2):
                # 存储玩家明牌信息
                self.obs[self.OFFSET_OBS.SHUN + p][
                    self.OFFSET_TILE[color + str(num + i)]
                ] += 1
                # 存储全局明牌信息
                self.shownTiles[color + str(num + i)] += 1
                # 存储玩家吃牌副露的顺子
                self.obs[
                    self.OFFSET_OBS.SHUN + p, self.OFFSET_TILE[color + str(num + i)]
                ] += 1

            # 检查是否是最后一张牌
            self.isWallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # 可选动作：Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                self._shanten_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT.Play + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                newly_known_tiles = []
                for i in range(-1, 2):
                    tile_name = color + str(num + i)
                    if tile_name != self.curTile:
                        newly_known_tiles.append(tile_name)
                self._update_known_tile(newly_known_tiles)
                # 更新玩家 p 的手牌数量信息
                self.vec[self.OFFSET_VEC.PLAYERS_HAND + p] -= 2
                return

        # Player N UnChi XX
        if t[2] == "UnChi":
            """
            玩家 p 取消吃牌 XX
            实际上这个分支根本没进去过，麻了
            """
            # 获取取消吃的牌
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            # 更新取消吃牌的记录
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return

        # Player N Peng
        if t[2] == "Peng":
            """
            玩家 p 碰牌
            """
            # 更新碰牌的记录
            self.packs[p].append(("PENG", self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2

            # 存储玩家碰牌副露的刻子
            self.obs[self.OFFSET_OBS.KE + p, self.OFFSET_TILE[self.curTile]] = 3

            # 检查是否是最后一张牌
            self.isWallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # 可选动作：Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self._shanten_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT.Play + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                self._update_known_tile([self.curTile] * 2)
                # 更新玩家 p 的手牌数量信息
                self.vec[self.OFFSET_VEC.PLAYERS_HAND + p] -= 2
                return

        # Player N UnPeng
        if t[2] == "UnPeng":
            """
            玩家 p 取消碰牌
            实际上这个分支根本没进去过，麻了
            """
            # 更新取消碰牌的记录
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return

        # Player N Gang
        if t[2] == "Gang":
            """
            玩家 p 杠牌
            """
            # 更新杠牌的记录
            self.packs[p].append(("GANG", self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3

            # 存储玩家杠牌副露的杠子
            self.obs[self.OFFSET_OBS.GANG + p, self.OFFSET_TILE[self.curTile]] = 4

            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            else:
                self._update_known_tile([self.curTile] * 3)
                # 更新玩家 p 的手牌数量信息
                self.vec[self.OFFSET_VEC.PLAYERS_HAND + p] -= 3
            return

        # Player N AnGang XX
        if t[2] == "AnGang":
            """
            玩家 p 暗杠 XX
            """
            # 更新暗杠的记录
            tile = "CONCEALED" if p else t[3]
            self.packs[p].append(("GANG", tile, 0))
            if p == 0:
                self.isAboutKong = True
                for i in range(4):
                    self.hand.remove(tile)
                    # 这里感觉可以更新下 obs 手牌信息？
            else:
                self.isAboutKong = False
                # 更新玩家 p 的手牌数量信息
                self.vec[self.OFFSET_VEC.PLAYERS_HAND + p] -= 4
                self.obs[self.OFFSET_OBS.ANGANG + p, self.OFFSET_TILE[tile]] += 4
            return

        # Player N BuGang XX
        if t[2] == "BuGang":
            """
            玩家 p 补杠 XX
            """
            # 更新补杠的记录
            tile = t[3]
            # 遍历玩家 p 的吃、碰、杠牌组合
            for i in range(len(self.packs[p])):
                # 找到与补杠牌匹配的碰牌记录
                if tile == self.packs[p][i][1]:
                    # 将碰牌记录更新为杠牌记录，保持原有的出牌者信息。
                    self.packs[p][i] = ("GANG", tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1

            # 存储玩家补杠副露的杠子
            self.obs[self.OFFSET_OBS.GANG + p, self.OFFSET_TILE[tile]] = 4
            # 移除玩家补杠前记录的刻子
            self.obs[self.OFFSET_OBS.KE + p, self.OFFSET_TILE[tile]] = 0

            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # 可选动作：Hu, Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn=False, isAboutKong=True):
                    self.valid.append(self.OFFSET_ACT.Hu)
                self.valid.append(self.OFFSET_ACT.Pass)
                self._shanten_embedding_update()
                # 更新玩家 p 的手牌数量信息
                self.vec[self.OFFSET_VEC.PLAYERS_HAND + p] -= 1
                return self._obs()
        raise NotImplementedError("Unknown request %s!" % request)

    """
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    """

    def action2response(self, action):
        """
        将动作索引转换为对应的动作字符串。
        在 __main__.py 中转换输出使用。
        """
        if action < self.OFFSET_ACT.Hu:
            return "Pass"
        if action < self.OFFSET_ACT.Play:
            return "Hu"
        if action < self.OFFSET_ACT.Gang:
            return "Play " + self.TILE_LIST[action - self.OFFSET_ACT.Play]
        if action < self.OFFSET_ACT.AnGang:
            return "Gang"
        if action < self.OFFSET_ACT.BuGang:
            return "Gang " + self.TILE_LIST[action - self.OFFSET_ACT.AnGang]
        if action < self.OFFSET_ACT.Peng:
            return "BuGang " + self.TILE_LIST[action - self.OFFSET_ACT.BuGang]
        if action < self.OFFSET_ACT.Chi:
            return "Peng"

        # Chi
        t = (action - self.OFFSET_ACT.Chi) // 3
        return "Chi " + "WTB"[t // 7] + str(t % 7 + 2)

    def response2action(self, response):
        """
        将动作字符串转换为对应的动作索引。
        也即提供分类的类别信息，在 preprocess.py 中提取训练标签使用。
        """
        t = response.split()
        if t[0] == "Pass":
            return self.OFFSET_ACT.Pass
        if t[0] == "Hu":
            return self.OFFSET_ACT.Hu
        if t[0] == "Play":
            return self.OFFSET_ACT.Play + self.OFFSET_TILE[t[1]]
        if t[0] == "Chi":
            return (
                self.OFFSET_ACT.Chi
                + "WTB".index(t[1][0]) * 7 * 3
                + (int(t[2][1]) - 2) * 3
                + int(t[1][1])
                - int(t[2][1])
                + 1
            )
        if t[0] == "Peng":
            return self.OFFSET_ACT.Peng + self.OFFSET_TILE[t[1]]
        if t[0] == "Gang":
            return self.OFFSET_ACT.Gang + self.OFFSET_TILE[t[1]]
        if t[0] == "AnGang":
            return self.OFFSET_ACT.AnGang + self.OFFSET_TILE[t[1]]
        if t[0] == "BuGang":
            return self.OFFSET_ACT.BuGang + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT.Pass

    def _obs(self):
        """
        生成当前观测信息，以及可执行动作的掩码
        """
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            # 存储当前可以执行的动作，通过置 1 来实现
            mask[a] = 1

        return {
            "observation": self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
            "vec": self.vec.copy(),
            "action_mask": mask,
        }

    def _hand_embedding_update(self):
        """
        根据 self.hand 更新 self.obs 矩阵中的手牌信息部分，以反映当前玩家手中的牌。
        """
        # 清空手牌信息部分
        self.obs[self.OFFSET_OBS.HAND, :] = 0
        d = defaultdict(int)  # d[tile] 表示手牌中牌 tile 的数量，默认值为 0
        # 统计手牌中各种牌的数量
        for tile in self.hand:
            d[tile] += 1
        for tile in d:
            # 更新 self.obs 矩阵中的手牌信息部分
            # OFFSET_OBS['HAND'] 是常量偏移 2，用于锁定手牌信息部分的起始位置
            # 等价于 self.obs[2:2+d[tile]]
            # 所以这里可以看出来，后面四个维度实际记录的是你拥有几张牌的信息，拥有 3 张牌则 self.obs[2:5] 都是 1
            # 也就是拥有几张就顺序记录几个维度
            self.obs[self.OFFSET_OBS.HAND, self.OFFSET_TILE[tile]] = d[tile]
        # 更新自己手牌总和
        self.vec[self.OFFSET_VEC.PLAYERS_HAND] = len(self.hand)

    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False):
        """
        检查是否可以和牌，即是否有足够的番数。
        """
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]),
                hand=tuple(self.hand),
                winTile=winTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=self.shownTiles[winTile] == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.isWallLast,
                seatWind=self.seatWind,
                prevalentWind=self.prevalentWind,
                verbose=True,
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8:
                raise Exception("Not Enough Fans")
        except Exception:
            return False
        return True

    def _shanten_compute(self, h):
        """
        Return:
        - 各种上听方式的(上听数，上听所需牌)列表
        - 最小上听数
        """
        # shanten_min = MahjongShanten(pack=tuple(self.packs[0]), hand=tuple(self.hand))
        hand_num = len(h)
        shanten_path = [(20, None) for _ in range(5)]
        min_shanten = 20
        # min_shanten_index = -1

        if hand_num == 13:
            shanten, useful = ThirteenOrphansShanten(hand=tuple(h))
            shanten_path[0] = (shanten, useful)
            if shanten < min_shanten:
                #     min_shanten_index = 0
                min_shanten = shanten

            shanten, useful = SevenPairsShanten(hand=tuple(h))
            shanten_path[1] = (shanten, useful)
            if shanten < min_shanten:
                #     min_shanten_index = 1
                min_shanten = shanten

            shanten, useful = HonorsAndKnittedTilesShanten(hand=tuple(h))
            shanten_path[2] = (shanten, useful)
            if shanten < min_shanten:
                #     min_shanten_index = 2
                min_shanten = shanten

            shanten, useful = KnittedStraightShanten(hand=tuple(h))
            shanten_path[3] = (shanten, useful)
            if shanten < min_shanten:
                #     min_shanten_index = 3
                min_shanten = shanten

            shanten, useful = RegularShanten(hand=tuple(h))
            shanten_path[4] = (shanten, useful)

            if shanten < min_shanten:
                #     min_shanten_index = 4
                min_shanten = shanten

        elif hand_num == 10:
            shanten, useful = KnittedStraightShanten(hand=tuple(h))
            shanten_path[3] = (shanten, useful)
            if shanten < min_shanten:
                #     min_shanten_index = 3
                min_shanten = shanten

            shanten, useful = RegularShanten(hand=tuple(h))
            shanten_path[4] = (shanten, useful)
            if shanten < min_shanten:
                #     min_shanten_index = 4
                min_shanten = shanten

        else:
            shanten, useful = RegularShanten(hand=tuple(h))
            shanten_path[4] = (shanten, useful)
            if shanten < min_shanten:
                #     min_shanten_index = 4
                min_shanten = shanten
        assert min([path[0] for path in shanten_path]) <= 6, "No useful tile found"

        return shanten_path, min_shanten

    def _shanten_embedding_update(self):
        """
        更新上听数、上听所需牌信息
        """
        if 3 * len(self.packs[0]) + len(self.hand) >= 14:
            """
            发生在刚刚抽完牌、吃/碰他人的牌时。
            """
            # 清空过牌上听的所需牌信息
            self.obs[self.OFFSET_OBS.SHANTEN_PASS] = 0
            # 清空打出牌上听的所需牌信息
            self.obs[
                self.OFFSET_OBS.SHANTEN_PLAY : self.OFFSET_OBS.SHANTEN_PLAY + 34
            ] = 0

            # 清空已上听时，过牌胡牌的概率信息
            self.vec[self.OFFSET_VEC.HU_PROB_PASS] = 0
            # 清空已上听时，打出牌胡牌的概率信息
            self.vec[
                self.OFFSET_VEC.HU_PROB_PLAY : self.OFFSET_VEC.HU_PROB_PLAY + 34
            ] = 0

            # 清空 self.vec 中的上听距离、期望信息
            self.vec[self.OFFSET_VEC.SHAN_EXP_PASS :] = 10

            # 此时，不可能可以 Pass
            shanten = -1
            for t in set(self.hand):
                h = self.hand.copy()
                h.remove(t)
                shanten_path, min_shanten = self._shanten_compute(h)
                shanten = min_shanten

                # 将所有 shanten 为最小值的上听方式的上听牌取并集
                useful_tiles = np.zeros(36)
                for path in shanten_path:
                    if path[0] == min_shanten:
                        for shan_tile in path[1]:
                            useful_tiles[self.OFFSET_TILE[shan_tile]] = 1

                self.obs[self.OFFSET_OBS.SHANTEN_PLAY + self.OFFSET_TILE[t]] = (
                    useful_tiles
                )
                self.vec[self.OFFSET_VEC.SHAN_DIS_PLAY + self.OFFSET_TILE[t]] = shanten

                # 打出后上听，计算胡牌的概率
                if shanten == 0:
                    # Bool 列表
                    self_drawn_hu_valid = [
                        self._check_mahjong(
                            win_tile, isSelfDrawn=True, isAboutKong=False
                        )
                        for win_tile in useful_tiles
                    ]
                    self_drawn_hu_prob = np.sum(
                        (self.obs[self.OFFSET_OBS.UNKNOWN] * useful_tiles)[
                            self_drawn_hu_valid
                        ]
                    ) / np.sum(self.obs[self.OFFSET_OBS.UNKNOWN])

                    others_hu_valid = [
                        self._check_mahjong(
                            win_tile, isSelfDrawn=False, isAboutKong=False
                        )
                        for win_tile in useful_tiles
                    ]
                    others_hu_prob = np.sum(
                        (self.obs[self.OFFSET_OBS.UNKNOWN] * useful_tiles)[
                            others_hu_valid
                        ]
                    ) / np.sum(self.obs[self.OFFSET_OBS.UNKNOWN])

                    self.vec[self.OFFSET_VEC.HU_PROB_PLAY + self.OFFSET_TILE[t]] = (
                        self_drawn_hu_prob * 1 / 4 + others_hu_prob * 3 / 4
                    )

                # 打出后未上听，计算来下一张牌后的上听数的期望
                else:
                    # 首先获得有效的上听方式
                    could_use_prob = np.sum(
                        self.obs[self.OFFSET_OBS.UNKNOWN] * useful_tiles
                    ) / np.sum(self.obs[self.OFFSET_OBS.UNKNOWN])
                    # 如果下一张牌是有效的上听牌，那么上听数减少 1，所以期望值为 shanten - could_use_prob
                    self.vec[self.OFFSET_VEC.SHAN_EXP_PLAY + self.OFFSET_TILE[t]] = (
                        shanten - could_use_prob
                    )

        else:
            """
            发生其他时间点
            """
            h = self.hand.copy()
            # 清空过牌上听的所需牌信息
            self.obs[self.OFFSET_OBS.SHANTEN_PASS] = 0
            # 清空打出牌上听的所需牌信息
            self.obs[
                self.OFFSET_OBS.SHANTEN_PLAY : self.OFFSET_OBS.SHANTEN_PLAY + 34
            ] = 0

            # 清空已上听时，过牌胡牌的概率信息
            self.vec[self.OFFSET_VEC.HU_PROB_PASS] = 0
            # 清空已上听时，打出牌胡牌的概率信息
            self.vec[
                self.OFFSET_VEC.HU_PROB_PLAY : self.OFFSET_VEC.HU_PROB_PLAY + 34
            ] = 0
            # 清空 self.vec 中的上听距离、期望信息
            self.vec[self.OFFSET_VEC.SHAN_EXP_PASS :] = 10

            # 此时，不可能可以打出，首先计算 PASS 的信息
            shanten_path, min_shanten = self._shanten_compute(h)
            useful_tiles = np.zeros(36)
            for path in shanten_path:
                if path[0] == min_shanten:
                    for shan_tile in path[1]:
                        useful_tiles[self.OFFSET_TILE[shan_tile]] = 1
            self.obs[self.OFFSET_OBS.SHANTEN_PASS] = useful_tiles
            self.vec[self.OFFSET_VEC.SHAN_DIS_PASS] = min_shanten

            if min_shanten == 0:
                # Bool 列表
                self_drawn_hu_valid = [
                    self._check_mahjong(win_tile, isSelfDrawn=True, isAboutKong=False)
                    for win_tile in useful_tiles
                ]
                self_drawn_hu_prob = np.sum(
                    (self.obs[self.OFFSET_OBS.UNKNOWN] * useful_tiles)[
                        self_drawn_hu_valid
                    ]
                ) / np.sum(self.obs[self.OFFSET_OBS.UNKNOWN])

                others_hu_valid = [
                    self._check_mahjong(win_tile, isSelfDrawn=False, isAboutKong=False)
                    for win_tile in useful_tiles
                ]
                others_hu_prob = np.sum(
                    (self.obs[self.OFFSET_OBS.UNKNOWN] * useful_tiles)[others_hu_valid]
                ) / np.sum(self.obs[self.OFFSET_OBS.UNKNOWN])

                self.vec[self.OFFSET_VEC.HU_PROB_PASS] = (
                    self_drawn_hu_prob * 1 / 4 + others_hu_prob * 3 / 4
                )

            else:
                could_use_prob = np.sum(
                    self.obs[self.OFFSET_OBS.UNKNOWN] * useful_tiles
                ) / np.sum(self.obs[self.OFFSET_OBS.UNKNOWN])
                self.vec[self.OFFSET_VEC.SHAN_EXP_PASS] = min_shanten - could_use_prob

    def _update_known_tile(self, newly_known):
        d = defaultdict(int)
        for tile in newly_known:
            d[tile] += 1

        for tile in d:
            self.knownTiles[tile] += d[tile]
            assert self.knownTiles[tile] <= 4, "Error occurs in known_tile_cnt!"

        self.vec[self.OFFSET_VEC.UNKNOWN] = 136 - sum(self.knownTiles.values())
        for tile in self.knownTiles.keys():
            self.obs[self.OFFSET_OBS.UNKNOWN, self.OFFSET_TILE[tile]] = (
                4 - self.knownTiles[tile]
            )
