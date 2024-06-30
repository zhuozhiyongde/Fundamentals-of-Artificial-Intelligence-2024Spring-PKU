"""
FeatureAgent继承自Agent类，按照麻将规则处理出每个决策点的所有可行动作以及简单的特征表示。
示例的特征为6*4*9，仅包含圈风、门风和自己手牌。

提示：需要修改这个类，从而支持：
1. 更加完整的特征提取（尽量囊括所有可见信息）。
"""

from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except:
    print(
        "MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information."
    )
    raise


class FeatureAgent(MahjongGBAgent):
    """
    observation: 6*4*9
        (men+quan+hand4)*4*9
        前两个维度表示门风和场风，后四个维度表示手牌信息。
        其中手牌信息的36个维度依次为 T(1-9, 筒) W(1-9, 万) B(1-9, 饼) [F(1-4, 风) J(1-3, 箭)]
        通过修改 obs[2:7] 的各个维度的值，则可以记录拥有同样的牌的张数
        例如，有 3 张 A 牌，则 obs[2:6][pos_of_A] 都会置为 1

    action_mask: 235
        pass1+hu1+discard34+chi63(3*7*3)+peng34+gang34+angang34+bugang34
        235 = 过1 + 胡1 + 弃牌34 + 明杠34 + 暗杠34 + 补杠34 + 碰牌34 + 吃牌63
        其中吃牌 63 = 花色万条饼3 * 中心牌二到八7 * 吃三张中的第几张3
    """

    OBS_SIZE = 6  # 2 + 4
    ACT_SIZE = 235

    # OBS 的偏移，都是写死的，不会变动
    OFFSET_OBS = {"SEAT_WIND": 0, "PREVALENT_WIND": 1, "HAND": 2}
    # ACT 的偏移，也是写死的，不会变动
    OFFSET_ACT = {
        "Pass": 0,
        "Hu": 1,
        "Play": 2,
        "Chi": 36,
        "Peng": 99,
        "Gang": 133,
        "AnGang": 167,
        "BuGang": 201,
    }

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
        # 标识是否已经到了最后一张牌。如果为True，表示已经到了最后一张牌。
        self.wallLast = False
        # 标识是否涉及杠牌。如果为True，表示当前操作涉及杠牌。
        self.isAboutKong = False
        # 记录当前玩家的手牌与观测信息
        # self.obs[0] 和 self.obs[1] 中的数值为 1，表示当前玩家的座风和场风。
        # self.obs[2:6] 中的数值表示玩家手中的牌。具体解释见前 Observation
        self.obs = np.zeros((self.OBS_SIZE, 36))
        # 更新观测信息中的座风信息
        self.obs[self.OFFSET_OBS["SEAT_WIND"]][
            self.OFFSET_TILE["F%d" % (self.seatWind + 1)]
        ] = 1

    def request2obs(self, request):
        """
        根据一行请求更新观测信息 self.obs。
        请求的类型可以多样化，下述任意一行均为有效：
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
            self.obs[self.OFFSET_OBS["PREVALENT_WIND"]][
                self.OFFSET_TILE["F%d" % (self.prevalentWind + 1)]
            ] = 1
            return
        # 发牌 / 起始摸牌
        if t[0] == "Deal":
            """
            Deal 牌1 牌2 ...
            """
            self.hand = t[1:]
            # 下面这个函数会根据 self.hand 更新 self.obs
            self._hand_embedding_update()
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
            # 检查是否是最后一张牌
            self.wallLast = self.tileWall[1] == 0
            # 获取摸到的牌
            tile = t[1]
            # 初始置空可用操作空间
            self.valid = []

            # 检查是否能胡牌
            if self._check_mahjong(
                tile, isSelfDrawn=True, isAboutKong=self.isAboutKong
            ):
                self.valid.append(self.OFFSET_ACT["Hu"])

            # 重置杠相关标志
            self.isAboutKong = False
            # 将摸到的牌添加到玩家手牌中，更新手牌
            self.hand.append(tile)
            self._hand_embedding_update()

            # 生成动作空间
            for tile in set(self.hand):  # set 可以去重
                # 生成打出牌的有效动作，添加 Play 牌X 的动作索引
                self.valid.append(self.OFFSET_ACT["Play"] + self.OFFSET_TILE[tile])
                # 检查暗杠的可能性，添加 AnGang 牌X 的动作索引
                if (
                    self.hand.count(tile) == 4
                    and not self.wallLast
                    and self.tileWall[0] > 0
                ):
                    self.valid.append(
                        self.OFFSET_ACT["AnGang"] + self.OFFSET_TILE[tile]
                    )

            # 检查补杠的可能性
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == "PENG" and tile in self.hand:
                        self.valid.append(
                            self.OFFSET_ACT["BuGang"] + self.OFFSET_TILE[tile]
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
            # 检查是否是最后一张牌
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
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
            # 将打出的牌记录在历史记录中
            self.history[p].append(self.curTile)

            # 如果是自己打出的牌
            if p == 0:
                # 从手牌中移除打出的牌，更新手牌
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # 可选动作：Hu, Gang, Peng, Chi, Pass
                self.valid = []
                # 检查是否能胡牌
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT["Hu"])
                # 检查是否是最后一张牌
                if not self.wallLast:
                    # 检查碰和杠的可能性
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(
                            self.OFFSET_ACT["Peng"] + self.OFFSET_TILE[self.curTile]
                        )
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(
                                self.OFFSET_ACT["Gang"] + self.OFFSET_TILE[self.curTile]
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
                                self.OFFSET_ACT["Chi"]
                                + "WTB".index(color) * 21
                                + (num - 3) * 3
                                + 2
                            )
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(
                                self.OFFSET_ACT["Chi"]
                                + "WTB".index(color) * 21
                                + (num - 2) * 3
                                + 1
                            )
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(
                                self.OFFSET_ACT["Chi"]
                                + "WTB".index(color) * 21
                                + (num - 1) * 3
                            )
                # 添加 Pass 动作
                self.valid.append(self.OFFSET_ACT["Pass"])
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
                self.shownTiles[color + str(num + i)] += 1
            # 检查是否是最后一张牌
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # 可选动作：Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT["Play"] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
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
            # 检查是否是最后一张牌
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # 可选动作：Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT["Play"] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return

        # Player N UnPeng
        if t[2] == "UnPeng":
            """
            玩家 p 取消碰牌
            实际上这个分支也是根本没进去过
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
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
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
            else:
                self.isAboutKong = False
            return

        # Player N BuGang XX
        if t[2] == "BuGang":
            """
            玩家 p 补杠 XX
            """
            # 更新补杠的记录
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ("GANG", tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # 可选动作：Hu, Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn=False, isAboutKong=True):
                    self.valid.append(self.OFFSET_ACT["Hu"])
                self.valid.append(self.OFFSET_ACT["Pass"])
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
        """
        if action < self.OFFSET_ACT["Hu"]:
            return "Pass"
        if action < self.OFFSET_ACT["Play"]:
            return "Hu"
        if action < self.OFFSET_ACT["Chi"]:
            return "Play " + self.TILE_LIST[action - self.OFFSET_ACT["Play"]]
        if action < self.OFFSET_ACT["Peng"]:
            t = (action - self.OFFSET_ACT["Chi"]) // 3
            return "Chi " + "WTB"[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT["Gang"]:
            return "Peng"
        if action < self.OFFSET_ACT["AnGang"]:
            return "Gang"
        if action < self.OFFSET_ACT["BuGang"]:
            return "Gang " + self.TILE_LIST[action - self.OFFSET_ACT["AnGang"]]
        return "BuGang " + self.TILE_LIST[action - self.OFFSET_ACT["BuGang"]]

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

    def response2action(self, response):
        """
        将动作字符串转换为对应的动作索引。
        """
        t = response.split()
        if t[0] == "Pass":
            return self.OFFSET_ACT["Pass"]
        if t[0] == "Hu":
            return self.OFFSET_ACT["Hu"]
        if t[0] == "Play":
            return self.OFFSET_ACT["Play"] + self.OFFSET_TILE[t[1]]
        if t[0] == "Chi":
            return (
                self.OFFSET_ACT["Chi"]
                + "WTB".index(t[1][0]) * 7 * 3
                + (int(t[2][1]) - 2) * 3
                + int(t[1][1])
                - int(t[2][1])
                + 1
            )
        if t[0] == "Peng":
            return self.OFFSET_ACT["Peng"] + self.OFFSET_TILE[t[1]]
        if t[0] == "Gang":
            return self.OFFSET_ACT["Gang"] + self.OFFSET_TILE[t[1]]
        if t[0] == "AnGang":
            return self.OFFSET_ACT["AnGang"] + self.OFFSET_TILE[t[1]]
        if t[0] == "BuGang":
            return self.OFFSET_ACT["BuGang"] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT["Pass"]

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
            "action_mask": mask,
        }

    def _hand_embedding_update(self):
        """
        根据 self.hand 更新 self.obs 矩阵中的手牌信息部分，以反映当前玩家手中的牌。
        前接 deal
        """
        # 清空手牌信息部分
        self.obs[self.OFFSET_OBS["HAND"] :] = 0
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
            self.obs[
                self.OFFSET_OBS["HAND"] : self.OFFSET_OBS["HAND"] + d[tile],
                self.OFFSET_TILE[tile],
            ] = 1

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
                isWallLast=self.wallLast,
                seatWind=self.seatWind,
                prevalentWind=self.prevalentWind,
                verbose=True,
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8:
                raise Exception("Not Enough Fans")
        except:
            return False
        return True