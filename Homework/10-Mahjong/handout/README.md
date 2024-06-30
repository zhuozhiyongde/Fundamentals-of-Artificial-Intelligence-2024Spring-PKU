# 国标麻将作业

<center>
  Hints, comments by <a href="https://github.com/zhuozhiyongde">Arthals</a> / GPT4o
  <br/>
  blog: <a href="https://arthals.ink">Arthals' ink</a>
</center>

你可能需要阅读：

-   [botzone / Mahjong-GB](https://botzone.org.cn/game/Mahjong-GB)
-   [botzong / Bot](https://wiki.botzone.org.cn/index.php?title=Bot)
-   [lianzhong / Mahjong-GB Rule](http://mj.lianzhong.com/gbmj/home/teaching_new_rule1)

如果你此前完全没打过麻将，强烈建议观看：

-   [酸奶芥麦面 / 麻将规则](https://www.bilibili.com/video/BV1Ju411X7bJ/)
-   [知乎 / 麻将的规则是什么？](https://www.zhihu.com/question/50126080/)

你不需要完整、熟练地掌握规则，但至少要有个大概的了解。

你也可以直接把这个作业当做一个分类任务即可。

以下内容来自 data/README.md，也建议阅读：

```md
文件：
data.txt 中包含 98209 个国标麻将对局记录，文件较大，建议不要在文件编辑器中打开，否则可能卡死。
sample.txt 中包含前 16 个对局，用于查看并熟悉格式。

每一局的数据格式：
（1）第一行为 Match <ID>，可以用https://botzone.org.cn/match/<ID>作为链接在浏览器中查看对局演示。
（2）第二行为 Wind 0..3，表示本局的圈风（0~3 分别表示东、北、西、南）。
（3）接下来四行为 Player <N> Deal XX XX ...，表示四名玩家的初始 13 张手牌，这里的 Player 0~3 分别表示东、北、西、南四个位置上的玩家。所有麻将牌均以“大写字母+数字”组合表示。如：“W4”表示“四万”，“B6”表示“六筒”，“T8”表示“八条”，“F1”～“F4”表示“东南西北”，“J1”～“J3”表示“中发白”。
（4）接下来为对局过程，有这样几种格式：
Player <N> Draw XX #玩家摸牌，XX 为摸到的牌
Player <N> Play XX #玩家打牌，XX 为打出的牌
Player <N> Chi XX #玩家吃牌，XX 为吃牌后形成的顺子中间一张牌，被吃的牌是上一行前一个玩家打出的牌。如上一个玩家打 B7，吃牌后形成 B7B8B9 顺子，则 XX 为 B8。
Player <N> Peng XX #玩家碰牌，XX 为被碰的牌，一定等于上一行被打出的牌。
Player <N> Gang XX #玩家杠牌（明杠），XX 为被杠的牌，一定等于上一行被打出的牌。
Player <N> AnGang XX #玩家杠牌（暗杠），XX 为被杠的牌，不一定等于上一行摸到的牌。
Player <N> BuGang XX #玩家补杠，XX 为补杠的牌，不一定等于上一行摸到的牌。
Player <N> Hu XX #玩家和牌，XX 为所胡的牌，可能是自摸（前一行为摸牌）、点和（前一行为打牌）、抢杠和（前一行为补杠）。
【注意】在 Peng, Gang, Hu 的格式中，可能在本行后面存在一个或多个 Ignore Player <N> Chi/Peng/Gang/Hu XX，表示在上一个玩家打牌后，有多个玩家同时宣布吃/碰/杠/胡（优先级为和>碰/杠>吃，同优先级按出牌人逆时针顺序），Ignore 表示被忽略的操作。
举例：
Player 2 Play B3
Player 1 Hu B3 Ignore Player 0 PENG B3 Ignore Player 3 CHI B4
表示 2 号位打 3 饼，3 号位吃、0 号位碰、1 号位胡，实际结果为 1 号位胡，另外两个操作被忽略。
Player 1 Play W7
Player 2 Hu W7 Ignore Player 0 HU W7
表示 1 号位打 7 万，0 号位和 2 号位同时胡，实际结果为 2 号位胡，0 号位操作被忽略（截胡）。
（5）对局结束有两种情况，要么某玩家胡牌，要么流局（荒庄）。如果是胡牌，在胡牌信息之后两行为：
Fan <F> <Description>
Score <S1> <S2> <S3> <S4>
表示胡牌番数、番型以及得分情况。如果是流局，在正常对局信息之后两行为：
Huang
Score 0 0 0 0
```
