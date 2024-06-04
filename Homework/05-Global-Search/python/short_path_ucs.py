from time import time

from problem.directed_graph import *
from algorithm.uniform_cost_search import *
from algorithm.heuristic_search import *

if __name__ == "__main__":
    pos_names = [
        "Oradea",  # 0
        "Zerind",
        "Arad",  # start:2
        "Sibiu",
        "Fagaras",
        "Timisoara",  # 5
        "Rimnicu Vilcea",
        "Lugoj",
        "Pitesti",
        "Mehadia",
        "Drobeta",  # 10
        "Craiova",
        "Neamt",
        "Iasi",
        "Vaslui",
        "Giurgiu",  # 15
        "Bucharest",  # end:16
        "Urziceni",
        "Hirsova",
        "Eforie",
    ]

    # 各点到目标点（编号16：Bucharest的直线距离）
    to_target_dis = [
        380,
        374,
        366,
        253,
        176,
        329,
        193,
        244,
        100,
        241,
        242,
        160,
        234,
        226,
        199,
        77,
        0,
        80,
        151,
        161,
    ]

    # u, v, w = zip(*edge_lists)
    edge_lists = [
        (0, 1, 71),
        (0, 3, 151),
        (1, 2, 75),
        (2, 5, 118),
        (2, 3, 140),
        (5, 7, 111),
        (7, 9, 70),
        (9, 10, 75),
        (10, 11, 120),
        (11, 6, 146),
        (11, 8, 138),
        (3, 6, 80),
        (3, 4, 99),
        (4, 16, 211),
        (6, 8, 97),
        (8, 16, 101),
        (16, 15, 90),
        (16, 17, 85),
        (12, 13, 87),
        (13, 14, 92),
        (14, 17, 142),
        (17, 18, 98),
        (18, 19, 86),
    ]

    graph = DirectedGraph(20)

    for x, y, z in edge_lists:
        graph.add_edge(x, y, z)
        graph.add_edge(y, x, z)

    state = DirectedGraphState(graph, 2, 16)

    hs = HeuristicSearch(state)
    # 一致代价搜索
    hs.search(lambda s: s.cumulative_cost())
    # 贪心
    hs.search(lambda s: to_target_dis[s.current_node])
    # A*
    hs.search(lambda s: to_target_dis[s.current_node] + s.cumulative_cost())
