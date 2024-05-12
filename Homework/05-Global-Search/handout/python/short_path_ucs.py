from time import time

from problem.directed_graph import *
from algorithm.uniform_cost_search import *
from algorithm.heuristic_search import *

if __name__ == "__main__":
        
    pos_names = [
        "Oradea", # 0
        "Zerind", 
        "Arad",    # start:2
        "Sibiu",  
        "Fagaras",

        "Timisoara", # 5
        "Rimnicu Vilcea",
        "Lugoj",
        "Pitesti",
        "Mehadia",

        "Drobeta", # 10
        "Craiova",
        "Neamt",
        "Iasi",
        "Vaslui",

        "Giurgiu", # 15
        "Bucharest",  # end:16
        "Urziceni",
        "Hirsova",
        "Eforie"
    ]

    # 各点到目标点（编号16：Bucharest的直线距离）
    to_target_dis = [
        380, 374, 366, 253, 176,
        329, 193, 244, 100, 241,
        242, 160, 234, 226, 199,
        77, 0, 80, 151, 161
    ]

    # 各条边的起点，无向边仅记录一次
    u = [0,0,1,2,2, 5,7,9,10,11, 11,3,3,4,6, 8,16,16,12,13, 14,17,18]

    # 各条边的终点
    v = [1,3,2,5,3, 7,9,10,11,6, 8,6,4,16,8, 16,15,17,13,14, 17,18,19]

    # 各条边的权重
    w = [71,151,75,118,140, 111,70,75,120,146, 138,80,99,211,97, 101,90,85,87,92, 142,98,86]
    
    graph = DirectedGraph(20)
    
    for x,y,z in zip(u, v, w):
        graph.add_edge(x, y, z)
        graph.add_edge(y, x, z)
    
    state = DirectedGraphState(graph, 2, 16)
    
    hs = HeuristicSearch(state)
    hs.search(lambda s: -s.cumulative_cost())
