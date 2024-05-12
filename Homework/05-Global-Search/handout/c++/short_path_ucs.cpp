#include "problem/directed_graph.hpp"

#include "algorithm/heuristic_search.hpp"
#include "algorithm/uniform_cost_search.hpp"

const char pos_names[][20] = {
    "Oradea", // 0
    "Zerind", 
    "Arad",    // start:2
    "Sibiu",  
    "Fagaras",

    "Timisoara", // 5
    "Rimnicu Vilcea",
    "Lugoj",
    "Pitesti",
    "Mehadia",

    "Drobeta", // 10
    "Craiova",
    "Neamt",
    "Iasi",
    "Vaslui",

    "Giurgiu", // 15
    "Bucharest",  // end:16
    "Urziceni",
    "Hirsova",
    "Eforie"
};

// 各点到目标点（编号16：Bucharest的直线距离）
double to_target_dis[] = {
    380, 374, 366, 253, 176,
    329, 193, 244, 100, 241,
    242, 160, 234, 226, 199,
    77, 0, 80, 151, 161
};

// 各条边的起点，无向边仅记录一次
int u[] = {0,0,1,2,2, 5,7,9,10,11, 11,3,3,4,6, 8,16,16,12,13, 14,17,18};

// 各条边的终点
int v[] = {1,3,2,5,3, 7,9,10,11,6, 8,6,4,16,8, 16,15,17,13,14, 17,18,19};

// 各条边的权重
double w[] = {71,151,75,118,140, 111,70,75,120,146, 138,80,99,211,97, 101,90,85,87,92, 142,98,86};

double a_star_estimator(const DirectedGraphState& s){
    // 填写A*的状态估值函数
}

double uniform_cost_estimator(const DirectedGraphState& s){
    return -s.cumulative_cost();
}

double greedy_estimator(const DirectedGraphState& s){
    // 填写贪心的状态估值函数
}


int main(){
    DirectedGraph graph(20);

    for (int i = 0; i < 23; ++ i){
        graph.add_edge(u[i], v[i], w[i]);
        graph.add_edge(v[i], u[i], w[i]);
    }

    DirectedGraphState s(graph, 2, 16);
    HeuristicSearch<DirectedGraphState> hs(s);
    
    hs.search(uniform_cost_estimator);

    return 0;
}