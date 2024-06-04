#pragma once

#include <type_traits>

#include "heuristic_search.hpp"
#include "../interface/state.hpp"

// 一致代价搜索一个解
template<typename StateType>
class UniformCostSearch : protected HeuristicSearch<StateType>{
private:

    // 一致代价搜索即为仅使用：到达当前状态的总花费*(-1) 作为状态估值的启发式搜索
    // 可以理解为搜到目标就结束的dijkstra算法
    static double state_value_estimator(const StateType& state){
        return -state.cumulative_cost();
    }
    
public:
    
    UniformCostSearch(const StateType& state) : HeuristicSearch<StateType>(state) {}

    void search(){
        HeuristicSearch<StateType>::search(state_value_estimator);
    }
};