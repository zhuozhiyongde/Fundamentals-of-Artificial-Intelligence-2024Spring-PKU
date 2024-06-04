#pragma once

#include <vector>
#include <cfloat>
#include <algorithm>
#include <unordered_map>
#include <iostream>

#include "../interface/game_state.hpp"
#include "../utils/show_path.hpp"

// alpha-beta 剪枝搜索算法，适用于（完美信息）扩展式双人零和博弈
template<typename ZeroSumGameStateType> 
class AlphaBetaSearch{
private:

    ZeroSumGameStateType initial_state;
    
    // 记录路径
    std::unordered_map<ZeroSumGameStateType, ZeroSumGameStateType> next_state_of;
    
    using ActionType = typename ZeroSumGameStateType::ActionBaseType;
    
    static_assert(std::is_base_of<NPlayerGameStateBase<ActionType, 2>, ZeroSumGameStateType>::value, 
        "ZeroSumGameStateType not derived from NPlayerGameStateBase<ActionType, 2>.");

    double search(const ZeroSumGameStateType& state, double alpha, double beta){

        // 到达结束状态，返回玩家0的分数
        if (state.done()){
            return state.cumulative_rewards()[0];
        }

        ZeroSumGameStateType next_state;
        double new_value;
        auto action_space = state.action_space();

        // 逐个尝试所有可行动作
        for (auto action : action_space){

            next_state = state.next(action);
            new_value = search(next_state, alpha, beta);
            
            // 如果当前是玩家0决策，则其最大化自身的分数（提升自己分数下界）
            if (state.active_player() == 0 and new_value > alpha){
            
                // TODO：更新alpha或者beta的值——1行
                next_state_of[state] = next_state;

            // 如果当前是玩家1决策，则其最小化玩家0的分数（降低对手分数上界）
            } else if (state.active_player() == 1 and new_value < beta){
                
                // TODO：更新alpha或者beta的值——1行
                next_state_of[state] = next_state;
            }

            // TODO：剪枝
        }
        return state.active_player() == 0 ? alpha : beta;            
    }

public:

    AlphaBetaSearch(const ZeroSumGameStateType& state) : initial_state(state) {}

    void search(){
        next_state_of.clear();

        // 初始alpha, beta值为(-∞,+∞)
        auto result = search(initial_state, -DBL_MAX, DBL_MAX);
        std::cout << "Score 0: " << result << std::endl;
        show_path(next_state_of, initial_state);
    }
};