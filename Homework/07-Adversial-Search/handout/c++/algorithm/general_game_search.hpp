#pragma once

#include <vector>
#include <cfloat>
#include <unordered_map>
#include <iostream>

#include "../interface/game_state.hpp"
#include "../utils/show_path.hpp"

// 一般博弈搜索算法，适用于（完美信息）多人扩展式博弈
template<typename GameStateType> 
class GeneralGameSearch{
private:

    GameStateType initial_state;

    // 记录路径
    std::unordered_map<GameStateType, GameStateType> next_state_of;

    using ActionType = typename GameStateType::ActionBaseType;
    
    static_assert(std::is_base_of<GameStateBase<ActionType>, GameStateType>::value, "GameStateType not derived from GameStateBase.");

    std::vector<double> search(const GameStateType& state){
        
        // 到达结束状态，返回所有玩家分数的列表
        if (state.done()){
            return state.cumulative_rewards();
        }

        GameStateType next_state;
        std::vector<double> best_cumulative_rewards(state.n_players(), -DBL_MAX), cumulative_rewards;
        auto action_space = state.action_space();
        
        // 逐个尝试所有可行动作
        for (auto action : action_space){
            
            next_state = state.next(action);
            cumulative_rewards = search(next_state);

            if (/*****TODO：如果某个动作会导致当前玩家的分数提高，则切换到该动作******/){
                best_cumulative_rewards = cumulative_rewards;
                next_state_of[state] = next_state;
            }
        }

        return best_cumulative_rewards;
    } 

public:

    GeneralGameSearch(const GameStateType& state) : initial_state(state) {}

    void search(){
        
        next_state_of.clear();
        auto results = search(initial_state);
        for (int i = 0; i < results.size(); ++ i){
            std::cout << "Score " << i << ": " << results[i] << std::endl;
        }
        show_path(next_state_of, initial_state);
    }
};
