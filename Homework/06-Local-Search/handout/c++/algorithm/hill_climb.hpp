#pragma once

#include <ctime>
#include <random>
#include <vector>
#include <type_traits>
#include <iostream>

#include "../interface/state_local.hpp"
#include "../utils/selection.hpp"
#include "../utils/random_variables.hpp"

// 爬山算法
template <typename StateLocalType> 
class HillClimb{
private:

    using ValueEstimatorType = double (*) (const StateLocalType&);
    
    static_assert(std::is_base_of<StateLocalBase, StateLocalType>::value, "StateLocalType not derived from StateLocalBase.");
    
    StateLocalType initial_state;

    // 传入：状态估值函数、算法终止条件、选择算法
    const StateLocalType& sample_path(ValueEstimatorType value_of, 
        double target_value, int max_steps, SelectionBase& selection){
        
        static StateLocalType state;
        static std::vector<int> permutation;
        
        state = initial_state;

        // 爬若干步，直到满足终止条件：存在邻居状态+未到达目标值+迭代步数条件
        for (int i = 0; state.neighbor_count() > 0 and value_of(state) < target_value and i < max_steps; ++ i){
            
            // 初始化选择器（第二个参数对First Better选择算法有效，作为比较的基准值）
            selection.initialize(state.neighbor_count(), value_of(state));

            // 打乱邻居顺序（对First Better选择算法有效）
            permutation = RandomVariables::uniform_permutation(state.neighbor_count());

            // 依照打乱的顺序以及各邻居的估值来选择下一步走到哪个邻居
            for (int j = 0; j < state.neighbor_count(); ++ j){
                selection.submit(value_of(state.neighbor(permutation[j])));
                if (selection.done()){
                    break;
                }
            }

            // TODO：状态转移（需要参考state，selection的接口）——1行
        }
        return state;
    }

public:

    HillClimb(const StateLocalType& state) : initial_state(state) {}
    
    void search(ValueEstimatorType value_of, 
        double target_value, int max_steps, SelectionBase& selection, int iterations){

        StateLocalType state;
        double state_value;

        for (int i = 0; i < iterations; ++ i){
            std::cout << "<begin>" << std::endl;
            
            // 随机初始化
            initial_state.reset();
            state = sample_path(value_of, target_value, max_steps, selection);
            state_value = value_of(state);

            if (state_value >= target_value){
                std::cout << "Successful search: " << i << std::endl;
                state.show();
                break;
            } else {
                std::cout << "Failed search" << i << std::endl;
            }
            
            std::cout << "Value: " << state_value << std::endl;
            std::cout << "<end>" << std::endl;
        }
    }
};
