#pragma once

#include <queue>
#include <vector>
#include <unordered_map>
#include <type_traits>

#include "../interface/state.hpp"
#include "../utils/show_path.hpp"

// 根据传入的状态估值函数启发式地搜索一个解
template<typename StateType>
class HeuristicSearch{
protected:

    StateType initial_state;

    using ActionType = typename StateType::ActionBaseType;
    
    using ValueEstimatorType = double (*) (const StateType&);

    static_assert(std::is_base_of<StateBase<ActionType>, StateType>::value, "StateType not derived from StateBase.");

public:
    
    HeuristicSearch(const StateType& state) : initial_state(state) {}

    // 传入状态估值函数
    void search(ValueEstimatorType value_of){
        
        // 状态的优先队列最长时候的长度
        int priority_queue_peak_size = 0;

        // 通过传入的状态估值函数value_of定义状态比较函数
        auto cmp_state_less = [value_of] (const StateType& s1, const StateType& s2) -> bool {return value_of(s1) < value_of(s2);};
       
        // 优先队列中估值高的状态在堆顶，先被访问       
        std::priority_queue<StateType, std::vector<StateType>, decltype(cmp_state_less)> states_queue(cmp_state_less);
        
        // 某个状态的最大估值（在最短路问题中为：最短估计距离*(-1) ）
        std::unordered_map<StateType, double> best_value_of;

        // 记录状态的前一个状态，用于记录路径
        std::unordered_map<StateType, StateType> last_state_of;
        StateType state, new_state;
        
        states_queue.push(initial_state);

        best_value_of[initial_state] = 0;

        while (not states_queue.empty()){

            if (states_queue.size() > priority_queue_peak_size){
                priority_queue_peak_size = states_queue.size();
            }

            // 优先探索开结点集中估值最高的状态        
            state = states_queue.top();
            states_queue.pop();

            if (state.success()){
                break;
            }

            if (state.fail()){
                continue;
            }

            // 从开结点集中估值最高的状态出发尝试所有动作
            for (ActionType action : state.action_space()){

                // TODO：状态转移——1行

                // 如果从当前结点出发到达新结点所获得的估值高于新结点原有的估值，则更新
                if (best_value_of.find(new_state) == best_value_of.end() 
                    or value_of(new_state) > best_value_of[new_state]){
                    // TODO：更新状态价值，扩展待访问的节点集——2行
                    // TODO：记录路径——1行
                }
            }
        }

        if (state.success()){
            show_reversed_path(last_state_of, state);
        }

        std::cout << "Priority queue peak size: " << priority_queue_peak_size << std::endl;
    }
};
