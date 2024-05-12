#pragma once

#include <type_traits>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <iostream>

#include "../interface/state.hpp"
#include "../utils/show_path.hpp"

// 广度优先搜索全部解
template<typename StateType> 
class BreadthFirstSearch{
private:

    StateType initial_state;

    using ActionType = typename StateType::ActionBaseType;
    
    static_assert(std::is_base_of<StateBase<ActionType>, StateType>::value, "StateType not derived from StateBase.");

public:

    BreadthFirstSearch(const StateType& state) : initial_state(state) {}

    // tree_search决定是否判断重复状态，require_path决定是否记录路径
    void search(bool tree_search=true, bool require_path=true){

        // 状态队列最长时候的长度
        int queue_peak_size = 0;

        // 宽搜状态队列
        std::queue<StateType> states_queue;

        // 记录状态的前一个状态，用于记录路径
        std::unordered_map<StateType, StateType> last_state_of;

        // 防止重复状态，判断哪些状态已经访问过
        std::unordered_set<StateType> explored_states;
        
        states_queue.push(initial_state);
        explored_states.insert(initial_state);

        StateType state, new_state;

        while (not states_queue.empty()){
            
            if (states_queue.size() > queue_peak_size){
                queue_peak_size = states_queue.size();
            }

            state = states_queue.front();
            states_queue.pop();

            if (state.success()){
                if (require_path){
                    show_reversed_path(last_state_of, state);
                } else {
                    state.show();
                }
                continue;
            }
            
            if (state.fail()){
                continue;
            }

            // 考虑所有可能动作，扩展全部子状态
            for (ActionType action : state.action_space()){
                
                new_state = state.next(action);
                
                if (tree_search){

                    // TODO：扩展待访问的结点集合——1行
                    
                    if (require_path){
                        // TODO：在 last_state_of 中记录路径——1行
                    }
                } else if (explored_states.find(new_state) == explored_states.end()){
                    
                    // TODO：扩展待访问的结点集合——2行

                    if (require_path){
                        // TODO：在 last_state_of 中记录路径——1行
                    }
                }
            }
        }

        std::cout << "Queue peak size: " << queue_peak_size << std::endl;
    }
};
