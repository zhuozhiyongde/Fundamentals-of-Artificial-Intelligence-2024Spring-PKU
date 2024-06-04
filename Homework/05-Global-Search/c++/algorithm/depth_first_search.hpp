#pragma once

#include <type_traits>
#include <unordered_set>
#include <unordered_map>
#include <stack>

#include "../interface/state.hpp"
#include "../utils/show_path.hpp"

// 深度优先搜索全部解
template <typename StateType>
class DepthFirstSearch {
private:

    StateType initial_state;

    using ActionType = typename StateType::ActionBaseType;

    static_assert(std::is_base_of<StateBase<ActionType>, StateType>::value, "StateType not derived from StateBase.");

public:

    DepthFirstSearch(const StateType& state) : initial_state(state) {}

    // tree_search决定是否判断重复状态，require_path决定是否记录路径
    void search(bool tree_search = true, bool require_path = true) {

        // 状态栈最长时候的长度
        int stack_peak_size = 0;

        // 记录：(当前状态, 当前状态已经访问的子节点个数)
        std::stack<std::pair<StateType, int> > states_stack;

        // 记录状态的前一个状态，用于记录路径
        std::unordered_map<StateType, StateType> last_state_of;

        // 防止重复状态，判断哪些状态已经访问过
        std::unordered_set<StateType> explored_states;

        states_stack.push(std::make_pair(initial_state, 0));
        explored_states.insert(initial_state);

        StateType state, new_state;
        std::pair<StateType, int> state_action;
        int action_id;

        while (not states_stack.empty()) {

            if (states_stack.size() > stack_peak_size) {
                stack_peak_size = states_stack.size();
            }

            state_action = states_stack.top();
            states_stack.pop();
            state = state_action.first;

            // 待尝试的动作编号
            action_id = state_action.second;

            if (state.success()) {
                // if (require_path) {
                //     show_reversed_path(last_state_of, state);
                // }
                // else {
                //     state.show();
                // }
                continue;
            }

            if (state.fail()) {
                continue;
            }

            // 如果当前状态还有动作未尝试过
            if (action_id < state.action_space().size()) {
                states_stack.push(std::make_pair(state, action_id + 1));
                // 尝试当前动作，获得下一步状态
                new_state = state.next(state.action_space()[action_id]);
                if (tree_search) {
                    states_stack.push(std::make_pair(new_state, 0));
                    if (require_path) {
                        last_state_of[new_state] = state;
                    }
                }
                else if (explored_states.find(new_state) == explored_states.end()) {
                    states_stack.push(std::make_pair(new_state, 0));
                    explored_states.insert(new_state);
                    if (require_path) {
                        last_state_of[new_state] = state;
                    }
                }
            }
        }
        std::cout << "Stack peak size: " << stack_peak_size << std::endl;
    }
};
