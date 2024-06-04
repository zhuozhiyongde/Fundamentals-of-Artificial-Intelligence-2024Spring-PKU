#pragma once

#include <vector>
#include <cfloat>
#include <cmath>
#include <type_traits>
#include <unordered_map>
#include <iostream>

#include "../interface/game_state.hpp"
#include "../utils/search_tree.hpp"
#include "../utils/selection.hpp"
#include "../utils/random_variables.hpp"

std::vector<double> operator+ (const std::vector<double>& a, const std::vector<double>& b){
    std::vector<double> result(a);
    for (int i = 0; i < result.size(); ++ i){
        result[i] += b[i];
    }
    return result;
}

std::vector<double>& operator+= (std::vector<double>& a, const std::vector<double>& b){
    a = a + b;
    return a;
}

template<typename GameState>
class MonteCarloTreeSearch{
private:

    using ActionType = typename GameState::ActionBaseType;

    static_assert(std::is_base_of<GameStateBase<ActionType>, GameState>::value, "GameState not derived from GameStateBase.");

    // 搜索树
    SearchTree tree;

    // 把状态映射到结点编号
    std::unordered_map<GameState, int> state_to_index;

    // 把结点编号映射为状态
    std::unordered_map<int, GameState> index_to_state;

    // 把结点的index映射到访问次数
    std::unordered_map<int, int> visit_count_of;

    // 把结点的index映射到访问该节点各玩家所获得的总价值
    std::unordered_map<int, std::vector<double> > value_sums_of;
    
    // 完全随机模拟
    std::vector<double> simulate_from(GameState state) const {
        int action_id;
        while (not state.done()){
            action_id = RandomVariables::uniform_int() % state.n_actions();
            state = state.next(state.action_space()[action_id]);
        }
        return state.cumulative_rewards();
    }

    // 采样一条路径，更新途径结点访问计数和总价值
    std::vector<double> sample_path(const GameState& state, double exploration){
        
        // 当前状态在搜索树中的编号
        int index = state_to_index[state];

        // 当前状态对应的搜索树结点
        SearchTreeNode* node = tree.node_of(index);
        
        SearchTreeNode* child;
        
        GameState next_state;
        
        std::vector<double> values;

        // 访问到的结点计数增加
        ++ visit_count_of[index];

        // 如果未完全扩展当前结点，选择一个没有做过的动作来尝试，扩展后模拟
        if (node->n_children() < state.n_actions()){
            
            // 扩展的结点对应的状态
            next_state = state.next(state.action_space()[node->n_children()]);

            // 在搜索树上添加子结点
            child = tree.create_node();
            tree.add_as_child(node, child);
            
            // 维护搜索树上结点编号与状态之间的对应关系
            state_to_index[next_state] = child->index();
            index_to_state[child->index()] = next_state;

            // 子结点初始访问计数为1
            visit_count_of[child->index()] = 1;

            // TODO：子结点初始累计收益 values 为模拟得到的值——1行
            
            value_sums_of[child->index()] = values;

        // 如果当前结点已经完全扩展，那么按照UCT算法选择其中一个子结点继续
        } else if (node->n_children() > 0){

            MaxSelection selection;
            selection.initialize(node->n_children(), -DBL_MAX);

            for (int i = 0, child; i < node->n_children(); ++ i){
                
                // child是当前选择的子结点在树中的编号
                child = node->child(i)->index();

                selection.submit(/******TODO：选择UCT值最大的子结点继续探索*****/);
            }

            next_state = state.next(state.action_space()[selection.selected_index()]);
            values = sample_path(next_state, exploration);
        
        // 如果当前结点到达终止状态，无法扩展，直接返回
        } else {
            values = state.cumulative_rewards();
        }

        value_sums_of[index] += values;
        return values;
    }

public:

    MonteCarloTreeSearch(const GameState& root_state){

        // _root_state状态对应树根，树根在搜索树中编号为0
        state_to_index[root_state] = 0;
        index_to_state[0] = root_state;

        // 初始时树根访问计数为0
        visit_count_of[0] = 0;

        // 初始时树根没有累计收益
        value_sums_of[0] = std::vector<double>(root_state.n_players(), 0);
    }
    
    ActionType select_action(int iterations, double exploration){

        GameState root_state = index_to_state[0];

        for (int i = 0; i < iterations; ++ i){

            // 从树根对应的状态开始，每次采样出一条路径，更新途经状态的访问计数和总价值
            sample_path(root_state, exploration);
        }
        
        SearchTreeNode* root = tree.root();
        
        MaxSelection selection;
        selection.initialize(root->n_children(), -DBL_MAX);
        
        // 依次考虑树根扩展出来的所有子结点，从中选择一个平均价值最高的
        for (int i = 0, child; i < root->n_children(); ++ i){

            // child是当前选择的子结点在树中的编号
            child = root->child(i)->index();

            selection.submit(/******* TODO： 按平均价值贪心选择（需要参考root_state的接口）******/);
        }
        return root_state.action_space()[selection.selected_index()];
    }
};