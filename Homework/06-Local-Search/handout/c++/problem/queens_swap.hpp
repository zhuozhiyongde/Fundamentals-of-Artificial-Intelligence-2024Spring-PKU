#pragma once

#include <vector>

#include "../interface/state_local.hpp"
#include "queens_move.hpp"

class QueensSwapState : public StateLocalBase{
private:

    std::vector<std::pair<int, int> > _action_space;
    QueensMoveState _state;

public:
    
    // 初始化为不同行且不同列的皇后
    QueensSwapState(int n) : _state(n){
        for (int i = 0; i < n; ++ i){
            for (int j = i + 1; j < n; ++ j){
                _action_space.push_back(std::make_pair(i, j));
            }
        }
    }
    QueensSwapState() = default;

    inline const QueensMoveState& state() const {return _state;}

    inline std::vector<std::pair<int, int> > action_space() const {return _action_space;}

    // 交换两行，得到新的状态
    const QueensSwapState& neighbor(int neighbor_index) const override {
        static QueensSwapState next_state;
        int row1 = _action_space[neighbor_index].first;
        int row2 = _action_space[neighbor_index].second;

        next_state = *this;
        next_state._state = next_state._state.neighbor(row1 * _state.n_queens() + _state.queens()[row2]);
        next_state._state = next_state._state.neighbor(row2 * _state.n_queens() + _state.queens()[row1]);
        
        return next_state;
    }

    inline int neighbor_count() const override {
        return _action_space.size();
    }

    void reset() override {
        _state.reset();
    }

    void show() const override {
        _state.show();
    }
};