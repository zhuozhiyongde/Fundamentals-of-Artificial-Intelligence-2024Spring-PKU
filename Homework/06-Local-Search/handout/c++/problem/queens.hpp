#pragma once

#include <cassert>
#include <vector>
#include <iostream>

#include "../interface/state.hpp"

using QueensAction = int;

// n皇后状态类
class QueensState : public StateBase<QueensAction>{
private:

    // 棋盘边长
    int _n_queens;

    // 已经填入皇后的行数
    int _current_row;

    // 每行皇后的位置
    std::vector<int> _queens;
    
    // 当前的空行可填入皇后的位置
    std::vector<QueensAction> _action_space;

    // 各列，右左对角线，左右对角线是否已有皇后
    std::vector<int8_t> _column, _right_left, _left_right;    

    inline int which_right_left(int c, int r) const {
        return c + r;
    }

    inline int which_left_right(int c, int r) const {
        return r - c + _n_queens - 1;
    }

    // 判断当前空行的某个位置能否落子
    bool is_valid_action(const QueensAction& action) const {
        return (action >= 0 and action < _n_queens and _column[action] == 0
            and _left_right[which_left_right(action, _current_row)] == 0
            and _right_left[which_right_left(action, _current_row)] == 0);
    }

    void update_action_space(){
        _action_space.clear();
        for (int c = 0; c < _n_queens; ++ c){
            if (is_valid_action(c)){
                _action_space.push_back(c);
            }
        }
    }

public:
    
    QueensState(int n) : 
        _n_queens(n), _current_row(0), _action_space(n),
        _queens(n), _column(n), _right_left(n<<1), _left_right(n<<1) {
        
        update_action_space();
    }

    QueensState() = default;

    int n_queens() const {return _n_queens;}
    int current_row() const {return _current_row;}
    std::vector<int> queens() const {return _queens;}

    std::vector<QueensAction> action_space() const override {return _action_space;}

    void show() const override {

        for (int r = 0; r < _n_queens; ++ r){
            
            if (r >= _current_row){
                std::cout << std::endl;
                continue;
            }
            
            for (int c = 0; c < _n_queens; ++ c){
                
                if (_queens[r] == c){
                    std::cout << "Q";
                } else {
                    std::cout << "+";
                }
            }
            std::cout << std::endl;
        }

        for (int r = 0; r < _n_queens; ++ r){
            std::cout << _queens[r] << ' ';
        }

        std::cout << std::endl << std::endl;
    }

    // 成功花费-1，失败花费1，中间状态花费为0
    double cost() const override {

        if (_current_row == _n_queens){
            return -1;
        } else if (_action_space.size() == 0){
            return 1;
        }

        return 0;
    }

    double cumulative_cost() const override {return cost();}

    bool success() const override {
        return cost() < 0;
    }

    bool fail() const override {
        return cost() > 0;
    }

    // 当前空行落子之后到达新的状态
    const QueensState& next(const QueensAction& action) const override {
        
        static QueensState next_state;
        
        next_state = *this;
        
        assert(next_state.is_valid_action(action));
        
        next_state._column[action] = 1;
        next_state._left_right[next_state.which_left_right(action, next_state._current_row)] = 1;
        next_state._right_left[next_state.which_right_left(action, next_state._current_row)] = 1;
        next_state._queens[next_state._current_row] = action;
        ++ next_state._current_row;
        next_state.update_action_space();
        
        return next_state;
    }

    friend struct std::hash<QueensState>;
    
    friend bool operator== (const QueensState& s1, const QueensState& s2){
        return s1._current_row == s2._current_row and s1._queens == s2._queens and s1._n_queens == s2._n_queens;
    }
};

// Cantor展开，用于计算全排列的哈希值，不用关注这一部分
template <> 
struct std::hash<QueensState> {
    
    size_t operator() (const QueensState& s) const {
        
        static std::vector<size_t> factorial = {1};
        size_t sum = 0;
        
        if (factorial.size() < s._current_row){
            for (size_t i = factorial.size(); i <= s._current_row; ++ i){
                factorial.push_back(factorial.back() * i);
            }
        }

        for (int i = 0, reverse; i < s._current_row; ++ i){
            
            reverse = 0;
            
            for (int j = i+1; j < s._current_row; ++ j){
                if (s._queens[j] < s._queens[i]){
                    ++ reverse;
                }
            }

            sum += size_t(reverse) * factorial[s._current_row-i-1];
        }

        return sum;
    }
};
