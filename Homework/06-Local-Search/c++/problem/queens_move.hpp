#pragma once

#include <cassert>
#include <vector>
#include <iostream>
#include <algorithm>

#include "../interface/state_local.hpp"
#include "../utils/random_variables.hpp"

class QueensMoveState : public StateLocalBase{
protected:

    int _n_queens, _conflicts;
    std::vector<int> _queens;
    std::vector<int> _column_count;
    std::vector<int> _left_right_count;
    std::vector<int> _right_left_count;

    inline void add_queen(int row, int column, int n=1){
        _queens[row] = column;

        _conflicts += n * (_column_count[column] 
            + _right_left_count[column + row] 
            + _left_right_count[column - row + _n_queens - 1]
        );
        _conflicts -= n < 0 ? n * 3 : 0;

        _column_count[column] += n;
        _right_left_count[column + row] += n;
        _left_right_count[column - row + _n_queens - 1] += n;
    }

public:

    QueensMoveState(int n) : _n_queens(n), _queens(n), _conflicts(0), 
        _column_count(n, 0), _left_right_count(n<<1, 0), _right_left_count(n<<1, 0) {
        
        reset();
    }

    QueensMoveState() = default;

    inline int n_queens() const {return _n_queens;}
    inline int conflicts() const {return _conflicts;}
    inline std::vector<int> queens() const {return _queens;}
    inline std::vector<int> column_count() const {return _column_count;}
    inline std::vector<int> right_left_count() const {return _right_left_count;}
    inline std::vector<int> left_right_count() const {return _left_right_count;}

    inline int neighbor_count() const override{
        return _n_queens * _n_queens;
    }
    
    const QueensMoveState& neighbor(int neighbor_index) const override {
        static QueensMoveState next_state;
        
        // 将(neighbor_index / _n_queens)行的皇后移动到(neighbor_index % _n_queens)列
        int row = neighbor_index / _n_queens;
        int column = _queens[row];
        int to_column = neighbor_index % _n_queens;
        
        next_state = *this;
        next_state.add_queen(row, column, -1);
        next_state.add_queen(row, to_column);

        return next_state;
    }

    // 随机生成初始状态
    void reset() override {
        _queens = RandomVariables::uniform_permutation(_n_queens);
        _column_count.assign(_column_count.size(), 0);
        _left_right_count.assign(_left_right_count.size(), 0);
        _right_left_count.assign(_right_left_count.size(), 0);
        
        _conflicts = 0;

        for (int i = 0; i < _n_queens; ++ i){
            add_queen(i, _queens[i]);
        }
    }

    void show() const override {

        std::cout << "Queens: ";
        for (int r = 0; r < _n_queens; ++ r){
            std::cout << _queens[r] << ' ';
        }
        std::cout << std::endl << "Conflicts: " << _conflicts << std::endl << std::endl;
    }
};
