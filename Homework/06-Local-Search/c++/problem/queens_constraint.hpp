#pragma once

#include <vector>
#include <iostream>

#include "../interface/constraint_satisfaction.hpp"
#include "../utils/random_variables.hpp"
#include "queens_move.hpp"

using QueensVariable = int;

// 将n皇后问题建模为约束满足问题
class QueensConstraintSatisfaction :
    public ConstraintSatisfactionBase<QueensVariable>,
    protected QueensMoveState {
private:

    std::vector<QueensVariable> choices;

public:

    QueensConstraintSatisfaction() = default;
    QueensConstraintSatisfaction(int n) : QueensMoveState(n), choices(n) {
        for (int i = 0; i < n; ++i) {
            choices[i] = i;
        }
    }

    inline int n_variables() const override { return _n_queens; }

    inline std::vector<QueensVariable> variables() const override {
        return _queens;
    }

    // 与某个皇后冲突的皇后个数
    inline int conflicts_of(int variable_index) const override {
        int row = variable_index, column = _queens[variable_index];
        return (_column_count[column]
            + _right_left_count[column + row]
            + _left_right_count[column - row + _n_queens - 1]
            - 3
            );
    }

inline int conflicts_if_set(int variable_index, QueensVariable new_variable) const {
    int row = variable_index;
    int column = new_variable;
    // int current_column = _queens[variable_index];

    // 计算如果设置到新位置的冲突数
    int conflicts = (_column_count[column]
        + _right_left_count[column + row]
        + _left_right_count[column - row + _n_queens - 1]
        );

    return conflicts;
}

    // 是否仍然存在冲突的皇后
    bool has_conflict() const override {
        for (int i = 0; i < _n_queens; ++i) {
            if (conflicts_of(i) > 0) {
                return true;
            }
        }
        return false;
    }

    // 皇后位置备选值[0-n_queens)
    inline std::vector<QueensVariable> choices_of(int variable_index) const override {
        return choices;
    }

    // 设置某一行皇后的位置
    void set_variable(int variable_index, QueensVariable new_variable) override {
        add_queen(variable_index, _queens[variable_index], -1);
        add_queen(variable_index, new_variable);
    }

    void show() const override {
        QueensMoveState::show();
    }

    void reset() override {
        QueensMoveState::reset();
    }
};