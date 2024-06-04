#pragma once

#include <vector>
#include <iostream>
#include <cassert>
#include <cinttypes>

#include "../interface/state.hpp"

class EightPuzzleState : public StateBase<int> {
private:
    int64_t _state;
    double _cumulative_cost;
    inline int take_digit_at(int index) const {
        return (_state >> (index << 2)) & 0b1111;
    }

    inline void set_digit_at(int index, int digit){
        assert(digit >= 0 and digit <= 8);
        _state &= ~(0b1111LL << (index << 2));
        _state |= (int64_t(digit) << (index << 2));
    }

    void swap_digit_at(int index1, int index2){
        int digit1 = take_digit_at(index1);
        int digit2 = take_digit_at(index2);
        set_digit_at(index1, digit2);
        set_digit_at(index2, digit1);
    }
    
    int zero_index() const {
        for (int i = 0; i < 9; ++ i){
            if (take_digit_at(i) == 0){
                return i;
            }
        }
        return -1;
    }

    // direction 0:Left, 1:Right, 2:Up, 3:Down
    static int near_index(int index, int direction){
        switch(direction){
            case 0:
                return index % 3 >= 1 ? index - 1 : -1;
            case 1:
                return index % 3 <= 1 ? index + 1 : -1;
            case 2:
                return index >= 3 ? index - 3 : -1;
            case 3:
                return index <= 5 ? index + 3 : -1;
            default:
                return -1;
        }
    }

public:

    EightPuzzleState() = default;
    EightPuzzleState(const std::vector<int>& state) : _state(0), _cumulative_cost(0) {
        
        for (int i = 0; i < 9; ++ i){
            set_digit_at(i, state[i]);
        }
    }

    std::vector<int> action_space() const override {
        int zero_index = this->zero_index();
        std::vector<int> actions;
        for (int d = 0; d < 4; ++ d){
            if (~near_index(zero_index, d)){
                actions.push_back(d);
            }
        }
        return actions;
    }

    const EightPuzzleState& next(const int& action) const override {
        static EightPuzzleState next_state;
        int zero_index = this->zero_index();
        int near_index = this->near_index(zero_index, action);
        
        assert(~near_index);
        next_state = *this;
        next_state.swap_digit_at(zero_index, near_index);
        next_state._cumulative_cost += next_state.cost();
        return next_state;
    }

    void show() const override {
        for (int i = 0; i < 3; ++ i){
            for (int j = 0; j < 3; ++ j){
                std::cout << take_digit_at(i * 3 + j);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    double cost() const override {
        return 1;
    }

    double cumulative_cost() const override {
        return _cumulative_cost;
    }

    bool success() const override {
        for (int i = 1; i < 9; ++ i){
            if (take_digit_at(i-1) + 1 != take_digit_at(i)){
                return false;
            }
        }
        return true;
    }
    
    bool fail() const override {
        return false;
    }

    friend struct std::hash<EightPuzzleState>;
    friend bool operator== (const EightPuzzleState& s1, const EightPuzzleState& s2){
        return s1._state == s2._state;
    }
};

template<>
struct std::hash<EightPuzzleState>{
    size_t operator() (const EightPuzzleState& s) const {
        return s._state;
    }
};