#pragma once

#include <array>
#include <cassert>
#include <cinttypes>
#include <iostream>

#include "../interface/game_state.hpp"
#include "../utils/union_find_set.hpp"

// N*N的六边形棋
template<int _N>
class HexState : public NPlayerGameStateBase<int, 2>{
private:
    
    static constexpr int N = _N;

    // 红方先手
    static constexpr int8_t R = 0b01, B = 0b10;

    int steps;
    std::array<int8_t, N*N> board;

    // 用于检测红色/蓝色棋块连通性
    mutable UnionFindSet r_detector, b_detector;

    bool r_win() const {
        return r_detector.find(N*N) == r_detector.find(N*N+1);
    }

    bool b_win() const {
        return b_detector.find(N*N) == b_detector.find(N*N+1);
    }

public:
    
    HexState() : steps(0), board{0}, r_detector(N*N+2), b_detector(N*N+2) {
        
        // N*N:   up, red      /  left, blue
        // N*N+1: bottom, red  /  right, blue
        for (int i = 0; i < N; ++ i){
            r_detector.join(N*N, i);
            r_detector.join(N*N+1, N*N-1-i);
            b_detector.join(N*N, N*i);
            b_detector.join(N*N+1, N*(i+1)-1);
        }
    }

    bool done() const override {
        return r_win() or b_win();
    }

    int active_player() const override {
        return steps & 1;
    }

    std::vector<double> rewards() const override {
        static const std::vector<double> score_r_win {1, -1},
            score_b_win {-1, 1},
            score_tie {0, 0};
        
        return r_win() ? score_r_win : (b_win() ? score_b_win : score_tie); 
    }

    std::vector<double> cumulative_rewards() const override {
        return rewards();
    }

    inline int n_actions() const override {
        return N*N - steps;
    }

    std::vector<int> action_space() const override {
        std::vector<int> actions;
        for (int i = 0; i < N*N; ++ i){
            if (board[i] == 0){
                actions.push_back(i);
            }
        }
        return actions;
    }

    const HexState& next(const int& action) const override {
        static HexState next_state;
        
        assert(board[action] == 0);
        next_state = *this;
        
        next_state.board[action] = active_player() == 0 ? R : B;
        
        std::vector<int> neighbors {
            action-N, action+1-N, 
            action+1, action+N, 
            action-1+N, action-1
        };

        bool not_top = action >= N, 
            not_bottom = action < N*N-N,
            not_left = action % N != 0,
            not_right = action % N != N-1;

        std::vector<int8_t> conditions {
            not_top, not_top and not_right,
            not_right, not_bottom,
            not_bottom and not_left, not_left 
        };

        UnionFindSet& detector = active_player() == 0 ? next_state.r_detector : next_state.b_detector;

        for (int i = 0; i < conditions.size(); ++ i){
            if (conditions[i] and 
                next_state.board[neighbors[i]] == next_state.board[action]){
                detector.join(neighbors[i], action);
            }
        }

        ++ next_state.steps;
        return next_state;
    }

    void show() const override {
        const static char pieces[] = "_XO";
        for (int i = 0; i < N; ++ i){
            for (int j = 0; j < i; ++ j){
                std::cout << ' ';
            }
            for (int j = 0; j < N; ++ j){
                std::cout << pieces[board[i*N+j]] << ' ';
            }
            std::cout << '\n';
        }
    }

    friend struct std::hash<HexState>;
    friend bool operator== (const HexState& s1, const HexState& s2){
        return s1.board == s2.board;
    }
};

template<int N>
struct std::hash<HexState<N> >{
    size_t operator() (const HexState<N>& s) const {
        size_t code = s.steps;
        for (int i = 0; i < s.board.size(); ++ i){
            code ^= size_t(s.board[i]) << (i & ((sizeof(size_t) << 3) - 1));
        }
        return code;
    }
};