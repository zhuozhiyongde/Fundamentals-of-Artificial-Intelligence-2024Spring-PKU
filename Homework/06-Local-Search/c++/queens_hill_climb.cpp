#include <cmath>
#include <chrono>
#include <ctime>

#include "interface/state_local.hpp"
#include "problem/queens_move.hpp"
#include "problem/queens_swap.hpp"
#include "algorithm/hill_climb.hpp"
#include "utils/selection.hpp"

int n = 220;
double log_n = log(n);

// 使用RouletteSelection时可以尝试修改下面的两个估值函数来获得更好的效果
double queens_move_state_value_estimator(const QueensMoveState& state) {
    return exp(-log_n * state.conflicts());
}

double queens_swap_state_value_estimator(const QueensSwapState& state) {
    return exp(-log_n * state.state().conflicts());
}

int main() {

    auto t0 = std::chrono::high_resolution_clock::now();

    std::ios::sync_with_stdio(false);

    // 选择第一个比当前状态好的邻居状态
    FirstBetterSelection f_selection;
    // 按照与价值成正比的概率选择邻居状态
    RouletteSelection r_selection;
    // 选择价值最高的邻居状态
    MaxSelection m_selection;

    // QueensMoveState state(n);

    // 用QueensMoveState的问题建模（动作为将某一行的皇后移动到某一列）
    // HillClimb<QueensMoveState> hcs(state);

    // 适应度达到1.0（无冲突皇后，由估值函数决定）则算法终止，至多迭代4n步，随机重启运行5次。
    // hcs.search(queens_move_state_value_estimator, 1.0, n << 2, m_selection, 5);

    // 用QueensSwapState的问题建模（动作为交换两行的皇后）
    QueensSwapState state(n);
    HillClimb<QueensSwapState> hcs(state);
    hcs.search(queens_swap_state_value_estimator, 1.0, n << 2, f_selection, 5);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Total time: " << duration << " ms" << std::endl;

    return 0;
}