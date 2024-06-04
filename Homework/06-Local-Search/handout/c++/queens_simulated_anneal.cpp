#include <cmath>

#include "problem/queens_move.hpp"
#include "problem/queens_swap.hpp"
#include "algorithm/simulated_anneal.hpp"

int n = 100;
int max_conflicts = n * (n-1) >> 1;

// 目标值：state.state().conflicts() == 0，即函数返回max_conflicts
double value_estimator_swap(const QueensSwapState& state){
    return max_conflicts - state.state().conflicts();
}

double value_estimator_move(const QueensMoveState& state){
    return max_conflicts - state.conflicts();
}

// 温度随时间变化的方式也可以尝试修改
double temperature_schedule_move(int time){
    static double log_n = log(n);
    static double start_temp_log = n/log_n;
    return exp(start_temp_log - double(time) / (n << 4));
}

double temperature_schedule_swap(int time){
    static double log_n = log(n);
    static double start_temp_log = n/log_n;
    return exp(start_temp_log - double(time) / n);
}

int main(){

    time_t t0 = time(nullptr);

    std::ios::sync_with_stdio(false);

    // 用QueensSwapState的问题建模（动作为交换两行的皇后）
    QueensSwapState q(n);
    SimulatedAnneal<QueensSwapState> sa(q);
    // 尝试4n次模拟退火，终态温度为10^-16
    sa.search(value_estimator_swap, temperature_schedule_swap, n << 2, max_conflicts, 1e-16); 
    
    // 用QueensMoveState的问题建模（动作为将某一行的皇后移动到某一列）
    //QueensMoveState q(n);
    //SimulatedAnneal<QueensMoveState> sa(q);
    //sa.search(value_estimator_move, temperature_schedule_move, n << 4, max_conflicts, 1e-18);

    std::cout << "Total time: " << time(nullptr) - t0 << std::endl;

    return 0;
}