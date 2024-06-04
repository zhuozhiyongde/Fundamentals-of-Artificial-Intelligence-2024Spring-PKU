#include <cmath>
#include <ctime>
#include <chrono>

#include "problem/queens_constraint.hpp"
#include "algorithm/conflicts_minimize.hpp"
#include "utils/selection.hpp"

// 在保证非负单调递增的前提下，估值函数仅对轮盘赌算法起作用
double alpha = 10;
double value_of(int value) {
    return exp(alpha * value);
}

int main() {

    // time_t t0 = time(nullptr);
    auto t0 = std::chrono::high_resolution_clock::now();

    std::ios::sync_with_stdio(false);

    int n = 15000;

    QueensConstraintSatisfaction q(n);

    ConflictMinimize<QueensConstraintSatisfaction> cm(q);

    FirstBetterSelection fbs;
    RouletteSelection rs;
    MaxSelection ms;

    std::cout << "Question: " << n << " queens" << std::endl;

    // 随机重启尝试10轮，每轮最多更改变元4n次
    // ms: 最大选择算法，优先选择冲突最多的变元更改，优先更改到冲突最小的值（移步algorithm/conflict_minimize.hpp阅读）
    // value_of: 因为使用最大选择算法，因此估值函数直接使用默认的指数函数即可
    cm.search(10, n << 2, ms);

    // std::cout << "Total time: " << time(nullptr) - t0 << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Total time: " << duration << " ms" << std::endl;
    return 0;
}