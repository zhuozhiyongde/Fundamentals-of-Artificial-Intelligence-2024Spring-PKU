#pragma once

#include <random>
#include <ctime>
#include <algorithm>

// Singleton Pattern, Provide random variables
class RandomVariables{
private:

    // 固定随机种子
    RandomVariables() = default;
    
    // 非固定随机种子
    // RandomVariables() : random_engine(time(nullptr)) {}
    
    ~RandomVariables() = default;

    std::default_random_engine random_engine;
    std::uniform_real_distribution<double> uniform_dist;
    std::uniform_int_distribution<int> uniform_int_dist;

    static RandomVariables rv;

public:

    // 均匀分布的正整数
    static int uniform_int(){
        return rv.uniform_int_dist(rv.random_engine);
    }

    // [0,1)均匀分布的实数
    static double uniform_real(){
        return rv.uniform_dist(rv.random_engine);
    }

    // 等概率分布的{0,1,2,n-1}排列
    static std::vector<int> uniform_permutation(int n){
        std::vector<int> permutation(n);
        for (int i = 0; i < n; ++ i){
            permutation[i] = i;
        }

        for (int i = 0, j; i < n; ++ i){
            j = uniform_int() % (n - i) + i;
            std::swap(permutation[i], permutation[j]);
        }

        return permutation;
    }
};

RandomVariables RandomVariables::rv;
