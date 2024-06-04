#pragma once

#include <vector>

// 种群类，ChromosomeType用于表示个体的基因序列
template <typename ChromosomeType>
class PopulationBase{
public:
    
    PopulationBase() = default;
    virtual ~PopulationBase() = default;

    using ChromosomeBaseType = ChromosomeType;

    // 当前种群中所有个体的列表
    virtual std::vector<ChromosomeType> population() const = 0;

    // 当前种群中所有个体的适应度
    virtual std::vector<double> adaptability() const = 0;

    virtual void show() const = 0;

    // 在种群中选择若干个体进行交叉，更新种群
    virtual void cross() = 0;

    // 在种群中选择若干个体进行突变，更新种群
    virtual void mutate() = 0;
};