#pragma once

#include <iostream>
#include <type_traits>

#include "../interface/population.hpp"
#include "../utils/random_variables.hpp"

// 遗传算法
template <typename PopulationType>
class GeneticAlgorithm{
private:
    
    using ChromosomeType = typename PopulationType::ChromosomeBaseType;
    
    static_assert(std::is_base_of<PopulationBase<ChromosomeType>, PopulationType>::value, "PopulationType not derived from PopulationBase.");
    
    PopulationType _population;
    
public:

    // 传入初始种群
    GeneticAlgorithm(const PopulationType& init_population) : _population(init_population) {}
    
    // 获得当前种群中每个个体的列表
    std::vector<ChromosomeType> population() const {return _population.population();} 

    // 当前种群中每个个体的适应度
    std::vector<double> adaptability() const {return _population.adaptability();}

    // 进化n轮
    void evolve(int n){

        for (int i = 0; i < n; ++ i){

            std::cout << "<epoch " << i << " begin>" << std::endl;

            // 交叉操作，在整个种群中选择若干个体交叉生成子代，改变种群
            _population.cross();

            // 突变操作：在整个种群中选择若干个体发生突变，改变种群
            _population.mutate();

            // 输出当前种群情况
            _population.show();

            std::cout << "<epoch " << i << " end>" << std::endl;
        }
    }
};