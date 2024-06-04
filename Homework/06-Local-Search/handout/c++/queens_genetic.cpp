#include <ctime>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "interface/population.hpp"
#include "utils/random_variables.hpp"

// 将n皇后问题备选解集合建模为种群，基因序列类型为int数组
class QueensPopulation : public PopulationBase<std::vector<int> >{
private:

    using ChromosomeType = std::vector<int>;
    
    // 当前种群中的所有个体
    std::vector<ChromosomeType> _population;

    // 每个个体的适应度
    std::vector<double> _adaptability;

    // 突变概率
    double mutation_rate;

    // 当前种群中最优的个体和次优的个体序号
    int best_index, second_index;

    // 评估一个基因序列的适应度
    double evaluate_chromosome(const ChromosomeType& c) const {
        int n_queens = c.size();
        std::vector<int> lr(n_queens << 1, 0), rl(n_queens << 1, 0);
        std::vector<int> column(n_queens, 0);
        int conflicts = 0;
        for (int i = 0; i < n_queens; ++ i){
            ++ column[c[i]];
            ++ rl[c[i]+i];
            ++ lr[c[i]-i+n_queens-1];   
        }
        for (int i = 0; i < n_queens; ++ i){
            conflicts += column[c[i]] + rl[c[i]+i] + lr[c[i]-i+n_queens-1] - 3;
        }
        // 目标适应度：0
        return -conflicts;
    }
    
    void update_adaptability(){
        for (int i = 0; i < _population.size(); ++ i){
            _adaptability[i] = evaluate_chromosome(_population[i]);
        }
    }

    // 选择亲本，选择适应度最高的两个个体作为下一代亲本
    const ChromosomeType& select_chromosome_with_adaptability() const {
        
        int coin = RandomVariables::uniform_int() & 1;
        return coin == 0 ? _population[second_index] : _population[best_index];
    }

    // 亲本交叉操作，随机选一个点断开，交叉互换
    ChromosomeType cross(const ChromosomeType& c1, const ChromosomeType& c2) const {
        int index = RandomVariables::uniform_int() % c1.size();
        ChromosomeType result(c1.begin(), c1.begin() + index);
        result.insert(result.end(), c2.begin() + index, c2.end());
        return result;
    }

    // 变异
    void mutate(ChromosomeType& c) const {
        // 随机变异一行皇后的位置
        int index = RandomVariables::uniform_int() % c.size();
        c[index] = RandomVariables::uniform_int() % c.size();
    }

    void update_best_and_second(){
        for (int i = 0; i < _adaptability.size(); ++ i){
            if (_adaptability[i] > _adaptability[best_index]){
                second_index = best_index;
                best_index = i;
            } else if (_adaptability[i] > _adaptability[second_index]){
                second_index = i;
            }
        }
    }

public:

    inline std::vector<ChromosomeType> population() const override {return _population;}
    inline std::vector<double> adaptability() const override {return _adaptability;}
    
    // 通过选择亲本以及交叉生成下一代种群
    void cross() override {
        update_best_and_second();
        std::vector<ChromosomeType> new_population(_population.size());
        ChromosomeType c1, c2;
        for (int i = 0; i < _population.size(); ++ i){
            c1 = select_chromosome_with_adaptability();
            c2 = select_chromosome_with_adaptability();
            new_population[i] = cross(c1, c2);
        }
        _population = new_population;
        update_adaptability();
    }
    
    // 对种群中成员进行变异操作
    void mutate() override {
        ChromosomeType new_c;
        for (int i = 0; i < _population.size(); ++ i){
            if (RandomVariables::uniform_real() > mutation_rate){
                continue;
            }
            new_c = _population[i];
            mutate(new_c);
            _population[i] = new_c;
        }
        update_adaptability();
    }
    
    void show() const override {
        for (int i = 0; i < _population.size(); ++ i){
            std::cout << "<" << i << "> Adaptability: " << _adaptability[i] << std::endl;
            
            // 只输出解
            if (_adaptability[i] == 0){
                std::cout << "Queens: " ;
                for (auto x : _population[i]){
                    std::cout << x << ' ';
                }
                std::cout << std::endl;
            }
        }
    }

    QueensPopulation(const std::vector<ChromosomeType>& p, double _mutation_rate) : _population(p), 
        mutation_rate(_mutation_rate), _adaptability(p.size()), best_index(0), second_index(0) {
        update_adaptability();
    }
};

// 生成初始种群
std::vector<std::vector<int> > generate_queens_population(int n_queens, int population_size){
    std::vector<std::vector<int> > p(population_size);

    for (int i = 0; i < population_size; ++ i){
        p[i] = RandomVariables::uniform_permutation(n_queens);
    }
    return p;
}

#include "algorithm/genetic.hpp"

int n = 20;

int main(){

    time_t t0 = time(nullptr);

    std::ios::sync_with_stdio(false);

    // n皇后，种群大小8n
    auto p = generate_queens_population(n, n << 3);
    // 突变概率0.9
    QueensPopulation pop(p, 0.9);
    GeneticAlgorithm<QueensPopulation> gen(pop);
    // 进化8n代
    gen.evolve(n << 3);
    
    std::cout << "Total time: " << time(nullptr) - t0 << std::endl;
    
    return 0;
}