#pragma once

#include <vector>
#include <cmath>
#include <type_traits>
#include <iostream>

#include "../interface/constraint_satisfaction.hpp"
#include "../utils/selection.hpp"

// 冲突最小化算法，求解约束满足问题
template <typename ConstraintSatisfactionType>
class ConflictMinimize {
private:

    using VariableType = typename ConstraintSatisfactionType::VariableBaseType;
    using ConflictValueEstimatorType = double (*) (int);

    static_assert(std::is_base_of<ConstraintSatisfactionBase<VariableType>, ConstraintSatisfactionType>::value,
        "ConstraintSatisfactionType shall be derived from ConstraintSatisfactionBase.");

    // 待求解的约束满足问题
    ConstraintSatisfactionType problem;

    // 为使发生冲突较多的变元更容易被选中，应当使冲突数选择价值函数单调递增
    // 在应用轮盘赌算法时，为使得发生冲突多的变元被替换的概率更大，应当使冲突数选择价值函数为非负函数

// 默认的冲突数选择价值函数，满足非负单调递增
static double default_conflict_value_estimator(int conflicts) {
    // if (conflicts < -20) {
    //     std::cout << conflicts << std::endl;
    // }
    // return exp(conflicts);
    return conflicts;
}

    // 传入参数：最大迭代步数，选择变元/可替换值的方法，冲突数选择价值函数（作为选择变元/可替换值的基准，对Roulette选择算法起作用）
    void sample_path(int max_steps, SelectionBase& selection, ConflictValueEstimatorType value_of) {

        // 用于重排变元/变元可替换值
        std::vector<int> permutation;

        // 用于恢复选中变元的取值，以便尝试下一个可替换值
        VariableType old_variable;

        // 选中变元可替换值列表
        std::vector<VariableType> choices;

        // 选中的变元编号
        int selected_index;

        // 终止条件：超过最大步数/各个变元与约束条件均不冲突
        for (int i = 0, j; i < max_steps and problem.has_conflict(); ++i) {

            // 生成随机排列，用于打乱变元顺序（对First Better选择算法起作用）
            permutation = RandomVariables::uniform_permutation(problem.n_variables());

            // 初始化选择器，用于选择需要改变的变元（第二个参数对First Better选择算法起作用，选择第一个有冲突的变元）
            selection.initialize(problem.n_variables(), value_of(0));

            // 依次考虑各个变元，将其冲突数的选择价值评估放入选择器（First Better算法可能提前结束循环）
            for (j = 0; j < problem.n_variables(); ++j) {

                // 将冲突数的选择价值评估提交给选择器
                selection.submit(value_of(problem.conflicts_of(permutation[j])));

                // 如果选择器已经找到目标则退出（对First Better算法起作用）
                // if (selection.done()){
                //     break;
                // }
            }

            // 选择到的变量下标
            selected_index = permutation[selection.selected_index()];
            int cur_conflict = problem.conflicts_of(selected_index);

            // 变量被替换前的值，用于恢复变量原有取值
            // old_variable = problem.variables()[selected_index];

            // 可用于替换该变量当前值的所有可能取值
            choices = problem.choices_of(selected_index);

            // 生成随机排列，用于打乱变元可替换值的顺序（对First Better算法起作用）
            permutation = RandomVariables::uniform_permutation(choices.size());

            // 初始化选择器，用于选择值，来替换选中的变元（第二个参数对First Better算法起作用，选择第一个产生冲突数小于选定变元当前取值的备选值）
            // 在value_of(x)为单调递增函数的情况下，value_of(-x)为单调递减函数，这可以为产生冲突少的新值赋予更高的权重
            selection.initialize(choices.size(), value_of(-problem.conflicts_of(selected_index)));

            // 依次考虑选中变元的各个替换值，将其价值评估
            for (j = 0; j < choices.size(); ++j) {

                // 将变元的值设置为替换值
                // problem.set_variable(selected_index, choices[permutation[j]]);

                // // TODO：将变元新值产生的冲突数的评估价值提交给选择器（需要参考selection, problem的接口）——1行
                // selection.submit(value_of(-problem.conflicts_of(selected_index)));

                // // 恢复变元的原值
                // problem.set_variable(selected_index, old_variable);
                selection.submit(value_of(-problem.conflicts_if_set(selected_index, choices[permutation[j]]) + cur_conflict));
            }

            // 将选中变元改变为选中的值
            problem.set_variable(selected_index, choices[permutation[selection.selected_index()]]);
        }
    }

public:

    ConflictMinimize(const ConstraintSatisfactionType& _problem) : problem(_problem) {}

    void search(int iterations, int max_steps, SelectionBase& selection,
        ConflictValueEstimatorType value_of = default_conflict_value_estimator) {

        for (int i = 0; i < iterations; ++i) {
            std::cout << "<begin>" << std::endl;
            // 随机重启
            problem.reset();

            // 尝试从随机重启的点开始优化，减少冲突数
            sample_path(max_steps, selection, value_of);

            // 若消灭全部冲突，则成功
            if (not problem.has_conflict()) {
                std::cout << "Successful search in try: " << i << std::endl;
                // problem.show();
                break;
            }
            else {
                std::cout << "Failed search: " << i << std::endl;
            }
            std::cout << "<end>" << std::endl;
        }
    }
};
