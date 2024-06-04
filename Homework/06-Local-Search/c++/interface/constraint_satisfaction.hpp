#pragma once

#include <cassert>
#include <vector>
#include <type_traits>

template<typename VariableType>
class ConstraintSatisfactionBase {
public:

    using VariableBaseType = VariableType;

    ConstraintSatisfactionBase() = default;
    virtual ~ConstraintSatisfactionBase() = default;

    static_assert(std::is_arithmetic<VariableType>::value, "Variable type not arithmetic.");

    virtual int n_variables() const = 0;

    // 约束满足问题的变元
    virtual std::vector<VariableType> variables() const = 0;

    // 每个变元不满足的约束个数
    virtual int conflicts_of(int variable_index) const = 0;

    // 是否仍然存在冲突
    virtual bool has_conflict() const = 0;

    // 每个变元取值的备选集
    virtual std::vector<VariableType> choices_of(int variable_index) const = 0;

    // 设置某个变元的值
    virtual void set_variable(int variable_index, VariableType new_value) = 0;

    // 重新开始
    virtual void reset() = 0;

    virtual void show() const = 0;
};