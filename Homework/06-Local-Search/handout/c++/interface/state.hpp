#pragma once

#include <vector>

// 用于搜索需要让State可哈希，以下提供一个哈希范例
/*
template <> 
struct std::hash<StateType> {
    size_t operator()(const StateType& x) const {
        return 0;
    }
};
*/

// 状态接口，实现此接口可用于各种全局搜索算法BFS/DFS/heuristic等
template<typename ActionType>
class StateBase{
public:
    
    StateBase() = default;
    virtual ~StateBase() = default;

    using ActionBaseType = ActionType ;

    // 当前状态下可选择的动作集
    virtual std::vector<ActionType> action_space() const = 0;
    
    // 转移到当前状态的步骤的花费
    virtual double cost() const = 0;
    
    // 从初始状态出发转移到当前状态的总花费
    virtual double cumulative_cost() const = 0;
    
    // 判断当前状态是否为目标状态
    virtual bool success() const = 0;

    // 判断从当前状态是否已经不可能转移到目标状态
    virtual bool fail() const = 0;

    // 打印当前状态
    virtual void show() const = 0;

    // 从当前状态通过动作生成下一步状态
    virtual const StateBase& next(const ActionType&) const = 0;

    // 状态哈希
    friend struct std::hash<StateBase>;

    // operator== 应当重载子类的该运算符，用于搜索中的状态判重
    friend bool operator== (const StateBase& s1, const StateBase& s2){
        return s1.cumulative_cost() == s2.cumulative_cost();
    }
};
