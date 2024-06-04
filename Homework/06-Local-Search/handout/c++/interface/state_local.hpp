#pragma once

#include <vector>

#include "state.hpp"

// 状态接口：实现此接口可用于局部搜索算法（包括爬山和模拟退火，但不包括最小化冲突和遗传算法，因为这两者基于不同的问题建模）
class StateLocalBase{
public:
    
    StateLocalBase() = default;
    virtual ~StateLocalBase() = default;

    // 生成初始状态
    virtual void reset() = 0;

    // 邻居节点的个数
    virtual int neighbor_count() const = 0;

    // 按照编号生成邻居节点
    virtual const StateLocalBase& neighbor(int neighbor_index) const = 0;

    // 打印状态
    virtual void show() const = 0;
};


// 将全局搜索模型的状态包装为局部搜索模型的状态
template<typename StateType>
class StateLocalWrapper : public StateLocalBase{
private:
    
    StateType _state;
    using ActionType = typename StateType::ActionBaseType;
    static_assert(std::is_base_of<StateBase<ActionType>, StateType>::value, "StateType not derived from StateBase.");

public:

    StateLocalWrapper() = default;
    StateLocalWrapper(const StateType& state) : _state(state) {}
    
    inline const StateType& state() const {return _state;}

    inline void reset() override {}

    inline int neighbor_count() const override {
        return _state.action_space().size();
    }

    const StateLocalWrapper& neighbor(int neighbor_index) const override {
        static StateLocalWrapper next_state;
        next_state._state = _state.next(_state.action_space()[neighbor_index]);
        return next_state;
    }

    void show() const override {
        _state.show();
    }
};