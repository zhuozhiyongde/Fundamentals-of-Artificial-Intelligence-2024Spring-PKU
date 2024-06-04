#pragma once

#include <vector>

// 游戏状态接口
template<typename ActionType>
class GameStateBase{
public:
    
    GameStateBase() = default;
    virtual ~GameStateBase() = default;

    using ActionBaseType = ActionType;

    virtual int n_players() const = 0;
    
    virtual int active_player() const = 0;

    virtual int n_actions() const = 0;

    // 当前状态下的动作空间
    virtual std::vector<ActionType> action_space() const = 0;
    
    // 各个玩家在之前的游戏过程中累计的回报
    virtual std::vector<double> cumulative_rewards() const = 0;
    
    // 各个玩家在转移到当前状态后获得的即时回报
    virtual std::vector<double> rewards() const = 0;

    // 游戏是否结束
    virtual bool done() const = 0;

    virtual void show() const = 0;

    // 当前玩家选择action后转移到的新状态
    virtual const GameStateBase& next(const ActionType&) const = 0;

    // 状态哈希
    friend struct std::hash<GameStateBase>;

    // 子类应重载operator==
    friend bool operator== (const GameStateBase& s1, const GameStateBase& s2){
        return s1.cumulative_rewards()[s1.active_player()] == s2.cumulative_rewards()[s2.active_player()];
    }
};

// N人游戏状态接口
template<typename ActionType, int n>
class NPlayerGameStateBase : public GameStateBase<ActionType>{
protected:
    static constexpr int _n_players = n;
public:
    NPlayerGameStateBase() = default;
    virtual ~NPlayerGameStateBase() = default;

    int n_players() const override {return _n_players;}    
};


#include "state.hpp"

// 将全局搜索问题的环境包装为单人游戏环境
template<typename StateType>
class GameStateWrapper : public NPlayerGameStateBase<typename StateType::ActionBaseType, 1> {
private:

    StateType _state;
    using ActionType = typename StateType::ActionBaseType;
    static_assert(std::is_base_of<StateBase<ActionType>, StateType>::value, "StateType not derived from StateBase.");

public:

    GameStateWrapper() = default;
    GameStateWrapper(const StateType& state) : _state(state) {}

    int active_player() const override {return 0;}
    
    int n_actions() const override {return _state.n_actions();}

    std::vector<ActionType> action_space() const override {
        return _state.action_space();
    }
    
    std::vector<double> rewards() const override {
        std::vector<double> rewards;
        double reward = -_state.cost();
        rewards.push_back(reward);
        return rewards;
    }

    std::vector<double> cumulative_rewards() const override {
        std::vector<double> cumulative_rewards;
        double cumulative_reward = -_state.cumulative_cost();
        cumulative_rewards.push_back(cumulative_reward);
        return cumulative_rewards;
    }
    
    bool done() const override {
        return _state.success() or _state.fail();
    }
    
    void show() const override {
        _state.show();
    }
    
    const GameStateWrapper& next(const ActionType& action) const override {
        static GameStateWrapper next_state;
        next_state._state = _state.next(action);
        return next_state;
    }

    friend struct std::hash<GameStateWrapper>;

    friend bool operator== (const GameStateWrapper& s1, const GameStateWrapper& s2){
        return s1._state == s2._state;
    }

    friend bool operator< (const GameStateWrapper& s1, const GameStateWrapper& s2){
        return s1._state < s2._state;
    }
};

template<typename ActionType>
struct std::hash<GameStateWrapper<ActionType> >{
    size_t operator()(const GameStateWrapper<ActionType>& s) const {
        std::hash<decltype(s._state)> hasher;
        return hasher(s._state);
    }
};
