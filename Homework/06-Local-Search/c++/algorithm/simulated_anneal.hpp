#pragma once

#include <ctime>
#include <cmath>
#include <random>
#include <type_traits>
#include <iostream>

#include "../interface/state_local.hpp"
#include "../utils/random_variables.hpp"

template <typename StateLocalType>
class SimulatedAnneal {
private:

    using ValueEstimatorType = double (*) (const StateLocalType&);
    using TemperatureScheduleType = double (*) (int);

    static_assert(std::is_base_of<StateLocalBase, StateLocalType>::value, "StateLocalType not derived from StateLocalBase.");

    double terminate_temperature;
    StateLocalType initial_state;

    // 尝试一次模拟退火
    const StateLocalType& sample_path(ValueEstimatorType value_of, TemperatureScheduleType temperature_at) {

        static StateLocalType state, new_state;
        double value_diff, temperature;
        int index;

        state = initial_state;
        temperature = temperature_at(0);

        // 终止条件：存在邻居状态+温度条件
        for (int t = 0; state.neighbor_count() > 0 and temperature >= terminate_temperature; ++t) {

            index = RandomVariables::uniform_int() % state.neighbor_count();

            new_state = state.neighbor(index);

            value_diff = value_of(new_state) - value_of(state);
            temperature = temperature_at(t);

            /********TODO: 如果新值更高，直接接受，若更低，则以exp(value_diff / temperature)概率接受*******/
            if (value_diff > 0 || RandomVariables::uniform_real() < std::exp(value_diff / temperature)) {
                state = new_state;
            }
        }

        return state;
    }

public:

    SimulatedAnneal(const StateLocalType& state) : initial_state(state) {}

    void search(ValueEstimatorType value_of, TemperatureScheduleType temperature_at,
        int iterations, double target_value, double _terminate_temperature = 1e-10) {

        StateLocalType state;
        double state_value;

        terminate_temperature = _terminate_temperature;

        for (int i = 0; i < iterations; ++i) {

            std::cout << "<begin>" << std::endl;

            state = sample_path(value_of, temperature_at);
            state_value = value_of(state);

            if (state_value >= target_value) {
                std::cout << "Successful search: " << i << std::endl;
                state.show();
                break;
            }
            else {
                std::cout << "Failed search: " << i << std::endl;
            }

            std::cout << "Value: " << state_value << std::endl;
            std::cout << "<end>" << std::endl;
        }
    }
};