#pragma once

#include <iostream>
#include <stack>
#include <unordered_map>

template<typename StateType>
void show_reversed_path(const std::unordered_map<StateType, StateType>& last_state_of, const StateType& state){
    StateType s = state;
    std::stack<StateType> path;
    while (last_state_of.find(s) != last_state_of.end()){
        path.push(s);
        s = last_state_of.at(s);
    }
    path.push(s);

    std::cout << "<begin>" << std::endl;
    while (not path.empty()){
        s = path.top();
        path.pop();
        s.show();
    }
    std::cout << "<end>" << std::endl;
}

template<typename StateType>
void show_path(const std::unordered_map<StateType, StateType>& next_state_of, const StateType& state){

    StateType s;
    std::cout << "<begin>" << std::endl;
    for (s = state; next_state_of.find(s) != next_state_of.end(); s = next_state_of.at(s)){
        s.show();
    }
    s.show();
    std::cout << "<end>" << std::endl;
}
