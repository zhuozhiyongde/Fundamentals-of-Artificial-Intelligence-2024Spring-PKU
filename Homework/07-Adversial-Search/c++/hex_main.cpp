#include <iostream>
#include "jsoncpp/json.h" // C++编译时默认包含此库 

#include "problem/hex.hpp"
#include "algorithm/monte_carlo_tree_search.hpp"

using namespace std;

using namespace std;

// 棋盘大小
const int N = 11;

// UCT探索项系数
const double exploration = 0.2;

// 迭代次数
const int iterations = 5000;

// 搜索生成state下的动作
Json::Value get_next_action(const HexState<N>& state, bool forced_flag){
    MonteCarloTreeSearch<HexState<N> > mcts(state);
    int action = mcts.select_action(iterations, exploration);

    if (forced_flag){
        action = 1*11+2;
    }

	Json::Value action_json;
	action_json["x"] = action / N;
	action_json["y"] = action % N;
	return action_json;
}

int main(){
       
    HexState<N> state;
	
    // 读入JSON
	string str;
	getline(cin, str);
	Json::Reader reader;
	Json::Value input;
	reader.parse(str, input); 
	
    // 分析自己收到的输入和自己过往的输出，并恢复状态
	int turn_id = input["responses"].size();
    int x, y;
    bool forced_flag;
	for (int i = 0; i < turn_id; i++) {
		x = input["requests"][i]["x"].asInt();
        y = input["requests"][i]["y"].asInt();
        if (x >= 0 and y >= 0){
            state = state.next(x * N + y);
        }
        x = input["responses"][i]["x"].asInt();
        y = input["responses"][i]["y"].asInt();
        state = state.next(x * N + y);
	}
    x = input["requests"][turn_id]["x"].asInt();
    y = input["requests"][turn_id]["y"].asInt();

    if (x >= 0 and y >= 0){
        state = state.next(x * N + y);
        forced_flag = false;
    } else if (input["requests"][0].isMember("forced_x")){
        forced_flag = true;
    } else {
        forced_flag = false;
    }

    // 做出决策存为action 
	// 输出决策JSON
	Json::Value result;
	
    result["response"] = get_next_action(state, forced_flag);

	Json::FastWriter writer;
	
    cout << writer.write(result) << endl;
	
    return 0;
}