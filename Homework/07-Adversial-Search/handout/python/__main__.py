import json
from algorithm.monte_carlo_tree_search import MonteCarloTreeSearch
from problem.hex import HexState
 
N = 11

state = HexState(11)

# 解析读入的JSON
full_input = json.loads(input())

turn_id = len(full_input["responses"])
for i in range(turn_id):
    x = int(full_input["requests"][i]["x"])
    y = int(full_input["requests"][i]["y"])
    if x >= 0 and y >= 0:
        state = state.next(x * N + y)
    x = int(full_input["responses"][i]["x"])
    y = int(full_input["responses"][i]["y"])
    state = state.next(x * N + y)

x = int(full_input["requests"][turn_id]["x"])
y = int(full_input["requests"][turn_id]["y"])

if x >= 0 and y >= 0:
    state = state.next(x * N + y)
    forced_flag = False
elif "forced_x" in full_input["requests"][0]:
    forced_flag = True
else:
    forced_flag = False

mcts = MonteCarloTreeSearch(state)

if not forced_flag:
    action = mcts.select_action(150, 0.2)
else:
    action = 1 * 11 + 2

my_action = {"x": int(action)//11, "y": int(action)%11}
print(json.dumps({
    "response": my_action
}))