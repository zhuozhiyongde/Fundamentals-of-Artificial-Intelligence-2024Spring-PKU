from copy import deepcopy
from queue import LifoQueue

from interface.state import StateBase
from utils.show_path import show_reversed_path


# 深度优先搜索全部解
class DepthFirstSearch:
    def __init__(self, state: StateBase):
        assert isinstance(state, StateBase)
        self.initial_state = deepcopy(state)

    # tree_search决定是否判断重复状态，require_path决定是否记录路径
    def search(self, tree_search: bool = True, require_path: bool = True) -> None:
        # 记录：(当前状态, 当前状态已经访问的子节点个数)
        states_stack = LifoQueue()

        # 记录状态的前一个状态，用于记录路径
        last_state_of = dict()

        # 防止重复状态，判断哪些状态已经访问过
        explored_states = set()

        states_stack.put((self.initial_state, 0))
        explored_states.add(self.initial_state)
        queue_peek_size = 0

        while not states_stack.empty():
            state, action_id = states_stack.get()

            if queue_peek_size < states_stack.qsize():
                queue_peek_size = states_stack.qsize()

            if state.success():
                # if require_path:
                #     show_reversed_path(last_state_of, state)
                # else:
                #     state.show()
                continue

            if state.fail():
                continue

            # 如果当前状态还有动作未尝试过
            if action_id < len(state.action_space()):
                states_stack.put((state, action_id + 1))

                # 尝试当前动作，获得下一步状态
                new_state = state.next(state.action_space()[action_id])

                if tree_search:
                    states_stack.put((new_state, 0))
                    if require_path:
                        last_state_of[new_state] = state
                elif new_state not in explored_states:
                    states_stack.put((new_state, 0))
                    if require_path:
                        last_state_of[new_state] = state
        
        print(f"Queue peek size: {queue_peek_size}")
