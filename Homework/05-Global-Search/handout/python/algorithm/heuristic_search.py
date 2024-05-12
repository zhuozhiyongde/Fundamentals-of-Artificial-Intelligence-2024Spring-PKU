from copy import deepcopy
from queue import PriorityQueue
from typing import Callable

from interface.state import StateBase
from utils.show_path import show_reversed_path

# 根据传入的状态估值函数启发式地搜索一个解
class HeuristicSearch:
    
    ValueEstimatorType = Callable[[StateBase], float]
    
    def __init__(self, state:StateBase):
        assert isinstance(state, StateBase)
        self.initial_state = deepcopy(state)
    
    # 传入状态估值函数
    def search(self, value_of:ValueEstimatorType) -> None:
        
        # 优先队列中估值高的状态在堆顶，先被访问
        states_queue = PriorityQueue()
        
        # 某个状态的最大估值（在最短路问题中为：最短估计距离*(-1) ）
        best_value_of = dict()
        
        # 记录状态的前一个状态，用于记录路径
        last_state_of = dict()
        
        states_queue.put((0, self.initial_state))
        best_value_of[self.initial_state] = 0
                
        while not states_queue.empty():
            _, state = states_queue.get()
            
            if state.success():
                break
            
            if state.fail():
                continue
            
            # 从开结点集中估值最高的状态出发尝试所有动作
            for action in state.action_space():
                
                # TODO：状态转移——1行
                
                # 如果从当前结点出发到达新结点所获得的估值高于新结点原有的估值，则更新
                if (new_state not in best_value_of 
                    or value_of(new_state) > best_value_of[new_state]):
                    # TODO：更新状态价值，扩展待访问的节点集——2行
                    # TODO：记录路径——1行
        
        if state.success():
            show_reversed_path(last_state_of, state)