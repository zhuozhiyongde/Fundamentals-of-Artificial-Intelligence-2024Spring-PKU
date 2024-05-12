from interface.state import StateBase
from .heuristic_search import HeuristicSearch

# 一致代价搜索一个解
class UniformCostSearch:
    
    def __init__(self, state:StateBase):
        self._heuristic_search = HeuristicSearch(state)
    
    # 一致代价搜索即为仅使用：到达当前状态的总花费*(-1) 作为状态估值的启发式搜索
    # 可以理解为搜到目标就结束的dijkstra算法
    def search(self) -> None:
        self._heuristic_search.search(lambda state: -state.cumulative_cost())