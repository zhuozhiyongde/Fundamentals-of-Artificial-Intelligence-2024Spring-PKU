from typing import List

from interface.state import StateBase

# 存储有向图的类
class DirectedGraph:
    
    def __init__(self, n_nodes:int):
        
        # 图顶点个数
        self.n_nodes : List[int] = n_nodes
        
        # 各条边的终点
        self.edge_end : List[int] = list()
        
        # 各条边的权重
        self.weight : List[float] = list()
        
        # _first[u]表示以顶点u为起点的第1条边的编号
        self.first : List[int] = [None] * n_nodes
        
        # _next[e]表示和编号为e的边起点相同的下一条边的编号
        self.next : List[int] = list()
    
    # 向图中添加边
    def add_edge(self, start_node:int, end_node:int, weight:float) -> None:
        self.edge_end.append(end_node)
        self.weight.append(weight)
        self.next.append(self.first[start_node])
        self.first[start_node] = len(self.edge_end) - 1
    
    # 返回起点为node的所有边的编号
    def edge_indexes_starting_from(self, node:int) -> List[int]:
        edge_indexes:List[int] = list()
        
        index = self.first[node]
        while index is not None:
            edge_indexes.append(index)
            index = self.next[index]
        
        return edge_indexes
    
# 有向图寻径问题的状态类
class DirectedGraphState(StateBase):
    
    def __init__(self, graph:DirectedGraph, current_node:int, target_node:int):
        self.graph = graph
        self.current_node = current_node
        self.target_node = target_node
        self.last_edge_index = None
        self._cumulative_cost:float = 0
    
    # 上一步花费即为进入当前顶点的边的权值
    def cost(self) -> float:
        return 0 if self.last_edge_index is None else self.graph.weight[self.last_edge_index]

    # 到达当前点的总花费
    def cumulative_cost(self) -> float:
        return self._cumulative_cost
    
    # 下一步能走哪些边（编号）
    def action_space(self) -> List[int]:
        return self.graph.edge_indexes_starting_from(self.current_node)
    
    def success(self) -> bool:
        return self.current_node == self.target_node
    
    def fail(self) -> bool:
        return not self.success() and self.graph.first[self.current_node] is None
    
    def next(self, action:int) -> "DirectedGraphState":
        next_state = DirectedGraphState(self.graph, self.graph.edge_end[action], self.target_node)
        next_state.last_edge_index = action
        next_state._cumulative_cost = self._cumulative_cost + self.graph.weight[action]
        return next_state
    
    def show(self) -> None:
        print(f"At: {self.current_node}, from edge {self.last_edge_index}, distance: {self._cumulative_cost}")
        
    def __hash__(self) -> int:
        return self.current_node
    
    # 判断重复状态，如果有环可以防止搜索死循环
    def __eq__(self, other:"DirectedGraphState") -> bool:
        return (self.current_node == other.current_node
            and self.target_node == other.target_node
            and self.graph == other.graph)
    
    # 重载<
    def __lt__(self, state:"DirectedGraphState") -> bool:
        return self._cumulative_cost > state._cumulative_cost