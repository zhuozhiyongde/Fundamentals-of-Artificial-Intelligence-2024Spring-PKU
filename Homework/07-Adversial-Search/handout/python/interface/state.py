from abc import ABCMeta, abstractmethod

# 状态接口，实现此接口可用于各种全局搜索算法BFS/DFS/heuristic等
class StateBase(metaclass=ABCMeta):
    
    # 当前状态下可选择的动作集
    @abstractmethod
    def action_space(self) -> list:
        raise NotImplementedError
    
    # 转移到当前状态的步骤的花费
    @abstractmethod
    def cost(self) -> float:
        raise NotImplementedError
    
    # 从初始状态出发转移到当前状态的总花费
    @abstractmethod
    def cumulative_cost(self) -> float:
        raise NotImplementedError
    
    # 判断当前状态是否为目标状态
    @abstractmethod
    def success(self) -> bool:
        raise NotImplementedError
    
    # 判断从当前状态是否已经不可能转移到目标状态
    @abstractmethod
    def fail(self) -> bool:
        raise NotImplementedError
    
    # 打印当前状态
    @abstractmethod
    def show(self) -> None:
        raise NotImplementedError
    
    # 从当前状态通过动作生成下一步状态
    @abstractmethod
    def next(self, action) -> "StateBase":
        raise NotImplementedError
    
    # ==运算符重载，用于判重
    @abstractmethod
    def __eq__(self, state:"StateBase") -> bool:
        raise NotImplementedError
    
    # <运算符重载，用于优先队列比较
    @abstractmethod
    def __lt__(self) -> bool:
        raise NotImplementedError
    
    # 用于搜索需要State可哈希
    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError
    
    