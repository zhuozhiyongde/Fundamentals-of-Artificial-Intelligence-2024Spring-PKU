from copy import deepcopy
from typing import List

from interface.state import StateBase

# n皇后状态类
class QueensState(StateBase):
    
    def __init__(self, n:int):
        
        # 棋盘边长
        self.n_queens = n
        
        # 已经填入皇后的行数
        self.current_row = 0
        
        # 每行皇后的位置
        self.queens:List[int] = [0] * n
        
        # 当前的空行可填入皇后的位置
        self._action_space:List[int] = list()
        
        # 各列，右左对角线，左右对角线是否已有皇后
        self.column:List[bool] = [False] * n
        self.right_left:List[bool] = [False] * (n * 2)
        self.left_right:List[bool] = [False] * (n * 2)
        
        self._update_action_space()
     
    def _which_right_left(self, c:int, r:int) -> int:
        return c + r
    
    def _which_left_right(self, c:int, r:int) -> int:
        return r - c + self.n_queens - 1
    
    # 判断当前空行的某个位置能否落子
    def _is_valid_action(self, action:int) -> bool:
        return (0 <= action < self.n_queens and self.column[action] == False
            and self.left_right[self._which_left_right(action, self.current_row)] == False
            and self.right_left[self._which_right_left(action, self.current_row)] == False
        )
    
    def _update_action_space(self) -> None:
        self._action_space.clear()
        for c in range(self.n_queens):
            if self._is_valid_action(c):
                self._action_space.append(c)
    
    def action_space(self) -> List[int]:
        return self._action_space

    def show(self) -> None:
        for r in range(self.n_queens):
            if r >= self.current_row:
                print()
                continue
            
            for c in range(self.n_queens):
                if self.queens[r] == c:
                    print("Q", sep="", end="")
                else:
                    print("+", sep="", end="")
            print()
            
        for r in range(self.n_queens):
            print(self.queens[r], sep=" ", end="")
        print()
    
    # 成功花费-1，失败花费1，中间状态花费为0
    def cost(self) -> float:
        if self.current_row == self.n_queens:
            return -1
        elif len(self._action_space) == 0:
            return 1
        return 0
    
    def cumulative_cost(self) -> float:
        return self.cost()
    
    def success(self) -> bool:
        return self.cost() < 0
    
    def fail(self) -> bool:
        return self.cost() > 0
    
    # 当前空行落子之后到达新的状态
    def next(self, action:int) -> "QueensState":
        next_state = deepcopy(self)
        assert next_state._is_valid_action(action)
        next_state.column[action] = True
        next_state.left_right[next_state._which_left_right(action, next_state.current_row)] = True
        next_state.right_left[next_state._which_right_left(action, next_state.current_row)] = True
        next_state.queens[next_state.current_row] = action
        next_state.current_row += 1
        next_state._update_action_space()
        return next_state
    
    _factorial:List[int] = [1]
    
    # Cantor展开，用于计算全排列的哈希值，不用关注这一部分
    def __hash__(self) -> int:
        if len(self._factorial) < self.current_row:
            for i in range(len(self._factorial), self.current_row+1):
                self._factorial.append(self._factorial[-1] * i)
        
        sum = 0
        
        for i in range(self.current_row):
            reverse = 0
            for j in range(i+1, self.current_row):
                if self.queens[j] < self.queens[i]:
                    reverse += 1
        
            sum += reverse * self._factorial[self.current_row-i-1]
        
        return sum
    
    # 重载==
    def __eq__(self, other:"QueensState") -> bool:
        return self.current_row == other.current_row and self.queens == other.queens and self.n_queens == other.n_queens
    
    # 重载<
    def __lt__(self, other:"QueensState") -> bool:
        return self.cumulative_cost() > other.cumulative_cost()
    
    
    
    

        