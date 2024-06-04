from copy import deepcopy

from interface.state_local import StateLocalBase
from utils.random_variables import RandomVariables
class QueensMoveState(StateLocalBase):
    
    def _add_queen(self, row:int, column:int, n:int=1) -> None:
        self.queens[row] = column
        self.conflicts += n * (self.column_count[column]
            + self.right_left_count[column + row]
            + self.left_right_count[column - row + self.n_queens - 1]
        )
        self.conflicts -= n * 3 if n < 0 else 0
        
        self.column_count[column] += n
        self.right_left_count[column + row] += n
        self.left_right_count[column - row + self.n_queens - 1] += n
    
    def __init__(self, n:int=0):
        self.n_queens = n
        self.reset()
    
    def reset(self) -> None:
        self.queens = RandomVariables.uniform_permutation(self.n_queens)
        self.column_count = [0] * self.n_queens
        self.left_right_count = [0] * (self.n_queens << 1)
        self.right_left_count = [0] * (self.n_queens << 1)
        self.conflicts = 0
        
        for i in range(self.n_queens):
            self._add_queen(i, self.queens[i])
    
    def neighbor_count(self) -> int:
        return self.n_queens ** 2
    
    def neighbor(self, neighbor_index:int) -> "QueensMoveState":
        row = neighbor_index // self.n_queens
        column = self.queens[row]
        to_column = neighbor_index % self.n_queens
        
        next_state = deepcopy(self)
        next_state._add_queen(row, column, -1)
        next_state._add_queen(row, to_column)
        
        return next_state
    
    def show(self) -> None:
        print(f"Queens: {self.queens}")
        print(f"Conflicts: {self.conflicts}\n")