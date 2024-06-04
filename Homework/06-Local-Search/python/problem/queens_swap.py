from copy import deepcopy

from interface.state_local import StateLocalBase
from .queens_move import QueensMoveState


class QueensSwapState(StateLocalBase):
    
    def __init__(self, n:int):
        
        self.state = QueensMoveState(n)
        self.action_space = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def neighbor(self, neighbor_index:int) -> "QueensSwapState":
        row1, row2 = self.action_space[neighbor_index]
        
        next_state = deepcopy(self)
        next_state.state = next_state.state.neighbor(row1 * self.state.n_queens + self.state.queens[row2])
        next_state.state = next_state.state.neighbor(row2 * self.state.n_queens + self.state.queens[row1])
        
        return next_state
    
    def neighbor_count(self) -> int:
        return len(self.action_space)

    def reset(self) -> None:
        self.state.reset()
    
    def show(self) -> None:
        self.state.show()