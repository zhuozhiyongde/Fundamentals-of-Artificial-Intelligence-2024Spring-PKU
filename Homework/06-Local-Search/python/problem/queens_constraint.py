from typing import List

from interface.constraint_satisfaction import ConstraintSatisfactionBase
from utils.random_variables import RandomVariables
from .queens_move import QueensMoveState

class QueensConstraintSatisfaction(QueensMoveState, ConstraintSatisfactionBase):
    
    def __init__(self, n:int):
        super().__init__(n)
        self._choices = list(range(n))
    
    def n_variables(self) -> int:
        return self.n_queens

    def variables(self) -> List[int]:
        return self.queens
    
    def conflicts_of(self, variable_index:int) -> int:
        row, column = variable_index, self.queens[variable_index]
        return (self.column_count[column]
            + self.right_left_count[column + row]
            + self.left_right_count[column - row + self.n_queens - 1]
            - 3
        )
    
    def has_conflict(self) -> bool:
        return any(self.conflicts_of(i) for i in range(self.n_queens))
    
    def choices_of(self, variable_index:int) -> List[int]:
        return self._choices
    
    def set_variable(self, variable_index:int, new_variable:float) -> None:
        self._add_queen(variable_index, self.queens[variable_index], -1)
        self._add_queen(variable_index, new_variable)
    
    def show(self) -> None:
        QueensMoveState.show(self)
    
    def reset(self) -> None:
        QueensMoveState.reset(self)