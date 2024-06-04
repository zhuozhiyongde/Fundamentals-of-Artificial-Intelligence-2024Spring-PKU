from typing import List, Tuple
from copy import deepcopy

from interface.game_state import NPlayerGameStateBase
from utils.union_find_set import UnionFindSet

class HexState(NPlayerGameStateBase):
    R, B = 0b01, 0b10
    
    def __init__(self, n:int):
        super().__init__(2)
        self._n = n
        self._steps = 0
        self._board = [0] * (n**2)
        self._r_detector = UnionFindSet(n**2+2)
        self._b_detector = UnionFindSet(n**2+2)
        for i in range(n):
            self._r_detector.join(n**2, i)
            self._r_detector.join(n**2+1, n**2-1-i)
            self._b_detector.join(n**2, n*i)
            self._b_detector.join(n**2+1, n*(i+1)-1)
    
    def _r_win(self) -> bool:
        return self._r_detector.find(self._n**2) == self._r_detector.find(self._n**2+1)

    def _b_win(self) -> bool:
        return self._b_detector.find(self._n**2) == self._b_detector.find(self._n**2+1)
    
    def done(self) -> bool:
        return self._r_win() or self._b_win()
    
    def active_player(self) -> int:
        return self._steps & 1
    
    def rewards(self) -> Tuple[float]:
        return (1, -1) if self._r_win() else (-1, 1) if self._b_win() else (0, 0)
    
    def cumulative_rewards(self) -> Tuple[float]:
        return self.rewards()
    
    def n_actions(self) -> int:
        return self._n ** 2 - self._steps
    
    def action_space(self) -> List[int]:
        return [i for i in range(self._n**2) if self._board[i] == 0]
    
    def next(self, action:int):
        assert self._board[action] == 0
        next_state = deepcopy(self)
        next_state._board[action] = self.R if self.active_player() == 0 else self.B
        
        neighbors = [
            action-self._n, action+1-self._n,
            action+1, action+self._n,
            action-1+self._n, action-1
        ]
        
        not_top = action >= self._n
        not_bottom = action < self._n**2-self._n
        not_left = action % self._n != 0
        not_right = action % self._n != self._n - 1
        
        conditions = [
            not_top, not_top and not_right,
            not_right, not_bottom,
            not_bottom and not_left, not_left
        ]
        
        detector = next_state._r_detector if self.active_player() == 0 else next_state._b_detector
        
        for cond, neighbor in zip(conditions, neighbors):
            if cond and next_state._board[neighbor] == next_state._board[action]:
                detector.join(neighbor, action)
        
        next_state._steps += 1
        return next_state

    def show(self) -> None:
        for i in range(self._n):
            print(" " * i, end="")
            for j in range(self._n):
                print("_XO"[self._board[i*self._n+j]], end=" ")
            print()
    
    def __eq__(self, other:"HexState") -> bool:
        return self._board == other._board
    
    def __hash__(self) -> int:
        code = 0
        for i in range(self._n**2):
            code ^= self._board[i] << (i & 0b111111); 
            code &= 0xffffffffffffffff
        return code