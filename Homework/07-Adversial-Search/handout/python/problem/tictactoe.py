from typing import List, Tuple
from copy import deepcopy

from interface.game_state import NPlayerGameStateBase

class TicTacToeState(NPlayerGameStateBase):
    
    X, O = 0b01, 0b10

    def __init__(self):
        super().__init__(2)
        self._state = 0
        self._round = 0
        self._action_space = list()
        self._update_action_space()
    
    def _take_at(self, index:int) -> int:
        return (self._state >> (index << 1)) & 0b11
    
    def _put_at(self, index:int) -> None:
        piece = self.X if self.active_player() == 0 else self.O
        assert(index >= 0 and index < 9 and self._take_at(index) == 0)
        self._state |= (piece << (index << 1))
    
    def _full(self) -> bool:
        return self._round == 9
    
    def _x_win(self) -> bool:
        return self._test_lines_of(self.X)
    
    def _o_win(self) -> bool:
        return self._test_lines_of(self.O)

    _row_mask =    0b000000000000111111 
    _column_mask = 0b000011000011000011
    _lr_mask =     0b110000001100000011 
    _rl_mask =     0b000011001100110000
    
    _row_target = (X|(X<<2)|(X<<4))
    _column_target = (X|(X<<6)|(X<<12))
    _lr_target = (X|(X<<8)|(X<<16))
    _rl_target = ((X<<4)|(X<<8)|(X<<12))

    def _test_lines_of(self, piece:int) -> bool:
        
        if self._round < 5:
            return False
        
        row_target = (self._row_target << (piece==self.O))
        column_target = (self._column_target << (piece==self.O))
        lr_target = (self._lr_target << (piece==self.O))
        rl_target = (self._rl_target << (piece==self.O))

        return ((self._state & self._row_mask) == row_target
            or ((self._state >> 6) & self._row_mask) == row_target
            or ((self._state >> 12) & self._row_mask) == row_target
            or (self._state & self._column_mask) == column_target
            or ((self._state >> 2) & self._column_mask) == column_target
            or ((self._state >> 4) & self._column_mask) == column_target
            or (self._state & self._lr_mask) == lr_target
            or (self._state & self._rl_mask) == rl_target
        )
    
    def _update_action_space(self) -> None:
        self._action_space = [i for i in range(9) if self._take_at(i) == 0]
    
    def n_actions(self) -> int:
        return 9 - self._round
    
    def active_player(self) -> int:
        return self._round & 1
    
    def action_space(self) -> List[int]:
        return self._action_space
    
    def rewards(self) -> Tuple[float]:
        return (1,-1) if self._x_win() else (-1,1) if self._o_win() else (0, 0)
    
    def cumulative_rewards(self) -> Tuple[float]:
        return self.rewards()
    
    def done(self) -> bool:
        return self._full() or self._x_win() or self._o_win()
    
    def show(self) -> None:
        for i in range(3):
            for j in range(3):
                print("_XO"[self._take_at(i*3+j)], end="")
            print()
        print()
    
    def next(self, action:int) -> "TicTacToeState":
        next_state = deepcopy(self)
        next_state._put_at(action)
        next_state._update_action_space()
        next_state._round += 1
        return next_state

    def __eq__(self, other:"TicTacToeState") -> bool:
        return self._state == other._state
    
    def __hash__(self) -> int:
        return hash(self._state)

class TicTacToe4State(NPlayerGameStateBase):
    
    X, O = 0b01, 0b10

    def __init__(self):
        self._state = 0
        self._round = 0
        self._action_space = list()
        self._update_action_space()
    
    def _take_at(self, index:int) -> int:
        return (self._state >> (index << 1)) & 0b11
    
    def _put_at(self, index:int) -> None:
        piece = self.X if self.active_player() == 0 else self.O
        assert(index >= 0 and index < 16 and self._take_at(index) == 0)
        self._state |= (piece << (index << 1))
    
    def _full(self) -> bool:
        return self._round == 16
    
    def _x_win(self) -> bool:
        return self._test_lines_of(self.X)
    
    def _o_win(self) -> bool:
        return self._test_lines_of(self.O)

    _row_mask =    0b00000000000000000000000011111111
    _column_mask = 0b00000011000000110000001100000011
    _lr_mask =     0b11000000001100000000110000000011 
    _rl_mask =     0b00000011000011000011000011000000
    
    _row_target = (X|(X<<2)|(X<<4)|(X<<6))
    _column_target = (X|(X<<8)|(X<<16)|(X<<24))
    _lr_target = (X|(X<<10)|(X<<20)|(X<<30))
    _rl_target = ((X<<6)|(X<<12)|(X<<18)|(X<<24))

    def _test_lines_of(self, piece:int) -> bool:
        
        if self._round < 7:
            return False
        
        row_target = (self._row_target << (piece==self.O))
        column_target = (self._column_target << (piece==self.O))
        lr_target = (self._lr_target << (piece==self.O))
        rl_target = (self._rl_target << (piece==self.O))

        return ((self._state & self._row_mask) == row_target
            or ((self._state >> 8) & self._row_mask) == row_target
            or ((self._state >> 16) & self._row_mask) == row_target
            or ((self._state >> 24) & self._row_mask) == row_target
            or (self._state & self._column_mask) == column_target
            or ((self._state >> 2) & self._column_mask) == column_target
            or ((self._state >> 4) & self._column_mask) == column_target
            or ((self._state >> 6) & self._column_mask) == column_target
            or (self._state & self._lr_mask) == lr_target
            or (self._state & self._rl_mask) == rl_target
        )
    
    def _update_action_space(self) -> None:
        self._action_space = [i for i in range(16) if self._take_at(i) == 0]
    
    def n_actions(self) -> int:
        return 16 - self._round
    
    def active_player(self) -> int:
        return self._round & 1
    
    def action_space(self) -> List[int]:
        return self._action_space
    
    def rewards(self) -> List[float]:
        return (1,-1) if self._x_win() else (-1,1) if self._o_win() else (0, 0)
    
    def cumulative_rewards(self) -> List[float]:
        return self.rewards()
    
    def done(self) -> bool:
        return self._full() or self._x_win() or self._o_win()
    
    def show(self) -> None:
        for i in range(4):
            for j in range(4):
                print("_XO"[self._take_at(i*4+j)], end="")
            print()
        print()
    
    def next(self, action:int) -> "TicTacToe4State":
        next_state = deepcopy(self)
        next_state._put_at(action)
        next_state._update_action_space()
        next_state._round += 1
        return next_state

    def __eq__(self, other:"TicTacToe4State") -> bool:
        return self._state == other._state
    
    def __hash__(self) -> int:
        return hash(self._state)
