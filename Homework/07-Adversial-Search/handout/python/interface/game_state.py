from typing import List, Tuple
from abc import ABCMeta, abstractmethod

class GameStateBase(metaclass=ABCMeta):
        
    @abstractmethod
    def n_players(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def active_player(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def n_actions(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def action_space(self) -> List:
        raise NotImplementedError
    
    @abstractmethod
    def cumulative_rewards(self) -> Tuple[float]:
        raise NotImplementedError
    
    @abstractmethod
    def rewards(self) -> Tuple[float]:
        raise NotImplementedError
    
    @abstractmethod
    def done(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def show(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def next(self, action) -> "GameStateBase":
        raise NotImplementedError
    
    @abstractmethod
    def __eq__(self, other:"GameStateBase") -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

class NPlayerGameStateBase(GameStateBase, metaclass=ABCMeta):
    
    def __init__(self, n:int):
        self._n_players = n
    
    def n_players(self) -> int:
        return self._n_players

from .state import StateBase

class GameStateWrapper(NPlayerGameStateBase):
    
    def __init__(self, state:StateBase):
        self.state = state
        
    def active_player(self) -> int: return 0
    
    def n_actions(self) -> int: return len(self.state.action_space())

    def action_space(self) -> List: return self.state.action_space()
    
    def rewards(self) -> Tuple[float]: return (-self.state.cost(),)
    
    def cumulative_rewards(self) -> Tuple[float]: return (self.state.cumulative_cost(),)
    
    def done(self) -> bool: return self.state.success() or self.state.fail()
    
    def show(self) -> None: self.state.show()
    
    def next(self, action) -> "GameStateWrapper":
        next_state = self.state.next(action)
        return next_state
    
    def __hash__(self) -> int:
        return hash(self.state)
    
    def __eq__(self, other:"GameStateWrapper") -> bool:
        return self.state == other.state
    
    def __lt__(self, other:"GameStateWrapper") -> bool:
        return self.state < other.state
    