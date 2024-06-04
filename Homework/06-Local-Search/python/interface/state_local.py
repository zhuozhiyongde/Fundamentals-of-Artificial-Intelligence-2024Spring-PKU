from abc import ABCMeta, abstractmethod


class StateLocalBase(metaclass=ABCMeta):
    
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def neighbor_count(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def neighbor(self, neighbor_index:int) -> "StateLocalBase":
        raise NotImplementedError
    
    @abstractmethod
    def show(self) -> None:
        raise NotImplementedError
    
class StateLocalWrapper(StateLocalBase):
    
    def __init__(self, state:StateLocalBase):
        self._state = state
    
    def state(self) -> StateLocalBase: return self._state
    
    def reset(self) -> None:
        pass
    
    def neighbor_count(self) -> int:
        return len(self._state.action_space())
    
    def neighbor(self, neighbor_index:int) -> "StateLocalBase":
        next_state = StateLocalWrapper(self._state.next(self._state.action_space()[neighbor_index]))
        return next_state
    
    def show(self) -> None:
        self._state.show()