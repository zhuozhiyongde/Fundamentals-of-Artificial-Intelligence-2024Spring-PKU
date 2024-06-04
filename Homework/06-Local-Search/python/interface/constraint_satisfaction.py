from abc import ABCMeta, abstractmethod
from typing import List, Union

class ConstraintSatisfactionBase(metaclass=ABCMeta):
    
    VariableType = Union[int, float]
    
    def __init__(self, n:int):
        pass
    
    @abstractmethod
    def n_variables(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def variables(self) -> List[VariableType]:
        raise NotImplementedError
    
    @abstractmethod
    def conflicts_of(self, variable_index:int) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def has_conflict(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def choices_of(self, variable_index:int) -> List[VariableType]:
        raise NotImplementedError
    
    @abstractmethod
    def set_variable(self, variable_index:int, new_value:VariableType):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def show(self) -> None:
        raise NotImplementedError