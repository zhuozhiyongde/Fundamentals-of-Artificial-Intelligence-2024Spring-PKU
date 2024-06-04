from abc import ABCMeta, abstractmethod
from typing import List, Any

class PopulationBase(metaclass=ABCMeta):
    
    ChromosomeType = Any    
    
    @abstractmethod
    def population(self) -> List[ChromosomeType]:
        raise NotImplementedError

    @abstractmethod
    def adaptability(self) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def show(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def cross(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def mutate(self) -> None:
        raise NotImplementedError