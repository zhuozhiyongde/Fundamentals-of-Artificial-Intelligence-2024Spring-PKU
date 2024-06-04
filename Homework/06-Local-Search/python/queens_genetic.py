from time import time
from typing import List

from interface.population import PopulationBase
from utils.random_variables import RandomVariables
from algorithm.genetic import GeneticAlgorithm

class QueensPopulation(PopulationBase):
    
    ChromosomeType = List[int]
    
    def __init__(self, p:List[ChromosomeType], mutation_rate:float):
        self._population = p
        self._mutation_rate = mutation_rate
        self._adaptability = [0] * len(p)
        self._update_adaptability()
        self._best_index, self._second_index = 0, 0
        
    
    def _evaluate_chromosome(self, c:ChromosomeType) -> float:
        
        column = [0] * len(c)
        lr = [0] * (len(c) << 1)
        rl = [0] * (len(c) << 1)
        
        conflicts = 0
        
        for i in range(len(c)):
            column[c[i]] += 1
            rl[c[i]+i] += 1
            lr[c[i]-i+len(c)-1] += 1
            
        conflicts = sum(column[c[i]] + rl[c[i]+i] + lr[c[i]-i+len(c)-1] - 3 for i in range(len(c)))
        
        return -conflicts

    def _update_adaptability(self) -> None:
        self._adaptability = [self._evaluate_chromosome(c) for c in self._population]
    
    def _select_chromosome_with_adaptability(self) -> ChromosomeType:
        coin = RandomVariables.uniform_int() & 1
        return self._population[self._second_index] if coin == 0 else self._population[self._best_index]
    
    @staticmethod
    def _cross(c1:ChromosomeType, c2:ChromosomeType) -> ChromosomeType:
        index = RandomVariables.uniform_int() % len(c1)
        result = c1[:index] + c2[index:]
        return result
    
    @staticmethod
    def _mutate(c:ChromosomeType) -> None:
        index = RandomVariables.uniform_int() % len(c)
        c[index] = RandomVariables.uniform_int() % len(c)
    
    def _update_best_and_second(self) -> None:
        for i in range(len(self._adaptability)):
            if self._adaptability[i] > self._adaptability[self._best_index]:
                self._second_index = self._best_index
                self._best_index = i
            elif self._adaptability[i] > self._adaptability[self._second_index]:
                self._second_index = i

    def population(self) -> List[ChromosomeType] : return self._population
    
    def adaptability(self) -> List[float] : return self._adaptability
    
    def cross(self) -> None:
        self._update_best_and_second()
        
        self._population = [
            self._cross(self._select_chromosome_with_adaptability(), self._select_chromosome_with_adaptability())
            for i in range(len(self._population))
        ]
        
        self._update_adaptability()
    
    def mutate(self) -> None:
        
        for i in range(len(self._population)):
            if RandomVariables.uniform_real() > self._mutation_rate:
                continue
            
            self._mutate(self._population[i])
        
        self._update_adaptability()
    
    def show(self) -> None:
        for i in range(len(self._population)):
            print(f"<{i}> Adaptability: {self._adaptability[i]}")
            
            if self._adaptability[i] == 0:
                print(f"Queens: {self._population[i]}")

if __name__ == "__main__":
    t0 = time()
        
    n = 20
    p = [RandomVariables.uniform_permutation(n) for i in range(n << 3)]
    pop = QueensPopulation(p, 0.9)
    
    gen = GeneticAlgorithm(pop)
    
    gen.evolve(n << 3)
    
    print(f"Total time: {time() - t0}")
        