from typing import List

from interface.population import PopulationBase

class GeneticAlgorithm:
    
    ChromosomeType = PopulationBase.ChromosomeType
    
    def __init__(self, init_population:PopulationBase):
        self.population = init_population
    
    def population(self) -> List[ChromosomeType]:
        return self.population.population()
    
    def adaptability(self) -> List[float]:
        return self.population.adaptability()
    
    def evolve(self, n:int) -> None:
        for i in range(n):
            print(f"<epoch {i} begin>")
            self.population.cross()
            self.population.mutate()
            self.population.show()
            print(f"<epoch {i} end>")