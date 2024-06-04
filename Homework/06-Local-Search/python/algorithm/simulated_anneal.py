from typing import Callable
from copy import deepcopy
from math import exp

from interface.state_local import StateLocalBase
from utils.random_variables import RandomVariables

class SimulatedAnneal:
    
    ValueEstimatorType = Callable[[StateLocalBase], float]
    TemperatureScheduleType = Callable[[int], float]
    
    def __init__(self, state:StateLocalBase):
        self._initial_state = deepcopy(state)
    
    def _sample_path(self, value_of:ValueEstimatorType,temperature_at:TemperatureScheduleType) -> StateLocalBase:
        
        state = deepcopy(self._initial_state)
        temperature = temperature_at(0)
        
        t = 0
        while state.neighbor_count() > 0 and temperature >= self._terminate_temperature:
            index = RandomVariables.uniform_int() % state.neighbor_count()
            new_state = state.neighbor(index)
            value_diff = value_of(new_state) - value_of(state)
            temperature = temperature_at(t)
            if # TODO: 如果新值更高，直接接受，若更低，则以exp(value_diff / temperature)概率接受
                state = new_state
            t += 1
        
        return state
    
    def search(self, value_of:ValueEstimatorType, temperature_at:TemperatureScheduleType,
        iterations:int, target_value:float, terminate_temperature:float=1e-10):
        
        self._terminate_temperature = terminate_temperature
        
        for i in range(iterations):
            print("<begin>")
            state = self._sample_path(value_of, temperature_at)
            state_value = value_of(state)
            
            if state_value >= target_value:
                print(f"Successful search: {i}")
                state.show()
                break
            else:
                print(f"Failed search: {i}")
            
            print(f"Value: {state_value}")          
            print("<end>")
