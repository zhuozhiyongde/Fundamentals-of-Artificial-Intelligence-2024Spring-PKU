from typing import Callable
from copy import deepcopy

from interface.state_local import StateLocalBase
from utils.selection import SelectionBase
from utils.random_variables import RandomVariables

class HillClimb:
    
    ValueEstimatorType = Callable[[StateLocalBase], float]
    
    def __init__(self, state:StateLocalBase):
        self._initial_state = deepcopy(state)
    
    def _sample_path(self, value_of:ValueEstimatorType,
        target_value:float, max_steps:int, selection:SelectionBase) -> StateLocalBase:
        
        state = deepcopy(self._initial_state)
        
        for i in range(max_steps):
            if state.neighbor_count() == 0 or value_of(state) >= target_value:
                break
            
            selection.initialize(state.neighbor_count(), value_of(state))
            permutation = RandomVariables.uniform_permutation(state.neighbor_count())
            for j in range(state.neighbor_count()):
                selection.submit(value_of(state.neighbor(permutation[j])))
                if selection.done():
                    break
            
            # TODO：状态转移（需要参考state，selection的接口）——1行

        
        return state
            
        
        
    
    def search(self, value_of:ValueEstimatorType, target_value:float,
        max_steps:int, selection:SelectionBase, iterations:int):
        
        for i in range(iterations):
            print("<begin>")

            self._initial_state.reset()
            state = self._sample_path(value_of, target_value, max_steps, selection)
            state_value = value_of(state)
            
            if state_value >= target_value:
                print(f"Successful search: {i}")
                state.show()
                break
            else:
                print(f"Failed search: {i}")
                
            print(f"Value: {state_value}")
            print("<end>")
        
        
    
    