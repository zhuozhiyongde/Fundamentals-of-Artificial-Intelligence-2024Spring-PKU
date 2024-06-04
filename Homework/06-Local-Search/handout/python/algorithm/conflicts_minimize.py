from typing import Callable
from math import exp

from interface.constraint_satisfaction import ConstraintSatisfactionBase
from utils.selection import SelectionBase
from utils.random_variables import RandomVariables

class ConflictMinimize:
    VariableType = ConstraintSatisfactionBase.VariableType
    ConflictValueEstimatorType = Callable[[int], float]
    
    
   
    default_conflict_value_estimator = lambda conflicts: exp(conflicts)
    
    
    def __init__(self, problem:ConstraintSatisfactionBase):
        self._problem = problem
    
    
    def _sample_path(self, max_steps:int, selection:SelectionBase, value_of:ConflictValueEstimatorType) -> None:
        
        for i in range(max_steps):
            if not self._problem.has_conflict():
                break
            permutation = RandomVariables.uniform_permutation(self._problem.n_variables())
            selection.initialize(self._problem.n_variables(), value_of(0))
            
            for j in range(self._problem.n_variables()):
                selection.submit(value_of(self._problem.conflicts_of(permutation[j])))
                if selection.done():
                    break
            
            selected_index = permutation[selection.selected_index()]
            old_variable = self._problem.variables()[selected_index]
            choices = self._problem.choices_of(selected_index)
            
            permutation = RandomVariables.uniform_permutation(len(choices))
            selection.initialize(len(choices), value_of(-self._problem.conflicts_of(selected_index)))
            
            for j in range(len(choices)):
                self._problem.set_variable(selected_index, choices[permutation[j]])
                # TODO：将变元新值产生的冲突数的评估价值提交给选择器（需要参考selection, problem的接口）——1行
                self._problem.set_variable(selected_index, old_variable)
                
                if selection.done():
                    break
            
            self._problem.set_variable(selected_index, choices[permutation[selection.selected_index()]]) 
    
    def search(self, iterations:int, max_steps:int, selection:SelectionBase,
        value_of:ConflictValueEstimatorType=default_conflict_value_estimator):
        
        for i in range(iterations):
            print("<begin>")
            self._problem.reset()
            self._sample_path(max_steps, selection, value_of)
            
            if not self._problem.has_conflict():
                print(f"Successful search: {i}")
                self._problem.show()
                break
            else:
                print(f"Failed search: {i}")
            print("<end>")