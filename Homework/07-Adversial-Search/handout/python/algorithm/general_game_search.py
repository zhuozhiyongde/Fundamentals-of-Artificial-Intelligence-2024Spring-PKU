from copy import deepcopy
from typing import List

from interface.game_state import GameStateBase
from utils.show_path import show_path

class GeneralGameSearch:
    
    def __init__(self, state:GameStateBase):
        self._initial_state = deepcopy(state)
    
    def _search(self, state:GameStateBase) -> List[float]:
        
        if state.done():
            return state.cumulative_rewards()
        
        best_cumulative_rewards = (-float("inf"),) * state.n_players()
        action_space = state.action_space()
        
        for action in action_space:
            next_state = state.next(action)
            cumulative_rewards = self._search(next_state)
            if # TODO：如果某个动作会导致当前玩家的分数提高，则切换到该动作
                best_cumulative_rewards = cumulative_rewards
                self._next_state_of[state] = next_state
        return best_cumulative_rewards        
        
    def search(self) -> None:
        self._next_state_of = dict()
        result = self._search(self._initial_state)
        print(f"Scores: {result}")
        show_path(self._next_state_of, self._initial_state)