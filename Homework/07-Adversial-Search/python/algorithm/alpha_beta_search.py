from copy import deepcopy

from interface.game_state import GameStateBase
from utils.show_path import show_path

class AlphaBetaSearch:
    
    def __init__(self, state:GameStateBase):
        self._initial_state = deepcopy(state)

    def _search(self, state:GameStateBase, alpha:float, beta:float) -> float:
        if state.done():
            return state.cumulative_rewards()[0]
        
        action_space = state.action_space()
        for action in action_space:
            next_state = state.next(action)
            new_value = self._search(next_state, alpha, beta)
            
            if state.active_player() == 0 and new_value > alpha:
                # TODO：更新alpha或者beta的值——1行
                alpha = new_value
                self._next_state_of[state] = next_state
            elif state.active_player() == 1 and new_value < beta:
                # TODO：更新alpha或者beta的值——1行
                beta = new_value
                self._next_state_of[state] = next_state

            # TODO：剪枝
            if alpha >= beta:
                break
        
        return alpha if state.active_player() == 0 else beta
    
    def search(self) -> None:
        self._next_state_of = dict()
        result = self._search(self._initial_state, -float("inf"), float("inf"))
        print(f"Score 0: {result}")
        show_path(self._next_state_of, self._initial_state)