from math import log, exp
from time import time

from problem.queens_move import QueensMoveState
from problem.queens_swap import QueensSwapState
from algorithm.hill_climb import HillClimb
from utils.selection import MaxSelection, FirstBetterSelection, RouletteSelection


if __name__ == "__main__":
    n = 20
    log_n = log(n)
    queens_move_state_value_estimator = lambda state : exp(-log_n * state.conflicts)
    queens_swap_state_value_estimator = lambda state : exp(-log_n * state.state.conflicts)
    
    t0 = time()
    f_selection = FirstBetterSelection()
    r_selection = RouletteSelection()
    m_selection = MaxSelection()
    
    # state = QueensMoveState(n)
    # hcs = HillClimb(state)
    # hcs.search(queens_move_state_value_estimator, 1.0, n<<2, f_selection, 5)
    
    state = QueensSwapState(n)
    hcs = HillClimb(state)
    hcs.search(queens_swap_state_value_estimator, 1.0, n<<2, f_selection, 5)
    
    print(time() - t0)