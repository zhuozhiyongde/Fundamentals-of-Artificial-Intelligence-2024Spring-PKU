from algorithm.monte_carlo_tree_search import MonteCarloTreeSearch
from problem.hex import HexState

if __name__ == '__main__':
    t = HexState(11)
    while not t.done():
        mcts = MonteCarloTreeSearch(t)
        action = mcts.select_action(150, 0.2)
        print(action)
        t = t.next(action)
        t.show()
