from algorithm.alpha_beta_search import AlphaBetaSearch
from algorithm.general_game_search import GeneralGameSearch
from problem.tictactoe import TicTacToeState

if __name__ == "__main__":

    t = TicTacToeState()
    fgs = GeneralGameSearch(t)

    fgs.search()
    
    abs = AlphaBetaSearch(t)

    abs.search()