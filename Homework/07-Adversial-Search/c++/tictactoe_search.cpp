#include "algorithm/alpha_beta_search.hpp"
#include "algorithm/general_game_search.hpp"
#include "problem/tictactoe.hpp"

int main(){

    TicTacToeState t;
    
    GeneralGameSearch<decltype(t)> fgs(t);

    fgs.search();
    
    AlphaBetaSearch<decltype(t)> abs(t);

    abs.search();

    return 0;
}