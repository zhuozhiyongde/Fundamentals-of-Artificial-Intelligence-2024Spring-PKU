#include "algorithm/monte_carlo_tree_search.hpp"
#include "problem/hex.hpp"

int main(){

    HexState<11> t;

    while (not t.done()){

        MonteCarloTreeSearch<decltype(t)> mcts(t);

        auto action = mcts.select_action(5000, 0.2);
        std::cout << action << std::endl;
        t = t.next(action);

        t.show();
    }

    return 0;
}