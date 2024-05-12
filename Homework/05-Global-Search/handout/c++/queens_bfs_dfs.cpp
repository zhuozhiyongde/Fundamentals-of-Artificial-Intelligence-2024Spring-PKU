#include <iostream>
#include <ctime>

#include "problem/queens.hpp"

#include "algorithm/depth_first_search.hpp"
#include "algorithm/breadth_first_search.hpp"

int main(){
    std::ios::sync_with_stdio(false);

    time_t t0 = time(nullptr);
    
    QueensState state(11);
    BreadthFirstSearch<QueensState> bfs(state);
    bfs.search(true, false);

    //DepthFirstSearch<QueensState> dfs(state);
    //dfs.search(true, false);
    
    std::cout << time(nullptr) - t0 << std::endl;
    return 0;
}
