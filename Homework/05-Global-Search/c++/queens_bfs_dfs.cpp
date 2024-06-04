#include <iostream>
#include <ctime>

#include "problem/queens.hpp"

#include "algorithm/depth_first_search.hpp"
#include "algorithm/breadth_first_search.hpp"

int main(){
    std::ios::sync_with_stdio(false);

    // time_t t0 = time(nullptr);
    
    // QueensState state(12);
    // BreadthFirstSearch<QueensState> bfs(state);
    // bfs.search(true, false);

    // //DepthFirstSearch<QueensState> dfs(state);
    // //dfs.search(true, false);
    
    // std::cout << time(nullptr) - t0 << std::endl;
    // return 0;

    for (int i = 8; i <= 15; ++i) {
        std::cout << "Queens State for " << i << " queens:" << std::endl;

        time_t t0 = time(nullptr);
        QueensState state(i);

        BreadthFirstSearch<QueensState> bfs(state);
        bfs.search(true, false);
        std::cout << "\tBFS time = " << time(nullptr) - t0 << "s" << std::endl;

        t0 = time(nullptr);
        DepthFirstSearch<QueensState> dfs(state);
        dfs.search(true, false);
        std::cout << "\tDFS time = " << time(nullptr) - t0 << "s" << std::endl;

        std::cout << "----" << std::endl;
    }
}
