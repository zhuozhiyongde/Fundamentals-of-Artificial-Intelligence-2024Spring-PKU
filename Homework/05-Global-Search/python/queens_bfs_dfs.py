from time import time

from algorithm.breadth_first_search import BreadthFirstSearch
from algorithm.depth_first_search import DepthFirstSearch
from problem.queens import QueensState

if __name__ == "__main__":
    # t0 = time()

    # s = QueensState(12)

    # bfs = BreadthFirstSearch(s)

    # bfs.search(True, False)

    # dfs = DepthFirstSearch(s)

    # dfs.search(True, False)
    for i in range(8, 14):
        t0 = time()
        print(f"Queens State for {i} queens:")
        s = QueensState(i)
        bfs = BreadthFirstSearch(s)
        bfs.search(True, False)
        print(f"\tBFS time = {time() - t0}s")
        t0 = time()
        dfs = DepthFirstSearch(s)
        dfs.search(True, False)
        print(f"\tDFS time = {time() - t0}s")
        print("----")

    # print(f"time = {time() - t0}s")
