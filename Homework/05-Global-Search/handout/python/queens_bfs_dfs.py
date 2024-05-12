from time import time

from algorithm.breadth_first_search import BreadthFirstSearch
from algorithm.depth_first_search import DepthFirstSearch
from problem.queens import QueensState

if __name__ == '__main__':    
    
    t0 = time()
    
    s = QueensState(11)
    
    bfs = BreadthFirstSearch(s)
    
    bfs.search(True, False)
    
    #dfs = DepthFirstSearch(s)
    
    #dfs.search(True, False)
    
    print(f"time = {time() - t0}s")