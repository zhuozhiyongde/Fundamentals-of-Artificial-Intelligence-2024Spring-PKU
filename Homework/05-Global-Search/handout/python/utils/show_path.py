from typing import Dict
from queue import LifoQueue, Queue

from interface.state import StateBase

def show_reversed_path(last_state_of:Dict[StateBase, StateBase], state:StateBase) -> None:
    
    s = state
    path = LifoQueue()
    
    while s in last_state_of.keys():
        path.put(s)
        s = last_state_of[s]
    path.put(s)
    
    print("<begin>")
    while not path.empty():
        s = path.get()
        s.show()
    print("<end>")
