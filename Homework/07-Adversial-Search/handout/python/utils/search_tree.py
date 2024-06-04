

class SearchTreeNode:
    
    def __init__(self, index:int):
        self.index = index
        self.parent = None
        self.n_children = 0
        self.children = list()
    
    def add_child(self, child:"SearchTreeNode") -> None:
        self.n_children += 1
        self.children.append(child)
        child.parent = self
    
    def child(self, index:int) -> "SearchTreeNode":
        return self.children[index]
    
    def __eq__(self, other:"SearchTreeNode") -> bool:
        return self.index == other.index
    
    def __hash__(self) -> int:
        return self.index
    
class SearchTree:
    
    def __init__(self):
        self.root = SearchTreeNode(0)
        self._unique_identifier = 1
        self.n_nodes = 1
        self.node_of = {0:self.root}
    
    def create_node(self) -> SearchTreeNode:
        new_node = SearchTreeNode(self._unique_identifier)
        self.node_of[self._unique_identifier] = new_node
        self._unique_identifier += 1
        return new_node

    def add_as_child(self, parent:SearchTreeNode, child:SearchTreeNode) -> None:
        parent.add_child(child)
        self.n_nodes += 1