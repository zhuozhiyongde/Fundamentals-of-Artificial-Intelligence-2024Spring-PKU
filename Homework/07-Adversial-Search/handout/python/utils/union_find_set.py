class UnionFindSet:
    
    def __init__(self, n:int):
        self.n = n
        self.color = list(range(n))
    
    def find(self, x:int) -> int:
        if x == self.color[x]:
            return x
        self.color[x] = self.find(self.color[x])
        return self.color[x]
    
    def join(self, x:int, y:int) -> None:
        cx, cy = self.find(x), self.find(y)
        if cx != cy:
            self.color[cx] = cy     