class Graph :
    def __init__(self, gdict):
        if gdict is None:
            self.gNodes = {}
        else:
            self.gNodes = gdict

    def addNode(self, fromNode, toNode):
        self.gNodes[fromNode].append(toNode)

    def bfsTraversal(self, startNode):
        visited = [startNode]
        queue = [startNode]
        while queue:
            visit = queue.pop(0)
            print(visit)
            for i in self.gNodes[visit]:
                if i not in visited:
                    queue.append(i)
                    visited.append(i)

    def dfsTraversal(self, startNode):
        visited = [startNode]
        stack = [startNode]
        while stack:
            popedNode = stack.pop()
            print(popedNode)
            for i in self.gNodes[popedNode]:
                if i not in visited:
                    stack.append(i)
                    visited.append(i)


sampleGraph = {"A": ["B","C"],
               "B": ["A","D","G"],
               "C": ["A", "D", "E"],
               "D": ["C", "F", "B"],
               "E": ["F", "C"],
               "F": ["G","E","D"],
               "G": ["B","F"]
               }

newGraph = Graph(sampleGraph)
newGraph.bfsTraversal("A")
print("=================")
newGraph.dfsTraversal("A")