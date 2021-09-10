class Graph :
    def __init__(self, gdict):
        if gdict is None:
            self.gNodes = {}
        else:
            self.gNodes = gdict

    def addNode(self, fromNode, toNode):
        self.gNodes[fromNode].append(toNode)

    def bfsTraversal(self, start):
        visited = [start]
        queue = [start]
        while queue:
            dequeued = queue.pop(0)
            print(dequeued)
            for adjacent in self.gNodes[dequeued]:
                if adjacent not in visited:
                    visited.append(adjacent)
                    queue.append(adjacent)

    def dfsTraversal(self, start):
        visited = [start]
        stack = [start]
        while stack:
            dequeued = stack.pop()
            print(dequeued)
            for adjacent in self.gNodes[dequeued]:
                if adjacent not in visited:
                    visited.append(adjacent)
                    stack.append(adjacent)


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