class Graph:
    def __init__(self, gNodes = None):
        if gNodes is None:
            self.graphNodes = {}
        else:
            self.graphNodes = gNodes

    def addNode(self, fromNode: str, toNode: str):
        self.graphNodes[fromNode].append(toNode)


sampleGraph = {"A": ["B","C"],
               "B": ["A","D","E"],
               "C": ["A","E"],
               "D": ["B","E","F"],
               "E": ["F","D"],
               "F": ["E","D"]
               }
newGraph = Graph(sampleGraph)
newGraph.addNode("E","C")
newGraph.addNode("E","B")
print(newGraph.graphNodes)