class TreeNode:
    def __init__(self, data, children=[]):
        self.data = data
        self.children = children

    def __str__(self, level=0):
        ret = "  " * level + str(self.data) + "\n"
        for i in self.children:
            ret += i.__str__(level + 1)
        return ret

    def addChild(self, TreeNode):
        self.children.append(TreeNode)


owner = TreeNode('boss', [])

CEO1 = TreeNode('CEO1', [])
CEO2 = TreeNode('CEO2', [])
owner.addChild(CEO1)
owner.addChild(CEO2)

employer11 = TreeNode('employer11', [])
employer12 = TreeNode('employer12', [])
employer13 = TreeNode('employer13', [])
CEO1.addChild(employer11)
CEO1.addChild(employer12)
CEO1.addChild(employer13)

employer21 = TreeNode('employer21', [])
employer22 = TreeNode('employer22', [])
CEO2.addChild(employer21)
CEO2.addChild(employer22)

employee11 = TreeNode('employee11', [])
employer11.addChild(employee11)
print(owner)
