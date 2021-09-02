class TreeNode:
    def __init__(self, val):
        self.value = val
        self.left = None
        self.right = None


newBT1 = TreeNode(1)
newBT2 = TreeNode(2)
newBT3 = TreeNode(3)
newBT1.left = newBT2
newBT1.right = newBT3
newBT4 = TreeNode(4)
newBT5 = TreeNode(5)
newBT2.left = newBT4
newBT2.right = newBT5
newBT6 = TreeNode(6)
newBT7 = TreeNode(7)
newBT8 = TreeNode(8)
newBT9 = TreeNode(9)
newBT3.left = newBT6
newBT3.right = newBT7
newBT4.left = newBT8
newBT4.right = newBT9
newBT10 = TreeNode(10)
newBT5.left = newBT10


def preorderTraversal(newRoot):
    if not newRoot:
        return
    print(newRoot.data)
    preorderTraversal(newRoot.left)
    preorderTraversal(newRoot.right)


# preorderTraversal(newBT1)

def inOrderTraversal(rootNode):
    if not rootNode:
        return
    inOrderTraversal(rootNode.left)
    print(rootNode.data)
    inOrderTraversal(rootNode.right)


inOrderTraversal(newBT1)
