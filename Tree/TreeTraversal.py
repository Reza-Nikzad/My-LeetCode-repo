from Tree import QueueLinkedList as queue


class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


# preOrdered = root -> left -> right
def preOrdered(treeNode: TreeNode):
    if not treeNode:
        return
    print(treeNode.data)
    preOrdered(treeNode.left)
    preOrdered(treeNode.right)


# inOrdered : left-> root -> right
def inOrdered(treeNode: TreeNode):
    if not treeNode:
        return
    inOrdered(treeNode.left)
    print(treeNode.data)
    inOrdered(treeNode.right)


# postOrder: left -> right -> root
def postOrderd(treeNode: TreeNode):
    if not treeNode:
        return
    postOrderd(treeNode.left)
    postOrderd(treeNode.right)
    print(treeNode.data)


# levelOrder root -> queue  print queue; lef,right add to queue
def levelOrder(treeNode: TreeNode):
    if not treeNode:
        return
    printOrderQueue = queue.Queue()
    printOrderQueue.enqueue(treeNode)

    while not printOrderQueue.isEmpty():
        root = (printOrderQueue.dequeue()).value
        print(root.data)
        if root.left:
            printOrderQueue.enqueue(root.left)
        if root.right:
            printOrderQueue.enqueue(root.right)


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
preOrdered(newBT1)
print('-------------------------------')
inOrdered(newBT1)
print('-------------------------------')
postOrderd(newBT1)
print('-------------------------------')
levelOrder(newBT1)
