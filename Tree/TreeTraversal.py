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

# insert a node to the end of a tree
def insert(root : TreeNode , value):
    newNode = TreeNode(value)
    if not root:
        root = newNode
        return
    else:
        customQueue= queue.Queue()
        customQueue.enqueue(root)
        while not customQueue.isEmpty():
            root = customQueue.dequeue().value
            if root.left:
                customQueue.enqueue(root.left)
            else:
                root.left = newNode
                print("{} value is added to left".format(value))
                return
            if root.right:
                customQueue.enqueue(root.right)
            else:
                root.right = newNode
                print("{} value is added to right".format(value))
                return


# delete a node:
# 1- get the last node,
# 2- delete the last node,
# 3. delete a node and substitute it with the last node.







# creating a complete binary tree from 1 to 10
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

# checking each traversal function
# print('--------------Pre-Ordered-----------------')
# preOrdered(newBT1)
# print('---------------In-Ordered----------------')
# inOrdered(newBT1)
# print('--------------Post-Ordered-----------------')
# postOrderd(newBT1)
# print('--------------Level-Ordered-----------------')
# levelOrder(newBT1)

insert(newBT1, 11)
print('-------------------------------')
levelOrder(newBT1)