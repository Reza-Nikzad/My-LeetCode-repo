from Tree import QueueLinkedList as queue

class AVLNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

def preOrderTraverse(rootNode: AVLNode):
    if not rootNode:
        return
    print(rootNode.value)
    preOrderTraverse(rootNode.left)
    preOrderTraverse(rootNode.right)

def inOrderTraverse(rootNode: AVLNode):
    if not rootNode:
        return
    preOrderTraverse(rootNode.left)
    print(rootNode.value)
    preOrderTraverse(rootNode.right)


def postOrderTraverse(rootNode: AVLNode):
    if not rootNode:
        return
    preOrderTraverse(rootNode.left)
    preOrderTraverse(rootNode.right)
    print(rootNode.value)


def levelOrderTravesral(rootNode : AVLNode):
    if not rootNode:
        return
    customQueue = queue.Queue()
    customQueue.enqueue(rootNode)
    while not (customQueue.isEmpty()):
        root = customQueue.dequeue().value
        if root.left:
            customQueue.enqueue(root.left)
        if root.right:
            customQueue.enqueue(root.right)


def searchNode(rootNode, value):
    if rootNode.value == value:
        print('Found')
    elif rootNode.value < value :
        if rootNode.right.value == value:
            print('Found')
        else:
            searchNode(rootNode.right, value)
    else:
        if rootNode.left.value == value:
            print('Found')
        else:
            searchNode(rootNode.left, value)

def getHeight(rootNode: AVLNode):
    if not rootNode:
        return
    return rootNode.height

def leftRotation(disbalancedNode : AVLNode) -> AVLNode:
    newRoot = disbalancedNode.right
    disbalancedNode.right = disbalancedNode.right.left
    newRoot.left = disbalancedNode
    disbalancedNode.height = 1+ max(disbalancedNode.left.height, disbalancedNode.right.height)
    newRoot.height = 1+ max(newRoot.right.hight, newRoot.left.height)
    return newRoot

def rightRotation(disbalancedNode: AVLNode) -> AVLNode :
    newRoot = disbalancedNode.left
    disbalancedNode.left = disbalancedNode.left.right
    newRoot.right = disbalancedNode
    disbalancedNode.height = 1 + max(disbalancedNode.left.height, disbalancedNode.right.height)
    newRoot.height = 1 + max(newRoot.right.hight, newRoot.left.height)

    return newRoot

def getBalance(rootNode: AVLNode):
    if not rootNode:
        return 0
    return getHeight(rootNode.left) - getHeight(rootNode.right)

def insert(rootNode: AVLNode, value):
    if not rootNode:
        return AVLNode(value)
    if value < rootNode.value :
        rootNode.left = insert(rootNode.left, value)
    else:
        rootNode.right = insert(rootNode.right, value)

    rootNode.height = 1+ max (getHeight(rootNode.right), getHeight(rootNode.left))
    balance = getBalance(rootNode)

    if balance > 1 and