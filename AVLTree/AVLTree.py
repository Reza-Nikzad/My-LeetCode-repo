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


