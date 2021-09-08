from Tree import QueueLinkedList as queue

class AVLNode:
    def __init__(self, val):
        self.data = val
        self.right = None
        self.left = None
        self.height = 1

def levelOrderTraversal(rootNode : AVLNode):
    if not rootNode:
        return
    else:
        customQueue = queue.Queue()
        customQueue.enqueue(rootNode)
        while not (customQueue.isEmpty()):
            root = customQueue.dequeue().value
            print(root.data)
            if root.left is not None:
                customQueue.enqueue(root.left)
            if root.right is not None:
                customQueue.enqueue(root.right)

def getHeight(rootNode : AVLNode):
    if not rootNode:
        return 0
    return rootNode.height

def getBalance(rootNode : AVLNode):
    if not rootNode:
        return 0
    return getHeight(rootNode.left) - getHeight(rootNode.right)

def leftRotation(disbalancedNode : AVLNode):
    newRoot = disbalancedNode.right
    disbalancedNode.right = disbalancedNode.right.left
    newRoot.left = disbalancedNode
    disbalancedNode.height = 1 + max(getHeight(disbalancedNode.left), getHeight(disbalancedNode.right))
    newRoot.height = 1 + max (getHeight(newRoot.left), getHeight(newRoot.right))
    return newRoot


def rightRotation(disbalancedNode : AVLNode):
    newRoot = disbalancedNode.left
    disbalancedNode.left = disbalancedNode.left.right
    newRoot.right = disbalancedNode
    disbalancedNode.height = 1 + max(getHeight(disbalancedNode.left), getHeight(disbalancedNode.right))
    newRoot.height = 1+max(getHeight(newRoot.left), getHeight(newRoot.right))
    return  newRoot


def insertNode(rootNode: AVLNode, value):
    if not rootNode:
        return AVLNode(value)
    elif rootNode.data > value:
        rootNode.left = insertNode(rootNode.left, value)
    else:
        rootNode.right = insertNode(rootNode.right, value)

    rootNode.height = 1+ max(getHeight(rootNode.left), getHeight(rootNode.right)) # update height of the root

    #balance the subtree
    balance = getBalance(rootNode)
    if balance > 1 and value < rootNode.left.data: # L L -> right rotation
        return rightRotation(rootNode)
    if balance > 1 and value > rootNode.left.data: # L- R -> left rotation on left and right rotation on root
        rootNode.left = leftRotation(rootNode.left)
        return rightRotation(rootNode)
    if balance < -1 and value > rootNode.right.data:  # R- R -> left rotation
        return leftRotation(rootNode)
    if balance < -1 and value < rootNode.right.data: # R - L -> right rotation on right then left rotation on root
        rootNode.right = rightRotation(rootNode.right)
        return leftRotation(rootNode)
    return rootNode

def getMinNode(rootNode: AVLNode):
    if rootNode is None or rootNode.left is None:
        return rootNode
    return getMinNode(rootNode.left)


def deleteNode(rootNode : AVLNode, value):
    if not rootNode:
        return rootNode
    elif rootNode.data > value :
        rootNode.left = deleteNode(rootNode.left , value)
    elif rootNode.data < value:
        rootNode.right = deleteNode(rootNode.right, value)
    else:
        if rootNode.left is None:
            temp = rootNode.right
            rootNode = None
            return temp
        elif rootNode.right is None:
            temp = rootNode.left
            rootNode = None
            return temp
        temp = getMinNode(rootNode.right)
        rootNode.data = temp.data
        rootNode.right = deleteNode(rootNode.right, temp.data)
    rootNode.height = 1 + max(getHeight(rootNode.left), getHeight(rootNode.right))

    balance = getBalance(rootNode)
    if balance > 1 and getBalance(rootNode.left) >= 0: # LL -> right
        return rightRotation(rootNode)
    if balance < -1 and getBalance(rootNode.right) <=0: # RR -> left rotation
        return leftRotation(rootNode)
    if balance > 1 and getBalance(rootNode.left) < 0 : # LR -> left rotation on left node and right rotation on root
        rootNode.left = leftRotation(rootNode.left)
        return rightRotation(rootNode)
    if balance < -1 and getBalance(rootNode.right) > 0 : #RL -> right rotation of right and left rotation on root
        rootNode.right = rightRotation(rootNode.right)
        return leftRotation(rootNode)

    return rootNode


def deletAVLTree(rootNode):
    rootNode.data = None
    rootNode.left = None
    rootNode.right = None
    return "successfully deleted"

def search(rootNode, value):
    if rootNode.data == value:
        print( "Found")
    elif rootNode.data > value:
        if rootNode.left.data == value:
            print( "Found")
        else:
            search(rootNode.left, value)
    else:
        if rootNode.right.data == value:
            print("Found")
        else:
            search(rootNode.right, value)



root = AVLNode(9)
root = insertNode(root, 12)
root = insertNode(root, 15)
root = insertNode(root, 5)
root = insertNode(root, 6)
root = insertNode(root, 2)
root = deleteNode(root, 6)
root = deleteNode(root, 5)
root = deleteNode(root, 20)
levelOrderTraversal(root)
search(root, 15)