from Tree import QueueLinkedList as queue

class BSTree:
    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None


# insertion
def insert(rootNode : BSTree, value): # time and space complexity is O(Log N)
    if rootNode.data is None:
        rootNode.data = value
    elif rootNode.data >= value:
        if rootNode.left is None:
            rootNode.left = BSTree(value)
            return "Node has been added successfully"
        else:
            insert(rootNode.left, value)
    else:
        if rootNode.right is None:
            rootNode.right = BSTree(value)
            return "Node has been added successfully"
        else:
            insert(rootNode.right, value)


# travers all pre, in, post, order
def preOrderTraversal(rootNode : BSTree):
    if not rootNode:
        return
    print(rootNode.data)
    preOrderTraversal(rootNode.left)
    preOrderTraversal(rootNode.right)


def inOrderTraversal(rootNode :BSTree):
    if not rootNode:
        return
    preOrderTraversal(rootNode.left)
    print(rootNode.data)
    preOrderTraversal(rootNode.right)


def postOrderTraversal(rootNode : BSTree):
    if not rootNode:
        return
    preOrderTraversal(rootNode.left)
    preOrderTraversal(rootNode.right)
    print(rootNode.data)


def levelTraversal(rootNode :BSTree) :
    if not rootNode:
        return
    customqueue = queue.Queue()
    customqueue.enqueue(rootNode)
    while not customqueue.isEmpty():
        root = customqueue.dequeue().value
        print(root.data)
        if root.left is not None:
            customqueue.enqueue(root.left)
        if root.right is not None:
            customqueue.enqueue(root.right)


# Search for a value
def search(rootNode : BSTree, value):
    if not rootNode:
        return
    else:
        if rootNode.data == value:
            print("element with the value of "+str(value)+" is found")
            return
        elif rootNode.data >= value:
            search(rootNode.left,value)
        else:
            search(rootNode.right, value)

# deletion of Node
def minvalue(rootNode : BSTree):
    currNode = rootNode
    while (currNode.left is not None):
        currNode= currNode.left
    return currNode


def deleteNode(rootNode: BSTree, value):
    if rootNode is None:
        return rootNode

    if rootNode.data > value :
        rootNode.left = deleteNode(rootNode.left, value)

    elif rootNode.data < value:
        rootNode.right = deleteNode(rootNode.right, value)

    else:
        if rootNode.left is None:
            temp = rootNode.right
            rootNode= None
            return temp

        if rootNode.right is None:
            temp = rootNode.left
            rootNode = None
            return temp
        temp = minvalue(rootNode.right)
        rootNode.data = temp
        rootNode.right = deleteNode(rootNode.right, temp.data)
    return rootNode


# creation
bstree70 = BSTree(70)

insert(bstree70, 60)
insert(bstree70, 50)
insert(bstree70, 80)
insert(bstree70, 65)
insert(bstree70, 90)
insert(bstree70, 20)
insert(bstree70, 95)
insert(bstree70, 85)

'''
         70
      /      \
     60       80
    /  \     /  \
   50  65   85   90
  /               \
 20                95 
'''
# print('----------pre order-------------')
# preOrderTraversal(bstree70)
# print('----------in order-------------')
# inOrderTraversal(bstree70)
# print('-----------post order------------')
# postOrderTraversal(bstree70)
# print('-----------level order------------')
# levelTraversal(bstree70)
# print('-----------search-----------')
# search(bstree70, 20)
# search(bstree70, 95)
# search(bstree70, 10)
# search(bstree70, 100)
print('-----------delete-----------')
print('-----------delete 85 -----------')
deleteNode(bstree70, 85)
levelTraversal(bstree70)
print('-----------delete 50 -----------')
deleteNode(bstree70, 50)
levelTraversal(bstree70)

# deletion of tree