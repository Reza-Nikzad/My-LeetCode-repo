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


# creation
bstree70 = BSTree(70)

insert(bstree70, 60)
insert(bstree70, 50)
insert(bstree70, 80)
insert(bstree70, 90)
insert(bstree70, 20)
insert(bstree70, 95)

# travers all nodes
def preOrderTraversal(rootNode : BSTree):
    if not rootNode:
        return


# deletion of Node
# Search for a value

# deletion of tree