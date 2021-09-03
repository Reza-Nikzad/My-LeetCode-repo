class TreePL:
    def __init__(self, size):
        self.list = size * [None]
        self.lastUsedIndex = 0
        self.maxSize = size

    def insertNode(self, value):
        if self.lastUsedIndex +1 == self.maxSize:
            return 'the tree is full'
        self.list[self.lastUsedIndex + 1] = value
        self.lastUsedIndex +=1

    def searchByNode(self, value):
        for i in range(1,self.lastUsedIndex) :
            if self.list[i] == value:
                return i
        return -1

    #pre Order
    def preOrderTraversePL(self, index ): # index is added to be able to use recursion
        if index > self.lastUsedIndex:
            return
        print(self.list[index])
        self.preOrderTraversePL(2*index)
        self.preOrderTraversePL(2*index+1)


    #in Order
    def inOrderTraversal(self, index  ):
        if index > self.lastUsedIndex:
            return
        self.inOrderTraversal(2*index)
        print(self.list[index])
        self.inOrderTraversal(2*index+1)


    #Post Order
    def postOrderTraversal(self, index ):
        if index > self.lastUsedIndex:
            return
        self.postOrderTraversal(2 * index)
        self.postOrderTraversal(2 * index + 1)
        print(self.list[index])

    # level order
    def levelOrderTraversal(self):
        for i in range(1, self.lastUsedIndex+1):
            print(self.list[i])

    # delete a node
    def deletNode(self, value):
        if self.lastUsedIndex == 0:
            return "There is no Node to delete"
        for i in range(1, self.lastUsedIndex+1):
            if self.list[i] == value:
                self.list[i] = self.list[self.lastUsedIndex]
                self.list[self.lastUsedIndex] = None
                self.lastUsedIndex-=1
                return "deleted successfully"

    # delete treePL
    def delete(self):
        self.list = None
        return "the tree has been successfully deleted"

newTree = TreePL(11)
newTree.insertNode(1)
newTree.insertNode(2)
newTree.insertNode(3)
newTree.insertNode(4)
newTree.insertNode(5)
newTree.insertNode(6)
newTree.insertNode(7)
newTree.insertNode(8)
newTree.insertNode(9)
newTree.insertNode(10)
# newTree.preOrderTraversePL(1)
# print('-------------------------------------')
# newTree.inOrderTraversal(1)
# print('-------------------------------------')
# newTree.postOrderTraversal(1)
# print('-------------------------------------')
# newTree.levelOrderTraversal()
print(newTree.deletNode(5))
print('-------------------------------------')
newTree.levelOrderTraversal()
newTree.delete()

