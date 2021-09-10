class Heap:
    def __init__(self, size):
        self.heapList = size * [None]
        self.lastIndex = 0
        self.maxSize = size

# create
heap = Heap(5)

# peek
def peek(rootNode: Heap):
    if not rootNode:
        return
    else:
        return rootNode.heapList[0]

# size
def sizeOfHeap(rootNode: Heap):
    if not rootNode:
        return
    else: return rootNode.lastIndex

# Travers level order
def lvlOrderTravers(rootNode: Heap):
    if not rootNode:
        return
    else:
        for i in range(rootNode.lastIndex):
            print(rootNode.heapList[i])


# insert value


print(sizeOfHeap(heap))