class Heap:
    def __init__(self, size, type : str):
        self.heapList = size * [None]
        self.lastIndex = 0
        self.maxSize = size
        self.type = type

    def heapify(self, index):
        rootIndex = int((index-1)/2)
        if self.type == "Max":
            if self.heapList[index] > self.heapList[rootIndex]:
                temp = self.heapList[index]
                self.heapList[index] = self.heapList[rootIndex]
                self.heapList[rootIndex] = temp
                self.heapify(rootIndex)
        if self.type == "Min":
            if self.heapList[index] < self.heapList[rootIndex]:
                temp = self.heapList[index]
                self.heapList[index] = self.heapList[rootIndex]
                self.heapList[rootIndex] = temp
                self.heapify( rootIndex)


    # insert value
    def insert(self, value):
        if self.maxSize <= self.lastIndex:
            return "heap is full"
        else:
            self.heapList[self.lastIndex] = value
            self.heapify(self.lastIndex)
            self.lastIndex += 1
            return " Added"

    def adjustHeap(self, i):
         

    def extract(self, value):
        for i in range(self.lastIndex):
            if value == self.heapList[i]:
                self.heapList[i] = self.heapList.pop(i)
                self.lastIndex-=1
                self.adjustHeap(i)




# create
heap = Heap(7, "Min")

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

heap.insert(2)
heap.insert(5)
heap.insert(3)
heap.insert(6)
heap.insert(1)
heap.insert(10)

lvlOrderTravers(heap)