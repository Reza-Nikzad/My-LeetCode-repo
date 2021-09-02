class Node:
    def __init__(self, val=0):
        self.value = val
        self.next = None

    def __str__(self):
        return str(self.value)


class MyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def __sizeof__(self):
        return self.size

    def __iter__(self):
        currNode = self.head
        while currNode:
            yield currNode
            currNode = currNode.next


class QueueList:

    def __init__(self):
        self.linkedList = MyLinkedList()

    def __str__(self):
        values = [str(x) for x in self.linkedList]
        return ','.join(values)

    # enqueue
    def enqueue(self, val):
        newNode = Node()
        newNode.value = val
        if self.linkedList.head is None:
            self.linkedList.head = newNode
            self.linkedList.tail = newNode
        else:
            currNode = self.linkedList.head
            self.linkedList.head = newNode
            self.linkedList.head.next = currNode

        self.linkedList.size += 1

    # dequeue
    def dequeue(self):
        if self.linkedList.head == None:
            return None
        else:
            yield self.linkedList.head.value
            self.linkedList.head = self.linkedList.head.next
            self.linkedList.size -= 1

    # peek
    def peek(self):
        if self.linkedList.__sizeof__() == 0 or self.linkedList.head == None:
            return 'list is empty'
        else:
            return self.linkedList.head

    # isEmpty
    def isEmpty(self):
        return self.linkedList.head == None

    # delete
    def delete(self):
        self.linkedList.head = None
        self.linkedList.tail = None


custQueue = QueueList()
custQueue.enqueue(1)
custQueue.enqueue(2)
custQueue.enqueue(3)
print(custQueue)
print(custQueue.peek())
print(custQueue)
