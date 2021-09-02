class NodeDouble:
    def __init__(self, val=0):
        self.value = val
        self.next = None
        self.previous = None


class DoubleLL:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def insert(self, val=0, index=0):
        newNode = NodeDouble(val)
        if self.head == None:
            self.head = newNode
            self.tail = newNode

        elif index == 0:
            newNode.next = self.head
            self.head.previous = newNode
            self.head = newNode

        elif index == self.size:
            newNode.previous = self.tail
            self.tail.next = newNode
            self.tail = newNode

        else:
            if index < 0 or index > self.size:
                return
            pointer = self.head
            for i in range(index - 1):
                pointer = pointer.next
            newNode.previous = pointer
            newNode.next = pointer.next
            pointer.next.previous = newNode
            pointer.next = newNode

        self.size += 1

    def delete(self, index):
        if index < 0 or index >= self.size - 1:
            return
        if index == 0:
            self.head = self.head.next
        elif index == self.size - 1:
            self.tail = self.tail.previous
            self.tail.next = None
        else:
            pointer = self.head
            for i in range(index - 1):
                pointer = pointer.next

            pointer.next.next.previous = pointer
            pointer.next = pointer.next.next

        self.size -= 1

    def printLinkedList(self):
        pointer = self.head
        while pointer != None:
            print(pointer.data)
            pointer = pointer.next


mydll = DoubleLL()

mydll.insert(0, 0)
mydll.insert(1, 1)
mydll.insert(2, 3)
mydll.insert(3, 2)
mydll.insert(4, 1)
mydll.printLinkedList()
mydll.delete(2)
print('---------')
mydll.printLinkedList()
