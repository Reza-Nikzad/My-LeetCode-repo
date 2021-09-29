
# 622. Design Circular Queue

class Node:
    def __init__(self, value):
        self.data = value
        self.next = None


class MyCircularQueue:

    def __init__(self, k: int):
        self.head = None
        self.tail = None
        self.maxSize = k
        self.size = 0

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        newNode = Node(value)
        if self.isEmpty():
            self.head = newNode
            self.tail = newNode
            self.tail.next = self.head

        else:
            self.tail.next = newNode
            self.tail = newNode
            self.tail.next = self.head
        self.size += 1
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        else:
            self.size -= 1
            if self.size <= 0:
                self.head = None
                self.tail = None
            else:
                self.head = self.head.next
                self.tail.next = self.head
            return True

    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.head.data

    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.tail.data

    def isEmpty(self) -> bool:
        if self.size == 0:
            return True
        return False

    def isFull(self) -> bool:
        if self.size == self.maxSize:
            return True
        return False
