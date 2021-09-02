class Node:
    def __init__(self, val=0):
        self.value = val
        self.next = None


class CSLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def insert(self, val=0, location=0):
        newNode = Node(val)
        if self.head == None:
            self.head = newNode
            self.tail = newNode
            self.tail.next = self.head
            self.size += 1
            return

        if location == 0:
            newNode.next = self.head
            self.head = newNode

        elif location == self.size:
            newNode.next = self.head
            self.tail.next = newNode
            self.tail = newNode
        else:
            pointer = self.head
            for i in range(location - 1):
                pointer = pointer.next
            newNode.next = pointer.next
            pointer.next = newNode
        self.size += 1
