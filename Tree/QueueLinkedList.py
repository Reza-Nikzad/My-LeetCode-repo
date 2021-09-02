class Node:
    def __init__(self, value):
        self.value = value  # treeNode
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None  # Node
        self.tail = None


class Queue:
    def __init__(self):
        self.linkedlist = LinkedList()

    # enqueue
    def enqueue(self, value):
        newNode = Node(value)

        if self.linkedlist.head is None:
            self.linkedlist.head = newNode
            self.linkedlist.tail = newNode

        else:
            self.linkedlist.tail.next = newNode
            self.linkedlist.tail = newNode

    # dequeue
    def dequeue(self):
        if self.linkedlist.tail is None:
            return None
        else:
            tempNode = self.linkedlist.head
            if self.linkedlist.head == self.linkedlist.tail:
                self.linkedlist.head = None
                self.linkedlist.tail = None
            else:
                self.linkedlist.head = self.linkedlist.head.next
            return tempNode

    # is Empty
    def isEmpty(self):
        if self.linkedlist.head == None:
            return True
        else:
            return False
