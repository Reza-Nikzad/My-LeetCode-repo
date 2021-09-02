class Node:
    def __init__(self, val: int):
        self.value = val
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def clear(self):
        if self.head is None:
            print('The linked list doesn\'t exist')
        else:
            self.head = None
            self.tail = None

    def insert(self, val: int, location=0):
        if self.size == 0:
            self.head = Node(val)
            self.tail = Node(val)
        elif location == 0:
            curr = self.head
            self.head = Node(val)
            self.head.next = curr
        elif location > self.size:
            print('index out of bonds')
            return
        else:
            curr = self.head
            for i in range(location - 1):
                curr = curr.next
            node = Node(val)
            node.next = curr.next
            curr.next = node

        self.size += 1

    def find(self, val: int) -> int:
        if self.head is None:
            return -1
        else:
            curr = self.head
            index = 0
            if curr.data == val:
                return 0
            while curr.next is not None:
                curr = curr.next
                index += 1
                if curr.data == val:
                    return index
        return -1

    def print(self) -> None:
        items = []
        if self.head is None:
            print("items: " + str(items))
            return
        curr = self.head
        items.append(curr.data)
        while curr.next is not None:
            curr = curr.next
            items.append(curr.data)
        print("items: " + str(items))
        return


obj = LinkedList()
obj.print()
obj.insert(1, 0)
obj.print()
obj.insert(2, 0)
obj.print()
obj.insert(3, 1)
obj.print()
obj.insert(4, 0)
obj.print()
obj.insert(2, 4)
obj.print()
print(obj.find(2))
print(obj.find(4))
print(obj.find(1))
obj.clear()
obj.print()
obj.clear()
