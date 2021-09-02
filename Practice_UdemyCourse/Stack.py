class Stack:
    def __init__(self):
        self.list = []

    def __str__(self) -> str:
        values = [str(s) for s in self.list[::-1]]
        return '\n'.join(values)

    def isEmpty(self):
        if self.list == []:
            return True
        else:
            return False

    # push
    def push(self, val):
        self.list.append(val)
        return "The item is added!!"

    # pop
    def pop(self):
        if self.isEmpty():
            return "The stack is empty"
        else:
            return self.list.pop()

    # peek
    def peek(self):
        if self.isEmpty():
            return 'the list is empty'
        else:
            return self.list[-1]

    # is full
    # delete
    def delete(self):
        self.list = None


class StackSized:
    def __init__(self, size):
        self.list = []
        self.size = size

    def __str__(self):
        values = [str(x) for x in self.list[::-1]]
        return '\n'.join(values)

    # push
    def push(self, val):
        if self.isFull():
            return 'the stack is full'
        else:
            self.list.append(val)

    # pop
    def pop(self):
        if self.isEmpty():
            return 'the stack is empty'
        else:
            return self.list.pop()

    # isEmpty
    def isEmpty(self):
        if len(self.list) == 0:
            return True
        else:
            return False

    # peek
    def peek(self):
        if self.isEmpty():
            return ' the stack is empty'
        else:
            return self.list[-1]

    # isFull
    def isFull(self):
        if len(self.list) >= self.size:
            return True
        else:
            return False


class Node:
    def __init__(self, val):
        self.value = val
        self.next = None


class LinkedList:
    def __int__(self):
        self.head = None
        self.size = 0


class StackLinkedList:
    def __int__(self):
        self.list = LinkedList()
        self.size = self.list.size


stack = StackSized(3)
stack.push(0)
stack.push(1)
stack.push(2)
stack.push(3)
print('print: ')
print(stack)
print('pop')
print(stack.pop())
print('print: ')
print(stack)
