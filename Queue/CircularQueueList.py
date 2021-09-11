class MyCircularQueue:
    def __init__(self, k: int):
        self.list = k * [0]
        self.max_size = k
        self.rear = -1
        self.front = -1
        self.size = 0
        #[---Front ------ Rear---]

    def enQueue(self, value: int):
        if self.isFull():
            return False
        if self.isEmpty():
            self.rear = self.front = 0
        else:
            if self.rear == self.max_size-1:
                self.rear = 0
            else:
                self.rear += 1

        self.list[self.rear] = value
        self.size+=1
        return True


    def deQueue(self):
        if self.isEmpty():
            return False
        else:
            if self.front == self.max_size - 1:
                self.front = 0
            else:
                self.front += 1
        self.size -= 1
        return True

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.max_size

    def Front(self):
        if self.isEmpty():
            return -1
        else:
            return self.list[self.front]

    def Rear(self):
        if self.isEmpty():
            return -1
        else:
            return self.list[self.rear]


#
# myQueue = MyCircularQueue(3)
# print(myQueue.enQueue(1))
# print(myQueue.enQueue(2))
# print(myQueue.enQueue(3))
# print(myQueue.Rear())
# print(myQueue.deQueue())
# print(myQueue.enQueue(4))
# print(myQueue.Rear())
myQueue = MyCircularQueue(2)
print(myQueue.enQueue(1))
print(myQueue.enQueue(2))
print(myQueue.deQueue())
print(myQueue.enQueue(3))
print(myQueue.deQueue())
print(myQueue.enQueue(3))
print(myQueue.deQueue())
print(myQueue.enQueue(3))
print(myQueue.deQueue())
print(myQueue.Front())


