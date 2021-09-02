from typing import List


def linearSearch(inputList: List[int], value: int) -> str:
    for i in inputList:
        if i == value:
            return value
    return 'not found'


lst = [1, 2, 3, 4, 6]
# print(linearSearch(lst,6))

arr = [[1, 2, 3, 4],
       [4, 5, 6, 7],
       [8, 9, 10, 11],
       [12, 13, 14, 15]]
for i in range(0, 4):
    print(arr[i].pop())

fruit = ['apple', 'banana', 'papaya', 'cherry']

fruit_list1 = ['Apple', 'Berry', 'Cherry', 'Papaya']
fruit_list2 = fruit_list1
fruit_list3 = fruit_list1[:]

fruit_list2[0] = 'Guava'
fruit_list3[1] = 'Kiwi'

sum = 0
for ls in (fruit_list1, fruit_list2, fruit_list3):
    if ls[0] == 'Guava':
        sum += 1
    if ls[1] == 'Kiwi':
        sum += 20

print(sum)
