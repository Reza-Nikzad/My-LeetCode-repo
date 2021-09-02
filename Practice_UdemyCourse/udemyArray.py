# there is no array in Python
# it is implemented by built in function
from array import *

arr1 = array('i', [1, 2, 3, 4, 5, 6])

# -------------insertion: shiftting(O(n))
"""
arr1.insert(2,9)
print(arr1)
"""
# ------------- traversing array(O(n))
"""
def traverse(array):
    for i in array:
        print(i)
traverse(arr1)
"""

# ------------- Access Element (O(1))
"""
def accessElement(array, index):
    if index > len(array)-1:
        print("out of array index!! ")
    else:
        print(array[index]) # NOTE: O(1)

accessElement(arr1,5)
"""
# ------------- Search by value in array
"""
def searchInArray(array, value):
    for i in range(len(array)): #--> O(1)
        if array[i] == value:
            return i
    return -1

print(searchInArray(arr1, 6))
"""
# ------------- Delete an element O(n)
"""
arr1.remove(4)
print(arr1)
"""

# 3. Append any value to the array using append() method
print('step 3: ')
arr1.append(7)
print(arr1)

# 4. Insert value in an array using insert() method
print('step 4: ')
arr1.insert(3, 22)
print(arr1)

# 5. Extend python array using extend() method
print('step 5: ')
arr2 = array('i', [12, 13, 14])
arr1.extend(arr2)
print(arr1)

# 6. Add items from list into array using fromlist() method
print('step 6: ')
list1 = [15, 16, 17]
print(arr2)
arr2.fromlist(list1)
print(arr2)
# 7. Remove any array element using remove() method
print('step 7: ')
print(arr1)
arr1.remove(4)
print(arr1)

# 8. Remove last array element using pop() method
print('step 8: ')
arr1.pop()
print(arr1)
# 9. Fetch any element through its index using index() method
print('step 9: ')
print(arr1.index(6))
print(arr1[6])
# 10. Reverse a python array using reverse() method
print('step 10: ')
print(arr2)
arr2.reverse()
print(arr2)

# 11. Get array buffer information through buffer_info() method
print('step11: ')
print(arr1.buffer_info())

# 12. Check for number of occurrences of an element using count() method
print('step12: ')
arr1.append(2)
print(arr1.count(2))
# 13. Convert array to string using tostring() method
print('step13: ')
print(arr1.tostring()[2])
# 14. Convert array to a python list with same elements using tolist() method
# 15. Append a string to char array using fromstring() method
# 16. Slice Elements from an array
