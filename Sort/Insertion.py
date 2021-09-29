from typing import List


def sort_insertion(nums : List[int]):
    for i in range(1,len(nums)):
        j = i - 1
        key = nums[i]
        while j >= 0 and nums[j] > key:
            nums[j+1] = nums[j]
            j -= 1
        nums[j+1] = key

nums = [5,5,2,4,6,1,3]
sort_insertion(nums)
print(nums)
