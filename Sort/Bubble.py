from typing import List

# Time Complexity : O(n^2) -> not time efficient
# Space Complexity : O(1)
# Stable Sort
# in Place Algorithm
def bubbleSort(nums : List[int]) -> List[int] :
    for i in range(len(nums)-1,0,-1):
        for j in range(i):
            if nums[j]> nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
    return nums


nums = [5,4,9,2,7,3,8,1,0,6]
print(bubbleSort(nums))
