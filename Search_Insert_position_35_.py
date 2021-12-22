from typing import List
'''
Example 1:
Input: nums = [1,3,5,6], target = 5
Output: 2

Example 2:
Input: nums = [1,3,5,6], target = 2
Output: 1

Example 3:
Input: nums = [1,3,5,6], target = 7
Output: 4
'''

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:

        if target <= nums[0]:
            return 0
        elif target > nums[-1]:
            return len(nums)

        pL, pR = 0, len(nums)
        mid = 0

        while pR - pL > 1:
            mid = (pR + pL) // 2
            mid_val = nums[mid]
            if target == mid_val:
                return mid
            elif target > mid_val:
                pL = mid
            else:
                pR = mid
        return pR