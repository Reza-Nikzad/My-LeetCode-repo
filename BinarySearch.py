from typing import List


class BinarySearch:
    def search(self, nums: List[int], target: int) -> int:
        pL, pR = 0, len(nums)-1
        mid = -1
        while pL <= pR:
            mid = (pR+pL)//2
            midVal = nums[mid]
            if midVal == target:
                return mid
            elif midVal < target:
                pL = mid + 1
            else:
                pR = mid -1

        return -1
nums = [1,2,3,4,5,6,7,8,9]
print(BinarySearch().search(nums, 2))