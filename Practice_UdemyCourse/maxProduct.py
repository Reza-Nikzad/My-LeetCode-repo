from typing import List

l = [1, 20, 30, 44, 5, 56, 57, 8, 9, 10, 31, 12, 13, 14, 35, 16, 27, 58, 19, 21]


def findMaxProd(nums: List[int]) -> int:
    nums.sort()
    return nums[-1] * nums[-2]


print(findMaxProd(l))
