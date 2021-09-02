from typing import List


def missingNumber(nums: List[int]) -> int:
    return int((len(nums) * (len(nums) + 1) / 2) - sum(nums))


l = [1, 2, 4, 0]
print(missingNumber(l))
# 3
