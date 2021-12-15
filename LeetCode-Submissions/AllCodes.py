# 3. Longest Substring Without Repeating Characters:
# Given a string s, find the length of the longest substring
# without repeating characters.
# Input: s = "pwwkew" ; Output: 3

def lengthOfLongestSubstring(s: str) -> int:
    start = maxLength = 0
    usedChar = {}

    for i in range(len(s)):
        if s[i] in usedChar and start <= usedChar[s[i]]:
            start = usedChar[s[i]] + 1
        else:
            maxLength = max(maxLength, i - start + 1)

        usedChar[s[i]] = i

    return maxLength


# -------------------------------------------------------
# 11. Container With Most Water
# Input: height = [1,8,6,2,5,4,8,3,7]
# Output: 49
from typing import List, Optional


def maxArea(height: List[int]) -> int:
    result = 0
    l = 0
    r = len(height) - 1
    hmax = max(height)

    while (r - l) * hmax > result:
        if (height[l] < height[r]):
            result = max(result, (r - l) * height[l])
            l += 1
        else:
            result = max(result, (r - l) * height[r])
            r -= 1

    return result


# -----------------------------------------------------
# 15. 3Sum
# finding all three numbers is a list which sum of them is 0
from collections import Counter
from typing import List


def threeSum(nums: List[int]) -> List[List[int]]:
    counter = Counter(nums)
    numbers = counter.keys()
    triplets = set()
    if counter[0] >= 3:
        triplets.add([0, 0, 0])
    positive, negative = [n for n in nums if n > 0], [n for n in nums if n < 0]
    for a in positive:
        for b in negative:
            c = -(a + b)
            if c in counter and ((c != a and c != b) or counter[c] > 1):
                triplets.add(tuple(sorted([a, b, c])))
    return triplets


sample = [-1, 0, 1, 2, -1, -4, -2, -3, 3, 0, 4]
# expected=[[-4,0,4],[-4,1,3],[-3,-1,4],[-3,0,3],[-3,1,2],[-2,-1,3],[-2,0,2],[-1,-1,2],[-1,0,1]]
print(threeSum(sample))
# -------------------------------------------------------
# 31. Next Permutation
from typing import List

nums = [1, 3, 2]


# expected [2, 1, 3]
def nextPermutation(nums: List[int]) -> None:
    def flip(nums, begin):
        l = begin
        r = len(nums) - 1

        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1

    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i >= 0:
        j = len(nums) - 1
        while nums[i] >= nums[j]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]

    flip(nums, i + 1)


nextPermutation(nums)
print(nums)


# -----------------------------------------------------
# 48. Rotate Image
def rotateImage(matrix: List[List[int]]) -> None:
    l, r = 0, len(matrix) - 1
    while l < r:
        for i in range(r - l):
            top, bottom = l, r
            temp = matrix[top][l + i]
            matrix[top][l + i] = matrix[bottom - i][l]
            matrix[bottom - i][l] = matrix[bottom][r - i]
            matrix[bottom][r - i] = matrix[top + i][r]
            matrix[top + i][r] = temp
        l += 1
        r -= 1


matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# expected Output: [[7,4,1],[8,5,2],[9,6,3]]
rotateImage(matrix)
print(matrix)


# -----------------------------------------------------
# 53. Maximum Subarray
# using Kadane's Algorithm
def maxSumSubArray(nums: List[int]) -> int:
    max_sum = - pow(10, 5)
    max_end_here = 0
    for i in range(len(nums)):
        max_end_here += nums[i]
        if max_end_here > max_sum:
            max_sum = max_end_here
        if max_end_here < 0:
            max_end_here = 0
    return max_sum


l = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# expected = 6
print(maxSumSubArray(l))


# -----------------------------------------------------
# 268. Missing Number
def missingNumber(nums: List[int]) -> int:
    return int((len(nums) * (len(nums) + 1) / 2) - sum(nums))


l = [1, 2, 4, 0]
print(missingNumber(l))
# 3
# -----------------------------------------------------
# 1. Two Sum
'''
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].'''


def twoSum(nums: List[int], target: int) -> List[int]:
    sNum = {}
    temp = 0  # temp = target - nums[i]

    for i in range(len(nums)):
        temp = target - nums[i]
        if temp not in sNum:
            sNum[nums[i]] = i
        else:
            return [sNum[temp], i]
    return []


nums = [2, 7, 11, 15]
target = 9
print(twoSum(nums, target))


# -----------------------------------------------------
# 55.Jump Games
def canJump(nums: List[int]) -> bool:
    maxJumpi = 0
    for i in range(len(nums)):
        if i > maxJumpi:
            return False
        maxJumpi = max(maxJumpi, nums[i] + i)
        if maxJumpi >= len(nums) - 1:
            return True


nums = [2, 3, 4, 2, 1, 0, 1]  # -> True
# nums = [2,3,3,2,1,0,1] # -> False
print(canJump(nums))


# --------------------------------------------------------
# 707.Linked List (singly)
class Node:
    def __init__(self, val):
        self.nextNode = None
        self.value = val


class MyLinkedList:

    def __init__(self):
        self.head = None
        self.length = 0

    def get(self, index: int) -> int:

        if index < 0 or index >= self.length:
            return -1

        if self.head is None:
            return -1

        curr = self.head
        for i in range(index):
            curr = curr.nextNode
        return curr.data

    def addAtHead(self, val: int) -> None:

        newNode = Node(val)
        newNode.nextNode = self.head
        self.head = newNode

        self.length += 1

    def addAtTail(self, val: int) -> None:
        if self.head == None:
            self.head = Node(val)
        else:
            curr = self.head
            while curr.nextNode is not None:
                curr = curr.nextNode

            curr.nextNode = Node(val)

        self.length += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.length:
            return

        if index == 0:
            self.addAtHead(val)
        else:
            curr = self.head

            for i in range(index - 1):
                curr = curr.nextNode

            newNode = Node(val)
            newNode.nextNode = curr.nextNode
            curr.nextNode = newNode

            self.length += 1

    def deleteAtIndex(self, index: int) -> None:  # done
        if index < 0 or index >= self.length:
            return

        curr = self.head
        if index == 0:
            self.head = curr.nextNode
        else:
            for i in range(index - 1):
                curr = curr.nextNode

            curr.nextNode = curr.nextNode.nextNode

        self.length -= 1


# Input: ["MyLinkedList","addAtHead","addAtTail","addAtIndex","get","deleteAtIndex","get"]
# [[],[1],[3],[1,2],[1],[1],[1]]
# OutPut: [null,null,null,null,2,null,3]

# --------------------------------------------------------------
# 2. add Two Numbers
'''You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.'''


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        val = 0
        carry = 0
        oldNode = ListNode(0)
        head = oldNode

        while l2 or l1:

            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

            val = (v1 + v2 + carry) % 10
            carry = (v1 + v2 + carry) // 10

            tempNode = ListNode(val)
            oldNode.next = tempNode
            oldNode = oldNode.next
        if carry > 0:
            tempNode = ListNode(carry)
            oldNode.next = tempNode
            oldNode = oldNode.next
        return head.next


'''Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.'''
# ----------------------------------------------------------------
# 21. Merge Two Sorted List
'''Merge two sorted linked lists and return it as a sorted list.
 The list should be made by splicing together the nodes of the first two lists.'''


def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    pointer = ListNode()
    head = pointer
    while l1 and l2:
        if l1.val < l2.val:
            pointer.next = l1
            pointer = pointer.next
            l1 = l1.next
        else:
            pointer.next = l2
            pointer = pointer.next
            l2 = l2.next
    if l1 or l2:
        pointer.next = l1 if l1 else l2
    return head.next


# input = [1,2,4]  and  [1,3,4]
# output = [1,1,2,3,4,4]
# --------------------------------------------------------------
# finding the missing number of numbers from 0-n
# n*n+1/2 = sum+(missing Number)
def missingNumber(nums: List[int]) -> int:
    return int((len(nums) * (len(nums) + 1) / 2) - sum(nums))


l = [1, 2, 4, 0]
print(missingNumber(l))
# 3

# --------------------------------------------------------------
# 141.Linked List Cycle

# -105 <= Node.val <= 105
# input = [3,2,0,-4]
# pos= 2 True , pos = -1 False
def linkedListCycle(head: Optional[ListNode]):
    while head != None:
        if head.val == 10e6:
            return True
        else:
            head.val = 10e6  # mark the visited node
            head = head.next
    return False

# --------------------------------------------------------------
# 88. Merge Sorted Array
def mergeArray(nums1: List[int], n: int, nums2: List[int], m) -> None:
    if n == 0:
        nums1[:] = nums2[:]
        return
    if m == 0:
        return
    if nums1[n - 1] <= nums2[0]:
        nums1[n:] = nums2[:]
        return
    else:
        pointer = m + n - 1
        i, j = n - 1, m - 1
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[pointer] = nums1[i]
                i -= 1
            else:
                nums1[pointer] = nums2[j]
                j -= 1

            pointer -= 1
        if j >= 0:
            nums1[0:pointer + 1] = nums2[0:j + 1]


# input
# [1,2,3,0,0,0], 3, [2,5,6], 3 ->out : [1,2,2,3,5,6]
# ------------------------------------------------------------
# 121. Best Time to Buy and Sell Stock
'''
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sel
'''


def bestBuySell(prices: List[int]):
    diff = 0
    low = prices[0]
    for i in prices[1:]:
        if i - low > diff:
            diff = i - low
        if low > i:
            low = i
    return diff
# ---------------------------------------------------------
# 704. Binary Search

def search(nums: List[int], target: int) -> int:
    pointerL = 0
    pointerR = len(nums)-1
    while pointerR - pointerL >= 1:
        n = int((pointerR - pointerL)/2)+pointerL

        if nums[n] == target:
            return n

        if nums[n] > target:
            pointerR = n-1

        if nums[n] < target:
            pointerL = n+1

    if nums[pointerR] == target:
        return pointerR
    return -1

nums = [-1,0,3,5,9,12]
target = 9
print(search(nums, target))
#------------------------------------------------------
# 104.Maximum depth of Binary Tree

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        return 1+max(self.maxDepth(root.left), self.maxDepth(root.right))
newBT1 = TreeNode(3)
newBT2 = TreeNode(9)
newBT3 = TreeNode(20)      #     3     -> 1
newBT1.left = newBT2    #      /  \
newBT1.right = newBT3   #     9   20   -> 2
newBT4 = TreeNode(15)   #        /  \
newBT5 = TreeNode(7)    #      15    7 -> 3
newBT3.left = newBT4    # depth = 3
newBT3.right = newBT5
depth = SolutionMaxDepth()
print(depth.maxDepth(newBT1))
#----------------------------------------------
# 226. Invert Binary Tree
def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    def travers(rootTree: TreeNode):
        if rootTree is None:
            return
        else:
            rootTree.left , rootTree.right = rootTree.right , rootTree.left
            travers(rootTree.left)
            travers(rootTree.right)

    travers(root)
    return root
#------------------------------------------------
#100 Same Tree
# check if two trees are completely the same

def isSameTree(p : Optional[TreeNode], q: Optional[TreeNode]):
    if p == None and q == None:
        return True
    if p!= None and q == None:
        return False
    if p == None and q != None:
        return False
    if p.val != q.val:
        return False
    return isSameTree(p.right, q.right) and isSameTree(p.left, q.left)
#------------------------------------------------
# 120. Triangle
triangle = [[-1],[2,3],[1,-1,-3]]
def minimumTotal( triangle: List[List[int]]) -> int:
    n = len(triangle)
    if not triangle:
        return
    for i in range(len(triangle)-2, -1, -1):
        for j in range(len(triangle[i])):
            triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1])
    return triangle[0][0]

print(minimumTotal(triangle))
#------------------------------------------------
#1143 Longest Common Subsequence
import bisect
import collections
import numpy as np
# my answer
def lcs(text1: str, text2: str) -> int:
    matrix = np.zeros([len(text1)+1, len(text2)+1], dtype = int)
    for i in range(len(text1)):
        for j in range(len(text2)):
            if text1[i] == text2[j]:
                matrix[i+1,j+1] = matrix[i,j]+1
            else:
                matrix[i+1,j+1] = max(matrix[i+1,j], matrix[i,j+1])

    return matrix[-1, -1]
# best answer
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = []
        d = collections.defaultdict(list)
        for i, c in enumerate(text2):
            d[c].append(i)
        for c in text1:
            if c in d:
                for i in reversed(d[c]):
                    ins = bisect.bisect_left(dp, i)
                    if ins == len(dp):
                        dp.append(i)
                    else:
                        dp[ins] = i
        return len(dp)
#========================================================================