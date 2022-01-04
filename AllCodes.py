# 3. Longest Substring Without Repeating Characters:
# Given a string s, find the length of the longest substring
# without repeating characters.
# Input: s = "pwwkew" ; Output: 3
# Input: s = " " ; Output: 1
import heapq
import math


def printTree(root):
    if not root:
        return []
    queue = Queue()
    queue.put(root)
    print(root.val, end=',')
    while not queue.empty():
        temp =queue.get()
        if temp.left:
            queue.put(temp.left)
            print(temp.left.val, end=',')
        else: print('null', end=',')
        if temp.right:
            queue.put(temp.right)
            print(temp.right.val,end=',')
        else: print('null',end=',')

def lengthOfLongestSubstring(s: str) -> int:
    start = res = 0
    dct = {}
    for i in range(len(s)):
        if (s[i] in dct) and dct[s[i]] >= start:
            res = max(i-start, res)
            start = dct[s[i]] + 1
        dct[s[i]] = i

    return max(len(s)-start, res)

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

class SolutionMaxDepth:
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
# 572.Subtree of another root
# Definition for a binary tree node.
# from typing import Optional
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Mysolution:
    def isSubtree(self, s1:Optional[TreeNode],s2:Optional[TreeNode]) -> bool:
        string_s1 = self.traverse(s1)
        string_s2 = self.traverse(s2)
        if string_s2 in string_s1:
            return True
        return False

    def traverse(self, root: Optional[TreeNode]):
        if root:
            return f"#{root.val} {self.traverse(root.left)} {self.traverse(root.right)}"
        return None
# output
#3 #4 #1 None None #2 #0 None None None #5 None None
#4 #1 None None #2 #0 None None None
True

s1 = TreeNode(3)
s1.left =TreeNode(4)
s1.right = TreeNode(5)
s1.left.left = TreeNode(1)
s1.left.right= TreeNode(2)
s1.left.right.left = TreeNode(0)

ss1 = TreeNode(4)
ss1.left =TreeNode(1)
ss1.right = TreeNode(2)
ss1.right.left =TreeNode(0)

print(Mysolution().isSubtree(s1,ss1))

#===============================================================
#242. Valid Anagram
from collections import Counter

class ValidAnagram:
    def isAnagram(self, s: str, t: str) -> bool:
        x=Counter(s)
        y=Counter(t)
        if x==y:
            return True
        else:
            return False

''' 
Input: s = "anagram", t = "nagaram"
Output: true

Input: s = "rat", t = "car"
Output: false

Input: s = "a", t = "ab"
Output: false
'''
print(ValidAnagram().isAnagram('car', 'rat'))
#======================================================
# 49. Group Anagrams
class Anagram:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams = {}
        for word in strs:
            key = ''.join(sorted(word))
            if key not in anagrams:
                anagrams[key] = []
            anagrams[key].append(word)

        return list(anagrams.values())

print(Solution().groupAnagrams(["eat","tea","tan","ate","nat","bat"]))
# print(Solution().groupAnagrams(["","b",""]))
# print(Solution().groupAnagrams(["",""]))
# output = [["eat","tea","ate"],[]]
#=======================================================

# 977. Squares of a Sorted Array

class SquaresOfArray:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        pL = 0
        pR = len(nums)-1
        result = []
        while pL <= pR:
            if abs(nums[pL]) > abs(nums[pR]):
                result.append(nums[pL]**2)
                pL += 1
            else:
                result.append(nums[pR] ** 2)
                pR -= 1
        result.reverse()
        return result
# Input: nums = [-4,-1,0,3,10]
# Output: [0,1,9,16,100]
# Explanation: After squaring, the array becomes [16,1,0,9,100].
# After sorting, it becomes [0,1,9,16,100].
nums = [-4,-1,0,3,10]
print(SquaresOfArray().sortedSquares(nums))
#==================================================
# 189. Rotate Array
# Input : [1,2,3,4,5,6,7] , k =3
# output : [5,6,7,1,2,3,4]

class RotateArray:
    def rotate(self, nums: List[int], k: int) -> None:
        end = len(nums)
        if end <=1:
            return
        if k >= end:
            k %= end
        temp = nums[-k::1]
        nums[k:] = nums[0:-k:1]
        nums[0:k:1] = temp
#================================================
# 435. Non-overlapping Intervals
from typing import List
from operator import itemgetter

# itemgetter is a sort function with O(nlogn)
class Scheduling:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # sort based on finishing time
        if len(intervals)<=1:
            return 0
        intervals = sorted(intervals, key=itemgetter(1))
        fi = intervals[0][1]
        cntr = 0
        for i in range(1, len(intervals)):
            if intervals[i][0] < fi:
                cntr += 1
            else :
                fi = intervals[i][1]
        return cntr
nums = [[1,2],[2,3],[3,4],[1,3]]
Scheduling().eraseOverlapIntervals(nums)
#output = 1
#======================================================
# 283. Move Zeroes

class MoveZeroes:
    def moveZeroes(self, nums: List[int]) -> None:
        p0 = len(nums)
        for i in range(len(nums)):
            if nums[i] != 0:
                if p0 < i:
                    nums[i], nums[p0] = nums[p0], nums[i]
                    p0 += 1
            elif p0 > i:
                p0 = i
# Input: nums = [0,1,0,3,12]
# Output: [1,3,12,0,0]
#========================================================
#167. Two sum II - Input Array is Sorted
class Two_Sum_II:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        i = 0
        j = len(nums)-1
        while i<=j:
            if (nums[i] + nums[j]) == target:
                return [i+1, j+1]
            elif (nums[i] + nums[j]) > target:
                j -= 1
            else:
                i+=1
        return []
''' 
First Answer: 
        dic = {}
        for i in range(len(nums)): 
            if nums[i] not in dic:
                dic[target - nums[i]] = i+1
            else:
                return [dic[nums[i]],i+1]
        return []
'''
nums = [3,24,50,79,88,150,345]
target = 200
print(Two_Sum_II().twoSum(nums, target))
#========================================================
# 577. Reverse Words in a String III
class ReverseWords:
    def reversStr(self, s: str) -> str:
        x = s[len(s)-1::-1]
        t = x.split(' ')
        t.reverse()
        return ' '.join(t)

s="Let's take LeetCode contest"
print(ReverseWords().reversStr(s))
#========================================================
# 102. Binary Tree Level Order Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from queue import Queue
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = Queue()
        queue.put(root)
        res = []

        while not queue.empty():
            curr_lvl = []
            for _ in range(queue.qsize()):
                node = queue.get()
                curr_lvl.append(node.val)
                if node.left:
                    queue.put(node.left)
                if node.right:
                    queue.put(node.right)

            res.append(curr_lvl)

        return res
print(Solution().levelOrder(TreeNode(1,TreeNode(2, TreeNode(4)),TreeNode(3,TreeNode(5)))))
# Input:  [1,2,3,4,null,null,5]
# Output: [[1],[2,3],[4,5]]
#====================================================================
# 876. Middle of the Linked-list
from typing import Optional
from queue import Queue

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class MiddleNode:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # better answer
        # by add on step at the beginning and two step to the end
        bgn = head
        end = head
        while end and end.next:
            bgn = bgn.next
            end = end.next.next
        return bgn

        ''' My Answer 
        queue = Queue()
        i = 0 # i even -> put, i odd-> put and get
        while head is not None:
            if i%2 == 0: 
                queue.put(head)
            else: 
                queue.get()
                queue.put(head)
            head = head.next
            i += 1
        return queue.get()
        '''
# Input: head = [1,2,3,4,5,6] -> Output: [4,5,6]
# Input: head = [1,2,3,4,5] -> Output: [3,5,6]
l = ListNode(1,ListNode(2, ListNode(3, ListNode(4, ListNode(5,ListNode(6))))))
result = MiddleNode().middleNode(l)
print(result.val)
#==============================================================================
#206 Reverse Linked-list
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        curr = head
        prev = None
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev
#Input: [1,2,3,4,5] #output = [5,4,3,2,1]
l = ListNode(1,ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
result = Solution().reverseList(l)
while result :
    print(result.val, end=',')
    result = result.next
#===================================================================
# 19. Remove Nth Node from end of Linked list
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        fast = slow = head

        for _ in range(n):
            fast = fast.next

        if not fast:
            return head.next

        while fast.next:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next
        return head

l = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
result = Solution().removeNthFromEnd(l, 4)
while result:
    print(result.val, end=',')
    result = result.next
#=============================================================================
# 567. Permutation in String
from collections import Counter

class StringPermutation:
    def check(self, s1, p2):
        return Counter(s1) == Counter(p2)

    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False
        p = len(s1)
        val1 = val2 = 0
        for i in range(p):
            val1 += ord(s1[i])

        for i in range(p-1):
            val2 += ord(s2[i])

        k = p-1
        while k < len(s2):
            val2 += ord(s2[k])
            if val2 == val1:
                if self.check():
                    return True
            val2 -= ord(s2[k-p+1])
            k += 1

        return False

Input : s1 = "ab"
Output = s2 = "eidboaoo" #-> false
#Output = s2 = "eidbaoo" -> True
print(StringPermutation().checkInclusion(s1,s2))
#============================================================
# 3. median of two sorted arrays
# input = [1,2,6] , [2,6,9]
# Output = 4.0
class MidOfTwoSortedArray:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            return self.findMedianSortedArrays(nums2, nums1)

        l_total = len(nums1) + len(nums2)
        start = 0
        end = len(nums1)
        even = False
        if l_total %2 ==0:
            even = True

        while True:
            P1 = (start+ end)//2
            P2 = ((l_total +1 )//2) - P1

            left1 = -float('inf') if P1 == 0 else nums1[P1-1]
            left2 = -float('inf') if P2 == 0 else nums2[P2-1]
            right1 = float('inf') if P1 == len(nums1) else nums1[P1]
            right2 = float('inf') if P2 == len(nums2) else nums2[P2]

            if left1 <= right2 and right1 >= left2 :
                if even :
                    return (max(left1, left2) + min(right1, right2))/2
                else:
                    return max(left1, left2)
            elif left1 > right2 :
                end = P1 - 1
            elif right1 < left2:
                start = P1 + 1
nums1 = [1,2,6]
nums2 = [2,6,9]
print(MidOfTwoSortedArray().findMedianSortedArrays(nums1, nums2))
#==========================================================================
# 617.Merge two binary trees

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# Input: root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
# Output: [3,4,5,5,4,null,7]

class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:

        if not root1 and not root2:
            return None

        if not root1 or not root2:
            return root1 if root1 else root2

        root1.left = self.mergeTrees(root1.left, root2.left)
        root1.right = self.mergeTrees(root1.right, root2.right)
        root1.val += root2.val

        return root1

l1 = TreeNode(1, TreeNode(3, TreeNode(5),TreeNode(2)))
l2 = TreeNode(2, TreeNode(1,left=None ,right=TreeNode(4)), TreeNode(3,left=None, right= TreeNode(7)))
printTree(l1)
print('=================')
printTree(l2)
print('=================')
result = Solution().mergeTrees(l1, l2)
printTree(result)
#========================================================================
# 116. Populating next right pointers in each node
from typing import Optional

class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

class Populating:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return None
        stack = []
        stack.append(root)

        while len(stack) > 0:
            level = len(stack)
            temp =[]
            for i in range(level):
                if stack[i].left :
                    temp.append(stack[i].left)
                    temp.append(stack[i].right)
                if i == level-1:
                    stack[i].next = None
                else:
                    stack[i].next = stack[i+1]
            stack = temp
        return root
#=========================================================================
# 733. Flood Fill
class FloodFill:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        preV = image[sr][sc]
        checked = [[0 for i in range(len(image[0]))] for j in range(len(image))]
        m, n = len(image), len(image[0])

        def traverse(i, j):
            if 0<= i < m and 0<= j < n and image[i][j] == preV and checked[i][j] == 0:
                checked[i][j] = 1
                image[i][j] = newColor
                traverse(i+1,j)
                traverse(i-1,j)
                traverse(i,j+1)
                traverse(i,j-1)

        traverse(sr, sc)
        return image
image , sr , sc, newColor  = [[1,1,1],[1,1,0],[1,0,1]] , 1, 1, 2
# Output: [[2,2,2],[2,2,0],[2,0,1]]
print(FloodFill().floodFill(image,sr,sc,newColor))
#===========================================================================
# 695. Max Area of Island
input = [[1,1,1,0,0],
        [1,1,0,0,0],
        [0,0,0,0,0],
        [1,1,0,0,0],
        [1,0,1,1,1]]
#Output = 5
class MaxArea:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        self.count = 0
        result = 0
        def travers(a, b):
            if a < 0 or a >= m or b < 0 or b >= n or grid[a][b] != 1:
                return

            self.count += 1
            grid[a][b] = '#'
            travers(a+1,b)
            travers(a-1,b)
            travers(a,b+1)
            travers(a,b-1)

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    self.count = 0
                    travers(i,j)
                    result = max(self.count, result)

        return result
print(MaxArea().maxAreaOfIsland(input))
#==============================================================================
# 542. 01 Matrix

class Matrix01:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        # one from top to bottom and update by up and left
        # another from bottom to top and update by down and right
        m,n = len(mat), len(mat[0])
        for i in range(m):
            for j in range(n):
                if mat[i][j] > 0:
                    top = mat[i-1][j] if i != 0 else m+n+1
                    left = mat[i][j-1] if j != 0 else m+n+1
                    mat[i][j] = min(top+1, left+1)

        for i in range(m-1,-1):
            for j in range(n-1,-1,-1):
                if mat[i][j] > 0:
                    down = mat[i+1][j] if i != m else m+n+1
                    right = mat[i][j+1] if j != n else m+n+1
                    mat[i][j] = min(mat[i][j], down+1, right+1)
        return mat

inp = [[0,1,0,1,1],
       [1,1,0,0,1],
       [0,0,0,1,0],
       [1,0,1,1,1],
       [1,0,0,0,1]]
print(Solution().updateMatrix(inp))
# Output[[0, 1, 0, 1, 2],
#        [1, 1, 0, 0, 1],
#        [0, 0, 0, 1, 0],
#        [1, 0, 1, 1, 1],
#        [1, 0, 0, 0, 1]]
#================================================================================
# 994.Rotting Oranges

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        queue = Queue()
        self.fresh = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.put((i, j))
                elif grid[i][j] == 1:
                    self.fresh += 1
        time = 0
        if self.fresh == 0:
            return 0
        def getAdjacents(i, j):

            if i <m - 1 and grid[i + 1][j] == 1:
                grid[i + 1][j] = 2
                queue.put([i + 1, j])
                self.fresh -= 1

            if i > 0 and grid[i - 1][j] == 1:
                grid[i - 1][j] = 2
                queue.put([i - 1, j])
                self.fresh -= 1

            if j < n-1 and grid[i][j + 1] == 1:
                grid[i][j + 1] = 2
                queue.put([i, j + 1])
                self.fresh -= 1

            if j > 0 and grid[i][j - 1] == 1:
                grid[i][j - 1] = 2
                queue.put([i, j - 1])
                self.fresh -= 1

        while not queue.empty():
            p = queue.qsize()
            for _ in range(p):
                temp = queue.get()
                getAdjacents(temp[0], temp[1])
            time+=1

        if self.fresh > 0:
            return -1
        return time-1

grid = [[2,1,1],[1,1,0],[0,1,1]]
print( Solution().orangesRotting(grid)) #Output = 4
#=========================================================


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        # levels: each lvl: start from i -> n
        # when we reach last lvl (k) = return answer go upper lvl
        res = []

        # base condition
        if k == 0:
            return res

        if k == n:
            return [[*range(1, n + 1, 1)]]

        def backtrack(lvlstarts, comb):
            if len(comb) == k:
                res.append(comb[:])
                return

            for i in range(lvlstarts, n + 1):
                comb.append(i)
                backtrack(i + 1, comb)
                comb.pop()

        backtrack(1, [])
        return res
#==============================================================
# 77. Combinations
class Combinations:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        if k == 0:
            return res
        if k == n:
            result.appen([*range(1,n+1,1)])

        def backtrack(lvlstart, comb):
            if len(comb) == k :
                res.append(comb[:])
                return
            for i in range(lvlstart, n+1):
                comb.appen(i)
                backtrack(i+1,comb)
                comb.pop(i)

        backtrack(1,[])
        return res
# InputInput: n = 4, k = 2
# Output:[[2,4],[3,4],[2,3],[1,2],[1,3],[1,4]]
print(Combinations().combine(4, 4))
#=============================================================
# 46. Permutations

class Permutate:
    def backtrack(self,avalbl, perm, res):
        if len(avalbl) == 0:
            res.append(perm[:])
            return

        for i in range(len(avalbl)):
            perm.append(avalbl[i])
            self.backtrack(avalbl[:i]+avalbl[i+1:], perm,res)
            perm.pop()


    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        self.backtrack(nums, [], res)
        return res

# nums = [1,2,3] # nums = [1,2] # nums = [2]
nums = []
print(Permutate().permute(nums))
#=====================================================================
# 784.Letter Case Permutation
# recursive
class LetterPermutation:
    def letterCasePermutation(self, s: str) -> List[str]:
        res = []

        def backtrack(ans: str, index: int):
            if len(ans) == len(s):
                res.append(ans)
                return

            if s[index].isdigit():
                backtrack(ans+s[index], index+1)
            else:
                backtrack(ans+s[index].lower(), index+1)
                backtrack(ans+s[index].upper(), index+1)

        backtrack('',0)
        return res

# Iterative
class LetterCasePermutation:
    def letterCasePermutation(self, s: str) -> List[str]:
        stk = list(s)
        res = ['']
        while stk :
            top = stk.pop()
            if top.isalpha():
                res = [top.lower()+x for x in res]+[top.upper()+x for x in res]
            else:
                res = [top+x for x in res]
        return res

s = "a1b2"
print(LetterCasePermutation().letterCasePermutation(s))
# Output: ["a1b2","a1B2","A1b2","A1B2"]
print(LetterPermutation().letterCasePermutation(s))
# =======================================================================
# 70. Climbing Stairs

class ClimbingStairs:
    def climbing(self, n: int)-> int:
        t1 , res = 0, 1
        for _ in range(n):
            t1, res = res, res + t1
        return res
print(ClimbingStairs().climbing(5)) # output = 8

#=========================================================
# 198. House Robber

class HouseRobber:
    def rob(self, nums: List[int]) -> int:
        # iteratively update nums[i] and put the max value of
        # nums[i]+nums[i-2] and nums[i-1]

        if len(nums) < 2:
            return max(nums)

        nums[1] = max(nums[0], nums[1])

        for i in range(2, len(nums)):
            nums[i] = max(nums[i] + nums[i - 2], nums[i - 1])
        return nums[-1]

num= [4,1,2,7,5,3,1]
print(HouseRobber().rob(num))
#=================================================================
# 231. Power of Two
# n == 2^x
class PowerOfTwo:
    def isPowerOfTwo(self, n: int) -> bool:
        # First answer
        #         while n >= 2:
        #             if n%2 != 0:
        #                 return False
        #             else:
        #                 n = n/2
        #         return n == 1

        if n <= 0:
            return False
        if n == 1:
            return True
        return math.log2(n).is_integer()
#===============================================================
# 136.Single Number
# input = [4,3,2,2,3,4,5]
# output = 5
class SingleNumber:
    def singleNumber(self, nums: List[int]) -> int:
        # xor 5 ^ 5 = 0
        # xor 5 ^ 3 = 6
        # (5^3)^5 = 3
        xor = 0
        for i in nums:
            xor ^= i
        return xor
print(SingleNumber().singleNumber([4,3,2,2,3,4,5]))
#================================================================
# 5. Longest Palindrom Substring
class LongestPalindromSubString():
    def longestPalindrome(self, s: str) -> str:
        res = s[0]

        for i in range(len(s)-1):
            pr = i + 1

            while pr < len(s) and s[i] == s[pr]:
                pr += 1
            pl= i-1
            res = res if len(s) > pr-i else s[i:pr+1]
            while pl >= 0 and pr < len(s):
                if s[pl] == s[pr]:
                    res = s[pl:pr+1] if len(s) < (pr-pl+1) else res
                    pl -= 1
                    pr += 1
                else:
                    break
            return res
s = "babad"
# Output: "bab""
print(LongestPalindromSubString().longestPalindrome(s))
#=====================================================================
# 20. valid Parentheses

class ValidParentheses:
    def isValid(self, s: str) -> bool:
        o_c = {'(': ')', '[': ']', '{': '}'}

        stack = []
        for i in s:
            if i in o_c.keys():
                stack.append(i)
            else:
                if len(stack) == 0 or o_c[i] != stack.pop():
                    return False

        return len(stack) == 0

s = "()[]{}" # -> True
print(ValidParentheses().isValid(s))
#========================================================================
# 23.Merge k Sorted list

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class MergeKList:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        tail = head = ListNode()
        min_heap = [(node.val, index) for index, node in enumerate(lists) if node]
        heapq.heapify(min_heap)

        while min_heap:
            _, min_index = heapq.heappop(min_heap)
            tail.next = lists[min_index]
            tail = tail.next
            lists[min_index] = lists[min_index].next

            if lists[min_index]:
                heapq.heappush(min_heap, (lists[min_index].val, min_index))

list1 = ListNode(1, ListNode(4, ListNode(5)))
list2 = ListNode(1, ListNode(3, ListNode(4)))
list3 = ListNode(2, ListNode(6))
head = MergeKList().mergeKLists([list1, list2, list3])
while head:
    print(head.val, end='->')
    head = head.next
#==============================================================================
# 33. Search in Rotated Sorted Array

class RotatedSortedSearch():
    def search(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums)-1

        while r >= l :
            mid = (r+l)//2
            if nums[mid] == target:
                return mid

            if nums[mid] < nums[r]:
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
            else:
                if nums[l]<= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
        return -1
nums = [4,5,6,7,0,1,2] # [4,5,6,7,0,1,2] # [1]
target = 0 # 3 -> -1 # 0 -> -1
print(RotatedSortedSearch().search(nums,target))
#=================================================================
# 39. Combination Sum
class CombinationSum:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def backT(comb, start, sUm):
            if sUm == target:
                res.append(comb[:])
                return
            elif sUm > target:
                return
            for i in range(start, len(candidates)):
                comb.append(i)
                backT(comb, i, sum(comb))
                comb.pop()

        backT([], 0, 0)
        return res

candidates = [2,3,5]
target =     8 #-> Output = [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
print(CombinationSum().combinationSum(candidates, target))
#=====================================================================