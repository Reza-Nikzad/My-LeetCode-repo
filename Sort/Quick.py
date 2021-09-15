from typing import List


class Quick:
    def partition(self, list, low, high):
        i = low -1
        pivot = list[high]
        for j  in range(low, high):
            if list[j] <= pivot:
                i+=1
                list[i], list[j] = list[j], list[i]
        list[i+1], list[high] = list[high], list[i+1]
        return i+1

    def quickSort(self, list, low , high):
        if low < high:
            index = self.partition(list, low, high)
            self.quickSort(list, low, index-1)
            self.quickSort(list, index+1, high)
