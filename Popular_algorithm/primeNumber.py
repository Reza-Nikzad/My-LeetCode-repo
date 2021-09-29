def isprimeNumber(num: int):
    if num <= 2:
        return True
    else:
        i = 2
        end = num
        while i < end:
            if num % i == 0:
                return False
            end = (num // i) + 1
            i += 1
    return True
print(isprimeNumber(121))


