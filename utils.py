import pandas as pd

def category_bin_splits(s):
    x = len(s)
    lst = []
    for i in range(1 << x):
        left = []
        right = []
        for j in range(x):
            if (i & (1 << j)):
                left.append(s[j])
            else:
                right.append(s[j])
        # lst.append([s[j] for j in range(x) if (i & (1 << j))])
        lst.append([left, right])
    return lst[:-1]

if __name__ == '__main__':
    print(powerset([2,3,4]))