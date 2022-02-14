import pandas as pd
import numpy as np

# df = pd.read_csv('data/adult.csv')
# print(df)
# print(df.columns)
# x = df['age'].to_numpy()
# print(np.histogram(x, bins="doane"))
# # print(np.isnan(x))
# for i in x:
#     print(i)
#     print(type(i))
#     if i=='nan':
#         print('a')
#     break
        
# # uniq_vals, counts = np.unique(x[~np.isnan(x)], return_counts=True)
# # print(uniq_vals)
# # print(counts)

# a = [[0,1] , [2,3] , [0,2]]
# print(a)
# a.remove([0, 1])
# a.sort(key= lambda x:x[0])
# print(a)

a = np.array([1,2,3,4,5])
b = [[0,3], [4,5]]
print(a[b[0]])
print(np.sum(a[b[0]:b[1]+1]))