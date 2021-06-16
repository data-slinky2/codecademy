#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

arr = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
print(arr)
print(arr[2, 1:])
print(arr.reshape(2,6))

