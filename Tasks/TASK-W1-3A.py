"""
Write a piece of code that will generate numbers for a coupon in Sayisal Loto 
that should include 8 colons and each column should have 6 random numbers. 
Please control the colons in terms of having no repetitions.
24.09.2022

"""
import numpy as np
import random

x = np.zeros(48).reshape(8,6)

for each in range(np.shape(x)[0]):
    rand_values = random.sample(range(1, 49), 6)
    x[each] = rand_values

print(x)