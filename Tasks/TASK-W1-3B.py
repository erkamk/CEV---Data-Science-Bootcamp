"""
Please write a piece of code to find the names and order no of the presidents 
with the highest and the lowest heights.
24.09.2022
"""

import pandas as pd

data = pd.read_csv(r"C:\Users\asus\Desktop\CEV\datasets\president_heights_new.csv")

highest_height = data.sort_values('height(cm)',ascending = False).head(1)
lowest_height = data.sort_values('height(cm)',ascending = False).tail(1)

o,n,cm = highest_height.iloc[0]
print("Highest Height : {}\nOrder no : {}\nName of the president : {}\n".format(cm, o, n))

o,n,cm = lowest_height.iloc[0]
print("Lowest Height : {}\nOrder no : {}\nName of the president : {}".format(cm, o, n))