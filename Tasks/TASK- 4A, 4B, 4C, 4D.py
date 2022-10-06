"""
TASK-4A: Please present a descriptive statistics analysis to provide a general understanding of the dataset. (describe)
06.10.2022
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_X = pd.read_csv(r"C:\Users\asus\Desktop\CEV\mri\MRI_and_CDRinfo_Values_X_train.csv")
data_Y = pd.read_csv(r"C:\Users\asus\Desktop\CEV\mri\CDR_Values_y_train.csv")

data_xy = [data_X, data_Y]

data = pd.concat(data_xy, axis=0 )
#print('CDRGLOB' in data.columns)

print(data.describe())

#%%
"""
TASK-4B: Please provide the visual illustrations of the dataset that will demonstrate the insightful distributions with respect to the target column (CDRGLOB: the level of the 
06.10.2022
"""

import seaborn as sns
sns.set(style = 'whitegrid')
#sns.violinplot(x ="SEX",y ="EDUC", hue = "CDRGLOB", data = data)

sns.scatterplot(data=data, x="SEX", y="NACCDAYS", hue="CDRGLOB")

#%%
"""
TASK-4C: Please provide a correlation analysis between two different variables/features and argue for the analysis outputs.
06.10.2022
"""
df = data[["NACCDAYS","VISITDAY"]]
df=(df-df.min())/(df.max()-df.min())
print(df.corr)

#%%
df = data[["EDUC","VISITDAY","VISITYR"]]
df=(df-df.min())/(df.max()-df.min())
print(df.corr)

