"""
Please provide an analysis (mean and std) for the volumes of hippocampus with respect to the level of AD.
28.09.2022
"""

import pandas as pd
import matplotlib.pyplot as plt

data_X = pd.read_csv(r"C:\Users\asus\Desktop\CEV\mri\MRI_and_CDRinfo_Values_X_train.csv")
data_Y = pd.read_csv(r"C:\Users\asus\Desktop\CEV\mri\CDR_Values_y_train.csv")

data_xy = [data_X["HIPPOVOL"], data_Y["CDRGLOB"]]
cols = ["H_VOL", "AD"]
data = pd.concat(data_xy, axis=1, keys=cols)

#print(data.shape)

print("\tMEAN\n",data.groupby('AD')[['H_VOL']].mean(),"\n------------------------\n")
print("\tSTD\n",data.groupby('AD')[['H_VOL']].std(),"\n------------------------\n")


pvt = pd.pivot_table(data, index=['AD'], values='H_VOL', aggfunc=['mean', 'std'])
print("\tPIVOT TABLE\n",pvt,"\n------------------------\n")

pvt.plot(kind="bar", grid = True)