"""
I've changed the dataset!!!
The dataset had gender values as it's index. I did reset the index values and added gender values as a new column.
"""

# =============================================================================
# OUTPUTS ---->https://github.com/erkamk/CEV---Data-Science-Bootcamp/blob/main/Tasks/TASK%20-%207_A_B_C_D_E_F/OUTPUTS_7_A-B-C-D-E-F.pdf
# =============================================================================

"""
27.10.2022
TASK-7A: Please apply one-hot encoding method on one of the categorical variables (not SEX/gender) 
for the given AD dataset. Please explain what kind of transformation occured on the dataset. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel(r'C:\Users\asus\Desktop\CEV\mri\mri.xlsx', index_col=0)
one_hot_data = pd.get_dummies(data, columns = ["RESIDENC"])
# the column named "RESIDENC" has gone. The unique values which it included has been added to dataframe column by column.

#%%
"""
TASK-7B: Please provide a couple of pivot tables in order to illustrate how SEX variable is influential
on our target array (CDRGLOB). Please provide a few multidimensional analyses (not limited to two variables).
"""
naccicv_data = pd.qcut(one_hot_data['NACCICV'], 2)
naccageb_data = pd.qcut(one_hot_data['NACCAGEB'], 2)
table1 = one_hot_data.pivot_table('CDRGLOB', ['SEX', naccicv_data], [naccageb_data, 'NACCFAM'])
print(table1)

naccbrnv = pd.qcut(one_hot_data['NACCBRNV'], 2)
grayvol = pd.qcut(one_hot_data['GRAYVOL'], 2)
table2 = one_hot_data.pivot_table('CDRGLOB', ['SEX', naccbrnv], [grayvol, 'HXHYPER'])
print(table2)

# The Gender which declared as "1", has higher correlation than "2" for both tables

#%%
"""
TASK-7C: Referring to the problem in part B, please provide a potential solution to the gender-based issue 
that appears as a bottleneck for the model development phase. One possible solution might be to provide 
normalization of the numeric columns with respect to the total brain volume (NACCBRNV column).
"""
data_c = one_hot_data.copy()
numeric_columns = []

# Detecting the columns which don't have the values (1,0) only
for each in data_c.columns:
    unq_val = data_c[each].unique()
    if not (([0,1] in unq_val or [1,0] in unq_val) and len(unq_val) == 2):
        numeric_columns.append(each)
numeric_columns.remove('NACCBRNV')

#dividing all the columns by specified column in the task and then normalizing between 0 and 1.
#all values of the column 'NACCBRNV' has been set to '1' due to the equation of x/x = 1

for each in numeric_columns:
    data_c[each]/= data_c['NACCBRNV']
data_c = (data_c- data_c.min()) / (data_c.max() - data_c.min())
data_c['NACCBRNV'] = 1 

#%%
"""
TASK-7D: After the normalization process, please provide a correlation matrix to report the critical features 
that have high correlation (positive or negative) with the target array (CDRGLOB).
"""
cor_cdr = data.corr()['CDRGLOB']
cor_cdr.sort_values(ascending=False)

#%%
"""
TASK-7E: Please develop 3 distinct formulas (in other words derived features like BMI score) 
that involve the critical features in Part-7D and then ensure that these derived features 
could be used as predictive variables for CDRGLOB (again correlation analysis will give you insight).
"""

data_c['FEATURE1'] = data_c['INDEPEND'] * data_c['LENTM'] / data_c['NACCMMSE']
data_c['FEATURE2'] = (data_c['LLATVENT'] * data_c['LHIPPO']) / (data_c['RLATVENT'] * data_c['RHIPPO'])
data_c['FEATURE3'] = (data_c['LLATVENT'] * data_c['RLATVENT']) / (data_c['LHIPPO'] * data_c['RHIPPO'])


#%%
"""
TASK-7F: Then please develop a machine learning model with the 3 derived features that you obtain in 7E 
via the use of train-test split (be careful about stratify parameter in this function) and gridsearchCV function 
as well as a classification algorithm (such as Gaussian Naive Bayes) but please do not forget to optimize its 
hyperparameters. Finally, please report your findings via the use of classification report function 
(also a sklearn function). 
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data_y = data['CDRGLOB']
data_c.drop(columns = ['CDRGLOB'], inplace = True)
data_c[data_c == np.inf]=np.nan
data_c.fillna(data_c.mean(), inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data_c, data_y, test_size=0.25, random_state=42)

param_grid_RFC = {'min_impurity_decrease': np.linspace(0.0,0.6,num = 5),
                  'min_samples_leaf':np.linspace(0.0,0.6,num = 5),
                  'n_estimators': np.arange(97,103)
                  }

grid_RFC = GridSearchCV(RandomForestClassifier(), param_grid_RFC, cv=3)
grid_RFC.fit(X_train, y_train)
print("RFC Best Score",grid_RFC.best_score_)

rfc_best=grid_RFC.best_estimator_
rfc_best.fit(X_train, y_train)
RFC_predicted = rfc_best.predict(X_test)
print("Testing RFC on Test dataset = ",accuracy_score(y_test, RFC_predicted))
print(classification_report(y_test, RFC_predicted))












