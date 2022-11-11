"""
11.11.2022
TASK 9 A-B-C-D-E-F
"""
###################################################
"""
TASK-9A: Please apply the following feature engineering steps on the numeric features of the AD dataset.

Divide by NACCBRNV (MRI)
Multiply by NACCAGEB
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel(r"C:\Users\asus\Desktop\CEV\mri\Temporary_data3_Left_Right_Copy.xlsx", index_col=0)

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_data = data.select_dtypes(include=numerics)
data_naccageb = numeric_data.NACCAGEB
data_naccbrnv = numeric_data.NACCBRNV
#%%
numeric_columns = []
for each in data.columns:
    unq_val = data[each].unique()
    if not (([0,1] in unq_val or [1,0] in unq_val) and len(unq_val) == 2):
        numeric_columns.append(each)
        
numeric_columns.remove('NACCAGEB')
numeric_columns.remove('NACCBRNV')
y_data = data['CDRGLOB']
data.drop(columns = ["CDRGLOB"],inplace = True)
numeric_columns.remove('CDRGLOB')
#%%
numeric_data = data[numeric_columns]
for each in numeric_data.columns:
    numeric_data[each] *= (data_naccageb / data_naccbrnv)
#%%
"""
TASK-9B: Please apply Gaussian Naive Bayes algorithm on the numeric values (after the transformation) 
to predict the CDRGLOB values (class-based). Train-test split, train by grid search (k=5) and optimize 
for the parameter values.
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(numeric_data, y_data, test_size=0.25, random_state=42)

parameters = {"var_smoothing": np.linspace(1e-1,1e-15, num = 13)}
grid_nb = GridSearchCV(GaussianNB(), parameters, cv=5)
grid_nb.fit(X_train, y_train)
print("Naive Bayes Best Score: ",grid_nb.best_score_)
#output = 0.48571428571428577
#%%
"""
TASK-9C: Please apply a one-hot encoding method on the categorical data of the AD dataset.
"""
categorical_columns = list(set(data.columns) - set(numeric_columns))
categorical_columns.remove('NACCAGEB')
categorical_columns.remove('NACCBRNV')
one_hot_categoric = data[categorical_columns]
#%%
"""
TASK–9D: Please apply Multinomial Naive Bayes algorithms on the one-hot encoded 
categorical data of the AD dataset and predict the labels (CDRGLOB).
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(one_hot_categoric, y_data, test_size=0.25, random_state=42)
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
predicted = mnb.predict(X_test)
print("Multinomial Naive Bayes Score : ", accuracy_score(y_test, predicted))
#output = 0.6902654867256637
#%%
"""
TASK-9E: Please merge the normalized numeric values and one-hot encoded categorical values 
and then apply the following regression methods (Linear, Ridge and Lasso) to predict the labels 
as numeric values. Please do not forget to optimize them and use R2 for scoring.
"""
merge_data = pd.concat([numeric_data, one_hot_categoric], axis = 1)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
r2_lr = 1 - (1-reg.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(r2_lr)
#output = 0.3369298174511409
#%%
from sklearn.linear_model import Ridge
rdg = Ridge(alpha=1.0).fit(X_train, y_train)
r2_rdg =  1 - (1-rdg.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(r2_rdg)
#output = 0.33662454283053833
#%%
from sklearn import linear_model
lasso = linear_model.Lasso(alpha=0.1).fit(X_train, y_train)
r2_lasso =  1 - (1-lasso.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(r2_lasso)
#output = 0.02111983150303376
#%%
"""
TASK-9F: Please find and apply the classification versions of the following algorithms 
(Linear, Ridge and Lasso) with the hyperparameter optimization.
"""
lr_params = {"fit_intercept": [True, False],
             "normalize": [True, False],
             "n_jobs": np.arange(0,6)}
grid_lr = GridSearchCV(LinearRegression(), lr_params, cv=5)
grid_lr.fit(X_train, y_train)
print("LR Best Score",grid_lr.best_score_)
# output = 0.3423780155516045
#%%
ridge_params = {"fit_intercept": [True, False],
             "normalize": [True, False],
             "tol": np.linspace(1e-1,1e-11,num=10)}
grid_ridge = GridSearchCV(Ridge(), ridge_params, cv=5)
grid_ridge.fit(X_train, y_train)
print("LR Best Score",grid_ridge.best_score_)
#output = 0.34327914359271483
#%%
lasso_params = {"fit_intercept": [True, False],
             "normalize": [True, False],
             "tol": np.linspace(1e-1,1e-11,num=10)}
grid_lasso = GridSearchCV(linear_model.Lasso(), lasso_params, cv=5)
grid_lasso.fit(X_train, y_train)
print("Lasso Best Score",grid_lasso.best_score_)
a = grid_lasso.best_score_
#output = -0.002786879354359417

















