
# =============================================================================
""" OUTPUTS"""
# https://github.com/erkamk/CEV---Data-Science-Bootcamp/tree/main/Tasks/TASK%20-%206_A_B_C_D_E/Outputs%206%20A%20B%20C%20D%20E
# =============================================================================


"""
15.10.2022
TASK-6A: Please use the latest version of the AD dataset to provide the following outputs:

y : CDRGLOB
X : all columns except for the label
Train-test split: .80-.20 (please use the stratify parameter)
Cross-validation both with k=5 and k=10

Please compare and discuss the outputs obtained from the cross validation step by k=5 and k=10.
"""

import pandas as pd

data = pd.read_excel(r'C:\Users\asus\Desktop\CEV\datasets\Temporary_data3_Left_Right_Copy.xlsx', index_col=0)
#%%
data_y = data['CDRGLOB']
del(data['CDRGLOB'])
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, data_y, test_size=0.2, random_state=42)

#%%
from sklearn.model_selection  import cross_val_score
from sklearn import svm
model = svm.SVC()
cross_val_scores_svm_5k = cross_val_score(model, X_train, y_train, cv=5)
cross_val_scores_svm_10k = cross_val_score(model, X_train, y_train, cv=10)
print("cross_val_scores_svm_5k : ",cross_val_scores_svm_5k, "\n\n\ncross_val_scores_svm_10k",cross_val_scores_svm_10k)


#%%
"""
TASK-6B: Please perform a grid search run with the use of a ML algorithm 
(and its 3 parameters at least with 5 different values for each) 
you select as opposed to the Gaussian Naive Bayes algorithm. 
Then compare the outputs with respect to the accuracy values on the test dataset. 
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


param_grid_GNB = {'var_smoothing': np.linspace(1e-01,1e-21, num=20)}

grid_GNB = GridSearchCV(GaussianNB(), param_grid_GNB, cv=7)
grid_GNB.fit(X_train, y_train)
print("GNB Best Score = ",grid_GNB.best_score_)

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
#%%
"""
TASK-6C: Please provide a visualization of the best algorithm 
with respect to the two of the dimensions in the dataset 
and please annotate the labels 
(separately for true labels and predicted labels in 2 different graphs). 
For instance, x-axis might be RPARCORT and y-axis might be LTEMPCOR.
"""
import matplotlib.pyplot as plt


plt.scatter(X_test['RPARCORT'], X_test['LTEMPCOR'], c= y_test)
plt.show()
#%%
plt.scatter(X_test['RPARCORT'], X_test['LTEMPCOR'], c= RFC_predicted)
plt.show()

#%%
"""
TASK-6D: Please apply one of the dimensionality reduction methods (PCA or isomap) 
and reduce the features matrix into 2 extracted dimensions. 
Then provide a visualization with respect to these dimensions. 
"""

from sklearn.decomposition import PCA  
import seaborn as sns

pca = PCA(n_components = 2)
pca.fit(data)
data_pca = pca.transform(data)
data_pca = pd.DataFrame(data_pca,columns=['PC1','PC2'])
sns.heatmap(data_pca.corr())
#%%
"""
TASK-6E: Please perform one of the clustering techniques (k-means or GMM) 
on the dataset (on the features matrix) then again provide a visual illustration 
with respect to the features like age, education. 
"""
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3).fit_predict(data)

plt.scatter(data['RPARCORT'], data['LTEMPCOR'], c= kmeans)



