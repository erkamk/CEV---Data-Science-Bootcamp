"""
13.11.2022
"""

"""
TASK-10A: Please model the AD dataset using the SVM, Decision Tree, and Random Forest algorithms.

Please apply train test split (%75-%25)
Then grid search (hyperparameter tuning)
Then classification report and confusion matrix (on the test dataset)
Then compare the findings from the three algorithms
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

data = pd.read_excel(r"C:\Users\asus\Desktop\CEV\mri\Temporary_data3_Left_Right_Copy.xlsx", index_col=0)
y_data = data["CDRGLOB"]
data.drop(columns = ["CDRGLOB"], axis = 1, inplace = True)
X_train, X_test, y_train, y_test = train_test_split(data, y_data, test_size=0.25, random_state=42)

#%%
"""
SVM
"""
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

svc_grid_parameters = {"C": np.linspace(2.0,3.5, num = 3),
                       "degree": np.arange(1,4),
                       "tol": np.linspace(1e-1,1e-5, num = 4),
                       "verbose": [True, False]}
svc_grid_model = GridSearchCV(SVC(), svc_grid_parameters)
svc_grid_model.fit(X_train, y_train)
svc_best_score = svc_grid_model.best_score_ # output = 0.5428571428571429
svc_best_params = svc_grid_model.best_params_ # output = {'C': 3.5, 'degree': 1, 'tol': 0.06667000000000001, 'verbose': True}
svc_best_estimator = svc_grid_model.best_estimator_
labels = list(sorted(y_data.unique()))
svc_predicted = svc_best_estimator.predict(X_test)
svc_test_ac = accuracy_score(y_test, svc_predicted) # output = 0.5870206489675516
print(classification_report(y_test, svc_predicted, labels=labels)) # output - PDF (1)
plot_confusion_matrix(svc_best_estimator, X_test, y_test, cmap = 'YlOrRd_r') # output - PDF (2)  
plt.show()

#%%
"""
Decision Tree
"""
from sklearn.tree import DecisionTreeClassifier

dt_grid_parameters = {"criterion": ["gini", "entropy", "log_loss"],
                      "min_samples_split": np.arange(2,5),
                      "min_samples_leaf": [1,2,3]}
dt_grid_model = GridSearchCV(DecisionTreeClassifier(), dt_grid_parameters)
dt_grid_model.fit(X_train, y_train)
dt_best_score = dt_grid_model.best_score_ # output = 0.6226600985221674
dt_best_params = dt_grid_model.best_params_ # output = {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 2}
dt_best_estimator = dt_grid_model.best_estimator_
dt_predicted = dt_best_estimator.predict(X_test)
dt_test_ac = accuracy_score(y_test, dt_predicted) # output = 0.0.6283185840707964
print(classification_report(y_test, dt_predicted, labels=labels)) # output - PDF (3)
plot_confusion_matrix(dt_best_estimator, X_test, y_test, cmap = 'YlOrRd_r') # output - PDF (4)  
plt.show()

#%%
"""
Random Forest Classifier
"""

from sklearn.ensemble import RandomForestClassifier

rfc_grid_parameters = {"n_estimators": [10,100],
                       "criterion": ["gini", "entropy", "log_loss"],
                       "min_samples_split": [2,3]
                       }

rfc_grid_model = GridSearchCV(RandomForestClassifier(), rfc_grid_parameters)
rfc_grid_model.fit(X_train, y_train)
rfc_best_score = rfc_grid_model.best_score_ # output = 0.6729064039408867
rfc_best_params = rfc_grid_model.best_params_ # output = {'criterion': 'entropy', 'min_samples_split': 3, 'n_estimators': 100}
rfc_best_estimator = rfc_grid_model.best_estimator_
rfc_predicted = rfc_best_estimator.predict(X_test)
rfc_test_ac = accuracy_score(y_test, rfc_predicted) # output = 0.6991150442477876
print(classification_report(y_test, rfc_predicted, labels=labels)) # output - PDF (5)
plot_confusion_matrix(rfc_best_estimator, X_test, y_test, cmap = 'YlOrRd_r') # output - PDF (6)  
plt.show()

#%%
classifier_name = ["Support Vector Machine", "Decision Tree", "Random Forest Classifier"]
classifier_test_score = [svc_test_ac, dt_test_ac, rfc_test_ac]
fig = plt.figure(figsize = (10, 8))
plt.bar(classifier_name, classifier_test_score, color ='turquoise', width = 0.7) # output - PDF (7)
plt.title("Test Accuracies of the 3 Classifiers")
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.grid()
plt.show()



