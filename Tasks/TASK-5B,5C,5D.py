"""
13.10.2022

TASK-5B: Please give concrete examples for each of the 4 domains of machine learning including classification,
regression, clustering, and dimensionality reduction.
"""
# classification : Kedi ve köpeklerin boyut veya fotoğraflarına göre ayırt edilmesi sonucu sınıflandırma
# regression     : Yıl bazında herhangi bir bitki türünün t anında uzunluğunun cm türünden tahmin edilmesi
# clustering     : Balinaların okyanuslarda hayatlarının çoğunluğunu geçirdiği derinlik (x,y,z) seviyesine göre kümelendirme
# dimensionality reduction : Ev fiyatlarını etkileyen 4 özelliğin 2 özelliğe matematiksel formüllerle indirgenmesi

#%%
"""
TASK-5C: Please run the GaussianNB algorithm for 20 different values with respect to varr_smoothing parameter 
to find the best model. Then please visualize your findings via a lineplot (matplotlib) 
to illustrate the more appropriate values/ranges.
"""
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pylab as plt

var = np.linspace(1e-01,1e-21, num=20)
GNB_dict = {}

for each in var:
    dataset = datasets.load_iris()
    model = GaussianNB(var_smoothing = each)
    
    model.fit(dataset.data, dataset.target)
    expected = dataset.target
    predicted = model.predict(dataset.data)
    
    print(accuracy_score(expected, predicted))
    GNB_dict[each] = accuracy_score(expected, predicted)

GNB_list = GNB_dict.items()
GNB_parameters, GNB_scores = zip(*GNB_list)
plt.plot(GNB_parameters, GNB_scores)
plt.xlabel('var_smoothing')
plt.ylabel('Predicted Value')
plt.title('My Dictionary')
plt.show()

GNB_max_score = max(GNB_scores)

#%%%
"""
TASK-5D: Please use two different algorithms (other than GaussianNB) with their default parameter values 
and then compare your findings to the ones obtained in Task-5C.
"""
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pylab as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


#SVM
dataset = datasets.load_iris()
model = svm.SVC()

model.fit(dataset.data, dataset.target)
expected = dataset.target
predicted = model.predict(dataset.data)

svc_score = accuracy_score(expected, predicted)
print("SVM Score:",svc_score)

#RFS
dataset = datasets.load_iris()
model = RandomForestClassifier()

model.fit(dataset.data, dataset.target)
expected = dataset.target
predicted = model.predict(dataset.data)

rfs_score = accuracy_score(expected, predicted)
print("RFS Score:",rfs_score)

algorithms = ["GNB", "SVM", "RFS"]
algo_scores = [GNB_max_score, svc_score, rfs_score]

fig = plt.figure(figsize = (10, 5))
plt.bar(algorithms, algo_scores, color ='red')
 
plt.xlabel("Algorithm")
plt.ylabel("Score")
plt.show()
print("-------------------\n")

for each in range (3):
    print(algorithms[each], " : ", algo_scores[each],"\n")



