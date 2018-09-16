import sys
import scipy
import numpy
import matplotlib
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal-length','sepal-width','petal-length','petal-width','class']

dataset = pandas.read_csv(url,names=names)

dataset.plot(kind='box', subplots=True, layout=(2,2),sharex=False, sharey=False)
print(dataset)

print(dataset.describe())

'''
Going to look at two types of plots for data visualization
    - Univariate Plots
        * Understand each attribute of the data
    - Multivariate Plots
        * Understand the relationship between attributes
'''

# Univariate Plots --> Box and Whisker 
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Histograms
dataset.hist()
plt.show()

# Multivariate Plots

# Scatter plot Matrix
scatter_matrix(dataset)
plt.show()

# Split-out validation dataset
array = dataset.values
x = array[:,0:4]
y = array[:,4]

# We will be using 80% of the data to train the model
# while 20% will be used for validating the dataset
validation_size = 0.20
seed = 7
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x,y,test_size=validation_size, random_state=seed)

'''
Training Data
    * x_train
    * y_train
Validation Data
    * x_validation
    * y_validation
'''
scoring = 'accuracy'

'''
Algorithms that we will be using to evaluate the data
    * Logistic Regression (LR)
    * Linear Discriminant Analysis (LDA)
    * K-Nearest Neighbors (KNN)
    * Classification and Regression Trees (CART)
    * Gaussian Naive Bayes (NB)
    * Support Vector Machines (SVM)
'''
models = []
models.append(('LR', LogisticRegression() ))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))

results = []
names = []

best_result = None

i = 0
for name, model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    if(i == 0):
        best_result = cv_results
    if(best_result.mean() < cv_results.mean()):
        best_result = cv_results
        best_algo = name
        print(best_algo)
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

print(best_result.mean())

svm = SVC()
svm.fit(x_train, y_train)
predictions = svm.predict(x_validation)
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))