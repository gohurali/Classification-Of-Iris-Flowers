import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

print('Python: {}'.format(sys.version))
print('Scipy: {}'.format(scipy.__version__))
print('Numpy: {}'.format(numpy.__version__))
print('MatPlotLib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal-length','sepal-width','petal-length','petal-width','class']

dataset = pandas.read_csv(url,names=names)

print(dataset)