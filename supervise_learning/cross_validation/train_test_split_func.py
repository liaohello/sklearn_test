import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X,y = datasets.load_iris(return_X_y = True)
# print(X.shape,y.shape) #(150, 4) (150,)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
# print(X_train.shape,y_train.shape)  #(90, 4) (90,)
# print(X_test.shape, y_test.shape)   # (60, 4), (60,)


