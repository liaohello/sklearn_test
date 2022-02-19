# from sklearn import linear_model
# reg = linear_model.LinearRegression()
# reg.fit([[0,0],[1,1],[2,2]],[0,1,2])
# print(reg.coef_)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error,r2_score

#load the diabetes（糖尿病）dataset
diabetes_X,diabetes_y = datasets.load_diabetes(return_X_y = True)
# print (diabetes_X.shape)  #二维矩阵 （442，10）
# print (diabetes_y.shape)  #一维矩阵 （442）
# print(diabetes_y)         #
#use only one feature 前行后列 下面这个操作其实是一种numpy的切片
#: 指拿走所有行 2指定第三列  np.newaxis 位置随意指定新加一个轴
diabetes_X = diabetes_X[:,2,np.newaxis]
# print (diabetes_X.shape) #二维矩阵（442，1）

#_______________________________
#留出法来划分训练集和测试集
# split the data into training/testing sets
#只有一个参数时表示对最外层的axis进行操作

diabetes_X_train = diabetes_X[:-20] #二维矩阵（422，1）
diabetes_X_test = diabetes_X[-20:] #二维矩阵（20，1）
# print (diabetes_X_test.shape)

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20] #一维矩阵（422）
diabetes_y_test = diabetes_y[-20:]  #一维矩阵（20）

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train,diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test,diabetes_y_test,color = 'black')
plt.plot(diabetes_X_test,diabetes_y_pred,color = 'blue',linewidth = 3)

#不显示刻度
plt.xticks(())
plt.yticks(())

plt.show()