# from sklearn import linear_model
# reg = linear_model.Ridge(alpha=.5)
# reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
# print(reg.coef_)
# print(reg.intercept_)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is the 10x10 Hilbert matrix 利用广播机制产生Hilbert matrix
#相当于10个样本10个feature
X = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# print(np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)
# #############################################################################
# Compute paths
#np.logspace(start=开始值，stop=结束值，num=元素个数，base=指定对数的底, endpoint=是否包含结束值)
#默认10为低
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    #fit_interceptbool, default=True Whether to fit the intercept for this model.
    # If set to false, no intercept will be used in calculations (i.e. X and y are expected to be centered).
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    # print(ridge.coef_.shape) #(10,0)
# aaaa = np.array(coefs)
# print(aaaa.shape) #(200,10)
# #############################################################################
# Display results
#gca就是get current axes的意思
ax = plt.gca()
#x(200,1) y(200,10)画10根线  必须保持第一维参数一致
ax.plot(alphas, coefs)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the regularization")
plt.axis("tight")
plt.show()

####################################################
#三步走 第一拿数据 第二调用函数得到最佳参数 第三可视化或者计算精度
#数据、参数、后续处理
'''
b = a[i:j:s]:
i为起始索引(缺省为0)，
j为结束索引(不包括，缺省为len(a))，
s为步进(缺省为1).
所以a[i:j:1]相当于a[i:j].
当s<0时:
i缺省时，默认为-1，
j缺省时，默认为-len(a)-1，
所以a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍。
'''