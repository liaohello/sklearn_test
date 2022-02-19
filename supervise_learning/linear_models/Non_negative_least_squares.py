import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# define a random seed that will remain a static number produced by random
#生成一个固定的随机值
np.random.seed(42)
n_samples, n_features = 200, 50
#随机产生200X50个真实输入数据
X = np.random.randn(n_samples, n_features)
# print(X.shape) #(200,50)
#随机产生50个真实参数
true_coef = 3 * np.random.randn(n_features)
# Threshold coefficients to render them non-negative
#保证参数>=0
true_coef[true_coef < 0] = 0
# w和x内积得y
y = np.dot(X, true_coef)
# print(y.shape)#(200,)
# 对每个样本加噪声
y += 5 * np.random.normal(size=(n_samples,))
#分割数据  留出法
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#++++++++++++++++++++++++++++++++++++++++
from sklearn.linear_model import LinearRegression
#参数为正
reg_nnls = LinearRegression(positive=True)
#得到预测y
y_pred_nnls = reg_nnls.fit(X_train, y_train).predict(X_test)
#计算R2值
r2_score_nnls = r2_score(y_test, y_pred_nnls)
print("NNLS R2 score", r2_score_nnls)

reg_ols = LinearRegression()
y_pred_ols = reg_ols.fit(X_train, y_train).predict(X_test)
r2_score_ols = r2_score(y_test, y_pred_ols)
print("OLS R2 score", r2_score_ols)

fig, ax = plt.subplots()
ax.plot(reg_ols.coef_, reg_nnls.coef_, linewidth=0, marker=".")

low_x, high_x = ax.get_xlim()
low_y, high_y = ax.get_ylim()
low = max(low_x, low_y)
high = min(high_x, high_y)
ax.plot([low, high], [low, high], ls="--", c=".3", alpha=0.5)
ax.set_xlabel("OLS regression coefficients", fontweight="bold")
ax.set_ylabel("NNLS regression coefficients", fontweight="bold")
# 上面展现的图比较了一般最小二乘和非负最小二乘之间的关系
#大于0时高度相关，小于0时无关
plt.show()