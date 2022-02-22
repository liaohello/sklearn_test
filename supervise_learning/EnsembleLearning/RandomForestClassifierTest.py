#其实%matplotlib inline这一句是IPython的魔法函数，可以在IPython编译器里直接使用
#，作用是内嵌画图，省略掉plt.show()这一步，直接显示图像。
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

wine = load_wine()

#实例化
#训练集带入实例化的模型去进行训练，使用的接口为fit
#使用其他接口将测试集导入我们训练好的模型，去获取我们希望过去的结果（score.Y_test）
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.3)

dtc = DecisionTreeClassifier(random_state = 0)
rfc = RandomForestClassifier(random_state = 0)

dtc = dtc.fit(Xtrain,Ytrain)
rfc = rfc.fit(Xtrain,Ytrain)

score_dtc = dtc.score(Xtest,Ytest)
score_rfc = rfc.score(Xtest,Ytest)

print("Single Tree:{}".format(score_dtc)
     ,"Random Forest:{}".format(score_rfc)
     )


#交叉验证：是数据集划分为n分，依次取每一份做测试集，每n-1份做训练集
#，多次训练模型以观测模型稳定性的方法
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rfc = RandomForestClassifier(n_estimators = 25)
rfc_cv = cross_val_score(rfc,wine.data,wine.target,cv=10)

dtc = DecisionTreeClassifier()
dtc_cv = cross_val_score(dtc,wine.data,wine.target,cv=10)

plt.plot(range(1,11),rfc_cv,label = "RandomForestClassifier")
plt.plot(range(1,11),dtc_cv,label = "DecisionTreeClassifier")
plt.legend()
plt.show()

#=========================另一种写法============================#
'''
label = "RandomForest"
for model in [RandomForestClassifier(n_estimators = 25),DecisionTreeClassifier()]:
    score = cross_val_score(model,wine.data,wine.target,cv=10)
    print("{}".format(label)),print(score.mean())
    plt.plot(range(1,11),score,label=label)
    plt.legend()
    label = "DecisionTree"
'''

dtc_1 = []
rfc_1 = []

for i in range(10):
    rfc = RandomForestClassifier(n_estimators = 25)
    rfc_cv = cross_val_score(rfc,wine.data,wine.target,cv = 10).mean()
    rfc_1.append(rfc_cv)
    dtc = DecisionTreeClassifier()
    dtc_cv = cross_val_score(dtc,wine.data,wine.target,cv = 10).mean()
    dtc_1.append(dtc_cv)

plt.plot(range(1,11),rfc_1,label="RandomForest")
plt.plot(range(1,11),dtc_1,label=" DecisionTree")
plt.legend()
plt.show()

#是否有注意到，单个决策树的波动轨迹和随机森林一致？
#再次验证了我们之前提到的，单个决策树的准确率越高，随机森林的准确率也会越高

#####【TIME WARNING: 2mins 30 seconds】#####
#  n_jobs  这个参数告诉引擎有多少处理器是它可以使用。 “-1”意味着没有限制，
#而“1”值意味着它只能使用一个处理器。 下面是一个用Python做的简单实验用来检查这个指标：
superpa = []

for i in range(200):
    rfc = RandomForestClassifier(n_estimators = i+1,n_jobs = -1)
    rfc_cv = cross_val_score(rfc,wine.data,wine.target,cv = 10).mean()
    superpa.append(rfc_cv)
#打印出：最高精确度取值，max(superpa))+1指的是森林数目的数量n_estimators
print(max(superpa),superpa.index(max(superpa)) + 1)
plt.figure(figsize=[20,5])
plt.plot(range(1,201),superpa)
plt.show()

#bootstrap参数默认True，代表采用这种有放回的随机抽样技术
#当n足够大时，这个概率收敛于1-(1/e)，约等于0.632。
#因此，会有约37%的训练数据被浪费掉，没有参与建模，
#这些数据被称为袋外数据(out of bag data，简写为oob)。

rfc = RandomForestClassifier(n_estimators = 25,oob_score = True)
rfc = rfc.fit(wine.data,wine.target)

import numpy as np
x=np.linspace(0,1,20)

y=[]
for epsilon in np.linspace(0,1,20):
    E = np.array([comb(25,i)*(epsilon**i)*((1-epsilon)**(25-i))
                  for i in range(13,26)]).sum()
    y.append(E)
plt.plot(x,y,"o-",label = "when estimators are different")
plt.plot(x,x,"--",color = "red",label = "if all estimators are same")
plt.xlabel("individual estimator's error")
plt.ylabel("RandomForest's error")
plt.legend()
plt.show()
