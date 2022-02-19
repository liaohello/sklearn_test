from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 导入数据
wine = load_wine()
'''
pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,
          copy=True)
          
objs：Series，DataFrame或Panel对象的序列或映射。如果传递了dict，则排序的键将用作键参数，除非它被传递，在这种情况下，将选择值（见下文）。任何无对象将被静默删除，除非它们都是无，在这种情况下将引发一个ValueError。
axis：{0,1，...}，默认为0。沿着连接的轴。
join：{'inner'，'outer'}，默认为“outer”。如何处理其他轴上的索引。outer为联合和inner为交集。
ignore_index：boolean，default False。如果为True，请不要使用并置轴上的索引值。结果轴将被标记为0，...，n-1。如果要连接其中并置轴没有有意义的索引信息的对象，这将非常有用。注意，其他轴上的索引值在连接中仍然受到尊重。
join_axes：Index对象列表。用于其他n-1轴的特定索引，而不是执行内部/外部设置逻辑。
keys：序列，默认值无。使用传递的键作为最外层构建层次索引。如果为多索引，应该使用元组。
levels：序列列表，默认值无。用于构建MultiIndex的特定级别（唯一值）。否则，它们将从键推断。
names：list，default无。结果层次索引中的级别的名称。
verify_integrity：boolean，default False。检查新连接的轴是否包含重复项。这相对于实际的数据串联可能是非常昂贵的。
copy：boolean，default True。如果为False，请勿不必要地复制数据。          
'''
# #查看数据
# import pandas as pd
# #连接数据 axis = 0 横向 axis = 1 竖向
# wine_pd = pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis = 1)
# # print(wine_pd)
# # print(wine.feature_names)
# # print(wine.target_names)
# #30%测试 70%训练 XXYY
# Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.3)
# #模型实例化
# clf = tree.DecisionTreeClassifier(criterion="gini",random_state=40 )
# #拟合
# clf = clf.fit(Xtrain,Ytrain)
# #查看性能指标
# score = clf.score(Xtest,Ytest)
#
# # print(score)
#
# #画图
# import graphviz
# # feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素',
# #                 '颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
# feature_name = wine.feature_names
# class_names= wine.target_names
# '''
#  filled = True  # 由颜色标识不纯度
#  rounded = True  # 树节点为圆角矩形
# '''
# '''
# random_state 随机种子 定义随机状态
# splitter="random" 分枝中加入随机
# ,max_depth=3  最大深度
# ,min_samples_leaf=10  叶子最大样本
# ,min_samples_split=10   分支最大样本
# max_features  树训练最大使用特征数
# min_impurity_decrease 分枝能分的最小不纯度减少量
# '''
# dot_data = tree.export_graphviz(clf,
#                                 out_file = None,
#                                 feature_names = feature_name,
#                                 class_names = class_names,
#                                 filled = True,
#                                 rounded = True)
# graph = graphviz.Source(dot_data)
# # graph.view()
#
# # print(clf.feature_importances_)#返回一个list
# # print([*zip(feature_name,clf.feature_importances_)])

#------------------------------------------------------------------
#  利用超参数曲线 确认最优的剪枝参数
import matplotlib.pyplot as plt

test = []
Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.3)

for i in range(10):
    clf = tree.DecisionTreeClassifier(  max_depth= i+1
                                        ,criterion="gini"
                                        ,random_state= 30
                                        ,splitter="random"
                                        )
    clf = clf.fit(Xtrain,Ytrain)
    score = clf.score(Xtest,Ytest)
    test.append(score)
plt.plot(range(1,11),test,color = "red",label = "max_delpth")
plt.legend() #加上label
plt.show()


