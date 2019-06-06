# -*- coding: utf-8 -*-
#sklearn.datasets中包含了很多知名的数据集，可以直接使用
from sklearn.datasets import load_iris
#iris中的data表示数据，target表示类别
iris=load_iris().data
iris_target=load_iris().target
#sklearn.preprocessing里面提供了大量的预处理工具，这里我们使用MinMaxScaler
#它的作用是按照一列中的最大值和最小值（极差）来进行数据的正则化
#基本使用方法是fit和transform
from sklearn.preprocessing import MinMaxScaler
MinMaxTransformer=MinMaxScaler()
MinMaxTransformer.fit(iris)
iris_transformed=MinMaxTransformer.transform(iris)
#sklearn.cross_validation中提供了一系列验证工具，这里我们使用的
# train_test_split可以用来分出测试集和训练集，其中X是数据，y是类别
# random_state是随机种子，不同的取值可以得到不同的结果，如果不进行设置，那么就完全随机
# 这里使用固定的数14，可以保证每次运行代码都得到相同的结果。
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(iris_transformed, iris_target, random_state=14)


# sklearn.tree里面包含了树模型，使用DecisionTree
# 基本用法是fit和predict
# accuracy的计算使用了numpy数组的一些技巧

import numpy as np

from sklearn.tree import DecisionTreeClassifier
Dtree=DecisionTreeClassifier()
Dtree.fit(X_train, y_train)
y_predict=Dtree.predict(X_test)
accuracy=np.mean(y_predict==y_test)*100
print("The accuracy of Dtree is {0}".format(accuracy))
