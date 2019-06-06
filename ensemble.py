# -*- coding: utf-8 -*-

# sklearn.datasets中包含了很多知名的数据集，可以直接使用
from sklearn.datasets import load_iris

# iris中的data表示数据，target表示类别
iris=load_iris().data
iris_target=load_iris().target

# sklearn.preprocessing提供了大量的预处理工具
# MinMaxScaler作用是按照一列中的最大值和最小值（极差）来进行数据的规范化
from sklearn.preprocessing import MinMaxScaler
MinMaxTransformer=MinMaxScaler()

# 基本方法：fit和transform
MinMaxTransformer.fit(iris)
iris_transformed=MinMaxTransformer.transform(iris)

# sklearn.cross_validation中提供了一系列验证工具
# train_test_split用来划分测试集和训练集，其中X是数据，y是类别
# random_state是随机种子，不同的取值可以得到不同的结果，如果不进行设置，则完全随机
# 使用固定数子可以保证每次运行代码都得到相同的结果
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(iris_transformed, iris_target, random_state=14)


# sklearn.tree包含了树模型，使用DecisionTree
# 基本方法：fit和predict
# accuracy的计算使用numpy数组的一些技巧
import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# Dtree=DecisionTreeClassifier()
# Dtree.fit(X_train, y_train)
# y_predict=Dtree.predict(X_test)
# accuracy=np.mean(y_predict==(y_test))*100
# print("The accuracy of Dtree is {0}".format(accuracy))

#%%
def ada_boost():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression()
    ada=AdaBoostClassifier(base_estimator=lr)
    ada.fit(X_train, y_train)
    predict_labels=ada.predict((X_test))
    accuracy=np.mean(predict_labels==(y_test))*100

#%%

#%%
def gbdt():
    from sklearn.ensemble import GradientBoostingClassifier
    gdbt=GradientBoostingClassifier()
    gdbt.fit(X_train, y_train)
    predict_y=gdbt.predict((X_test))
    accuracy=np.mean(predict_y==y_test)*100

#%%
def rf():
    from sklearn.ensemble import RandomForestClassifier
    rf=RandomForestClassifier()
    rf.fit(X_train, y_train)
    predict_y=rf.predict((X_test))
    accuracy=np.mean(predict_y==y_test)*100
#%%




