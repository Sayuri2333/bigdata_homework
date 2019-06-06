#-*-coding:utf-8-*-
from numpy import *
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    '''
    输入：sigmoid函数的输入值
    输出：sigmoid函数输出结果
    描述：sigmoid函数实现
    '''
    return 1.0/(1.0+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    '''
    输入：数据集，对应数据的类标签
    输出：回归系数
    描述：批量梯度上升法实现
    '''
    dataMatrix = mat(dataMatIn)             #转为NumPy矩阵
    labelMat = mat(classLabels).transpose() #转为NumPy矩阵
    m,n = shape(dataMatrix)      #得到数据集 行数（样本数）和 列数（特征数)
    alpha = 0.001                #梯度上升法迭代式中的 迭代步长                    
    maxIter = 500                #最大迭代次数
    weights = ones((n,1))        #将回归系数各分量初始化为1
    weights_array = zeros((maxIter,3))
    for k in range(maxIter):                
        h = sigmoid(dataMatrix*weights)     #得到sigmoid的输出值
        error = (labelMat - h)              #使用向量减法计算误差
        weights = weights + alpha * dataMatrix.transpose()* error #梯度上升法迭代式实现
        weights_array[k,:] = weights.transpose()
    return weights,weights_array

def stocGradAscent0(dataMatrix, classLabels):
    '''
    输入：数据集，对应数据的类标签
    输出：回归系数
    描述：随机梯度上升法实现
    '''
    m,n = shape(dataMatrix)  #得到数据集 行数（样本数）和 列数（特征数)
    alpha = 0.01             #梯度上升法迭代式中的 迭代步长   
    weights = ones(n)        #将回归系数各分量初始化为1
    weights_array = zeros((m,3))
    for i in range(m):       #对于所有数据样本
        h = sigmoid(sum(dataMatrix[i]*weights))            #得到sigmoid的输出值
        error = classLabels[i] - h                         #使用向量减法计算误差
        weights = weights + alpha * error * dataMatrix[i]  #梯度上升法迭代式实现
        weights_array[i,:] = weights.transpose()
    return weights, weights_array

def diff(X, y, p): # 计算一阶导数
    return -(y - p) @ X # (h(x) - y) * x为一阶导数

def hess_mat(X, p, X_XT): # 返回海森矩阵
    hess = np.zeros((X.shape[1], X.shape[1])) # 海森矩阵的维度为(n+1) * (n+1), n为样本特征维度
    for i in range(X.shape[0]): # 对于每一个样本迭代一次
        hess += X_XT[i] * p[i] * (1 - p[i]) # self.X_XT[i]为样本i特征取值相乘生成的(n+1) * (n+1)矩阵 p[i]为标量
    return hess

def newton_method(X, y, max_epoch): # 牛顿法本体
    weight = np.ones(X.shape[1]) # 初始化权重
    X_XT = []
    weights_array = zeros((max_epoch, X.shape[1]))
    for i in range(X.shape[0]): # 对于每一行
        t = X[i, :].reshape((-1, 1)) # 把每一行变成列向量
        X_XT.append(t @ t.T) # 相乘得到矩阵
        
    for i in range(max_epoch): # 在最大迭代次数前
        inX = dot(weight, X.T)
        p = sigmoid(dot(weight, X.T)) # 函数预测值
        dif = diff(X, y, p.T) # 一阶导数
        hess = hess_mat(X, p, X_XT) # 海森矩阵
        new_weight = weight - (np.linalg.pinv(hess) @ dif.reshape((-1, 1))).flatten() # 权重更新
        weight = new_weight
        weights_array[i,:] = weight
    return weight, weights_array

def stocGradAscent1(dataMatrix, classLabels, maxIter=150):
    '''
    输入：数据集，对应数据的类标签
    输出：回归系数
    描述：stocGradAscent0函数的改进版本，降低了结果的
    周期性波动，提高了结果的收敛速度
    注：相对stocGradAscent0改进的部分用
    '''
    m,n = shape(dataMatrix)  #得到数据集 行数（样本数）和 列数（特征数)
    weights = ones(n)        #将回归系数各分量初始化为1
    weights_array = zeros((m*maxIter, 3))
    k = 0
    for j in range(maxIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001 #【改进1】：alpha在每次迭代都会调整，会一定程度上缓解
                                       #结果的周期性波动。同时，由于常数项的存在，虽然alpha会
                                       #随着迭代次数不断减少，但永远都不会减少到 0。这保证了多
                                       #次迭代后新数据仍然会有影响
            randIndex = int(random.uniform(0,len(dataIndex)))#【改进2】：随机选取样本更新回归系数
                                                             #也可缓解结果的周期性波动
            h = sigmoid(sum(dataMatrix[randIndex]*weights))            #得到sigmoid的输出值
            error = classLabels[randIndex] - h                         #使用向量减法计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  #梯度上升法迭代式实现
            weights_array[k,:] = weights.transpose()
            k+=1
            del(dataIndex[randIndex])  #将随机选择的样本从数据集中删除，避免影响下一次迭代
    return weights,weights_array
	
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def classifyVector(inX, weights):
    '''
    输入：样本数据，回归系数
    输出：分类结果（0 或 1）
    描述：使用优化后的回归系数对数据进行分类
    '''
    prob = sigmoid(sum(inX*weights)) #计算sigmoid函数输出结果
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')#打开训练集文件
    frTest = open('horseColicTest.txt')     #打开测试集文件
    #将训练集数据和对应标签存放到trainingSet和trainingLabels中
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    #训练算法，得到回归系数trainWeights
    trainWeights, weights_array = newton_method(array(trainingSet), trainingLabels, 3)
    #每行读取测试集数据，使用回归系数得到分类结果
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        #若分类结果不一致，则errorCount加一
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    #打印此次测试的错误率
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest(numTests=10):
    errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations,the average error rate is: %f" % (numTests, errorSum/float(numTests)))

def plotWeights(weights_array1,weights_array2, weights_array3):
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=3,sharex=False, sharey=False, figsize=(20,10))
    x1 = arange(0, len(weights_array1), 1)
    #绘制w0与迭代次数的关系
    axs[0][0].plot(x1,weights_array1[:,0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][0].plot(x1,weights_array1[:,1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][0].plot(x1,weights_array1[:,2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
 
    x2 = arange(0, len(weights_array2), 1)
    #绘制w0与迭代次数的关系
    axs[0][1].plot(x2,weights_array2[:,0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][1].plot(x2,weights_array2[:,1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][1].plot(x2,weights_array2[:,2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    
    x2 = arange(0, len(weights_array3), 1)
    #绘制w0与迭代次数的关系
    axs[0][2].plot(x2,weights_array3[:,0])
    axs0_title_text = axs[0][2].set_title(u'newton上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][2].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][2].plot(x2,weights_array3[:,1])
    axs1_ylabel_text = axs[1][2].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][2].plot(x2,weights_array3[:,2])
    axs2_xlabel_text = axs[2][2].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][2].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    plt.savefig('p.png')
    plt.show() 