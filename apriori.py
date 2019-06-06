from numpy import *

def loadDataSet():
    return [[1, 2,3, 4,6], [2, 3,4, 5,6], [1, 2, 3, 5,6], [1,2,4, 5,6]]

def createC1(dataSet): # 创建第一个组合列表项集
    '''
        对于dataset中的项目
    '''
    C1 = [] # 初始化组合列表项集
    for transaction in dataSet: # 数据集中的每一项
        for item in transaction: # 项中每一个元素
            if not [item] in C1:
                C1.append([item]) # 不重复的添加到C1中          
    C1.sort() # 排序
    return list(map(frozenset, C1))#use frozen set so we can use it as a key in a dict    

def scanD(D, Ck, minSupport):
    '''
        D表示dataset，对于Ck中的每个项目组合计算在D中出现的次数，
        然后计算支持度，最后返回retList：Ck中符合要求的组合的列表
        和supportData:符合要求的组合的具体支持度
    '''
    ssCnt = {}
    for tid in D: # 对于数据集里的每一项
        for can in Ck: # 对于组合列表中的每一项
            if can.issubset(tid): # 如果组合列表项是数据集项的子集
                if not can in ssCnt:
                    ssCnt[can]=1
                else: ssCnt[can] += 1 # 设置字典[组合列表项(frozenSet)]+1
    numItems = float(len(list(D))) # 数据集项的项数
    retList = [] # 返回列表
    supportData = {} # 符合要求的组合的具体支持度
    for key in ssCnt: # 对于字典[组合列表项]的每一个组合列表项
        support = ssCnt[key]/numItems # 计算对应的支持度
        if support >= minSupport: # 如果支持度 >= 最小支持度
            retList.insert(0,key) # 组合列表项插入返回列表中
        supportData[key] = support # 将(组合列表项, 对应支持度) 添加到supportData中
    return retList, supportData # 返回列表, supportData


def aprioriGen(Lk, k): #creates Ck
    '''
        根据Lk中的集合自动组合生成下一个Ck
    '''
    retList = [] # 初始化返回列表
    lenLk = len(Lk) # 上一轮支持度检测后的返回列表长度
    for i in range(lenLk): # 对于支持度检测后的返回列表中的每一个元素
        for j in range(i+1, lenLk): # 对于这个元素后的所有元素
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2] # 截取这两项中出最后一个元素外的项作为L1,L2
            L1.sort(); L2.sort() # 排序再进行比较
            if L1==L2: # 如果两项除了最后一个元素外都相同
                retList.append(Lk[i] | Lk[j]) # 这两项的并集添加到返回列表中
    return retList # 返回列表

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet) # 从dataSet中得出C1
    D = list(map(set, dataSet)) # 将数据集中的每一项集合化后作为D
    #print(D)
    L1, supportData = scanD(D, C1, minSupport) # 根据支持度筛选后得出支持度列表以及返回项列表
    L = [L1] # 频繁项集汇总
    k = 2 # 第二轮
    while (len(L[k-2]) > 0): # 上一轮LK不是空的的话
        Ck = aprioriGen(L[k-2], k) # 生成下一个待检测的CK
        Lk, supK = scanD(D, Ck, minSupport) # 对Ck进行支持度检测得出LK以及支持度情况
        supportData.update(supK) # 把每一轮的支持度情况汇总
        L.append(Lk) # 把每一轮的频繁项集汇总
        k += 1 # 轮数++
    return L, supportData # 返回频繁项集汇总以及支持度汇总


def generateRules(L, supportData, minConf=0.7): # 生成规则
    '''
    L:频繁项集列表，supportData：包括对应频繁项集合的字典。minConf:最小可信度阈值
    返回bigRuleList，规则列表
    '''
    bigRuleList = [] # 初始化规则列表
    for i in range(1, len(L)): # 从频繁项集汇总中选取每一轮的频繁项集(长度至少为2)
        for freqSet in L[i]: # 对于频繁项集中的每一个频繁项 eg:[[1, 2], [3, 4]]
            # 注意此时H1为a list of frozenset, 元素一开始都是单个出现的,如[frozenset(1), frozenset(2)], 而不会有[frozenset(1, 2), frozenset(3, 4)]
            H1 = [frozenset([item]) for item in freqSet] # 把频繁项中的元素frozenSet化
            if (i > 1): #如果有两个以上元素的频繁项集，尝试进一步合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else: #如果只有两个元素，那么直接计算合乎置信度要求的规则。
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

def calcConf(freqSet, H, supportData, brl, minConf=0.7): # 两个元素的计算
    '''
        freqSet: 频繁项集 H：频繁项集中的所有元素的列表，supportData：包括对应频繁项的支持数据的字典。
        minConf：最小置信度，brl：规则列表(一开始为空)
        返回：prunedH：满足最小可信度要求的规则的右边项
        对brl做修改，加入合乎要求的规则
    '''
    prunedH = [] # 初始化返回列表
    for conseq in H: #conseq：后件，对于H中的每一个元素尝试把它作为后件
        # 因为此时频繁项只有两个元素,前件元素=频繁项-后件元素
        conf = supportData[freqSet]/supportData[freqSet-conseq] # 置信度 = 支持度(包含后件元素的频繁项) / 支持度(前件元素)
        if conf >= minConf: # 置信度 > 最小置信度
            print(freqSet-conseq,'-->',conseq,'conf:',conf) # 输出规则
            brl.append((freqSet-conseq, conseq, conf)) # 规则以元组形式添加进规则列表
            prunedH.append(conseq) # 后件元素添加进后件列表
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7): # 多个元素的计算
    '''
        freqSet: 频繁项集 H：频繁项集中的所有元素的列表，supportData：包括对应频繁项的支持数据的字典。
        minConf：最小置信度，brl：规则列表(一开始为空)
    '''
    if len(H)==0: # 如果此轮频繁项为空(最后一轮)
        return # 返回空
    
    m = len(H[0]) # 频繁项长度(含有元素个数)
    
    if m == 1: # 后件只有一项(一开始就只有一项)
        # 过滤只有一个项的后件（产生规则如1,2,3->4，并返回过滤后的后件）
        H = calcConf(freqSet, H, supportData, brl, minConf)
    
    # 因为前件 = 频繁集某项 - 后件(同一个元素不能同时出现在前后), 所以频繁集 > 后件长度 + 1
    if (len(freqSet) > (m + 1)): # 如果后件的长度 < 频繁集长度 - 1
        Hmp1 = aprioriGen(H, m+1) # 根据当前frozenset列表产生更长的后件的列表
        # 扔进calcConf与当前频繁集计算规则,返回符合最小置信度的规则的后件Hmp1
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # 如果Hmp1不只有一个,证明还可以继续组合这些后件,递归调用(只有一个就组合不了了)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

'''
dataset=loadDataSet()
C1=createC1(dataset)
retList,supportData=scanD(dataset,C1,0.5)
print 'C1:',C1
print 'retList:',retList
print 'supportData:',supportData
'''
# dataSet=loadDataSet()
# L,supportData=apriori(dataSet,0.7)
# brl=generateRules(L, supportData,0.7)
# print('brl:',brl)