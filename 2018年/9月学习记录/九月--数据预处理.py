# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 14:47:15 2018

@author: Administrator
"""


'''2018-9-5   counter 统计频数 '''

import pandas as pd
file = r'C:\Users\Administrator\Desktop\learning\data\provences.csv'
data = pd.read_csv(file)
dataset1 = data.iloc[:,2]
dataset2 = data.iloc[:,4]
a = dataset1 + dataset2
from collections import Counter       
counts= Counter(a)

from collections import Counter
words = "This is for the test of Counter class".split()
cnt = Counter(words)


'''矩阵拼接'''
import numpy as np
u= np.mat([1,1,2,3,5,8])

x = np.mat([[1,0],[0,1]])
y = np.mat([[1,2],[3,4]])
z = np.mat([[1,2,1,2],[3,4,3,4],[1,2,1,2]])
xx=np.concatenate((x,y), axis=0)
yy=np.concatenate((x,y), axis=0)
ww=np.concatenate((xx,yy), axis=1)


#import numpy as np
#a = np.array([[1, 0], [0, 1]])
#b = np.array([[5, 6], [7, 8]])
#cc=np.concatenate((a,b), axis=0)
#dd=np.concatenate((a,b), axis=0)
#ff=np.concatenate((cc,dd), axis=1)


import pandas as pd
import matplotlib.pyplot as plt
file = r'C:\Users\Administrator\Desktop\learning\data\Table.csv'
data = pd.read_csv(file,encoding = 'gbk')
time = data.iloc[:,0]
Highest = data.iloc[:,1]
fig1 = plt.figure(figsize=(10,5))
fig1.add_subplot(1,1,1)
fig1.autofmt_xdate()     # 日期的排列根据图像的大小自适应
plt.plot(time,Highest,'o-')


'''Decision Tree '''

'''官网例子
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
clf.classes_
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

#保存到pdf文件
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 
'''

使用np.array()函数把DataFrame转化为np.ndarray()，
再利用tolist()函数把np.ndarray()转为list


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from sklearn import tree
file = r'C:\Users\Administrator\Desktop\learning\data\lenses.txt'
fr = open(file)
lenses = [line.strip().split('\t') for line in fr]
arr = np.array(lenses)
lenses_pd = pd.DataFrame({'age':arr[:,0],
                   'prescript':arr[:,1],
                   'astigmatic':arr[:,2],
                   'tearRate':arr[:,3]})
lenses_target = arr[:,4]
'''
clf = tree.DecisionTree()
lenses = clf.fit(lenses,lensesLabels)

在fit()函数不能接收`string`类型的数据，通过打印的信息可以看到，数据都是`string`类型的。
在使用fit()函数之前，我们需要对数据集进行编码，这里可以使用两种方法：
LabelEncoder ：将字符串转换为增量值
OneHotEncoder：使用One-of-K算法将字符串转换为整数
这样方便我们的序列化工作。编写代码如下：'''

from sklearn.preprocessing import LabelEncoder
from sklearn.externals.six import StringIO
le = LabelEncoder()                                                        #创建LabelEncoder()对象，用于序列化           
for col in lenses_pd.columns:                                            #序列化
    lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # print(lenses_pd)                                                        #打印编码信息
clf = tree.DecisionTreeClassifier(max_depth = 4)                        #创建DecisionTreeClassifier()类
clf = clf.fit(lenses_pd.values.tolist(), lenses_target)                    #使用数据，构建决策树
dot_data = StringIO()
tree.export_graphviz(clf, out_file = dot_data ,                            #绘制决策树
                     feature_names = lenses_pd.keys(),
                     class_names = clf.classes_,
                     filled=True, rounded=True,
                     special_characters=True)
                                                                          #保存绘制好的决策树，以PDF的形式存储。
graph = graphviz.Source(dot_data)



a = lenses_pd.values.tolist()



















