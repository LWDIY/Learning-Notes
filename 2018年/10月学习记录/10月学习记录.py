# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:45:28 2018

@author: Administrator
"""
'''决策树'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]

data = np.array(dataSet)
datas = pd.DataFrame({'年龄':data[:,0], 
                     '有工作':data[:,1], 
                     '有自己的房子':data[:,2], 
                     '信贷情况':data[:,3],
                     '是否贷款':data[:,4]})
target = datas.iloc[:,4]
values= datas.iloc[:,0:4].values.tolist()     #  利用tolist函数转化成list
features = datas.iloc[:,0:4]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(values,target)
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names = features.keys(),
                         class_names = clf.classes_,
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph
graph.render(r"C:\Users\Administrator\Desktop\tree4") 


'''朴素贝叶斯分类器'''

import pandas as pd
import matplotlib.pyplot as plt

file = r'C:\Users\Administrator\Desktop\balanced1.csv'
data = pd.read_csv(file)
#from random import shuffle        # 打乱数据，要列表格式
#data = data.values.tolist()       # 转换成列表
#shuffle(data)
#data = pd.DataFrame(data,columns=["holiday","times","weather","if happen"])   #  列表转回成dataframe，其中名称需要改回
target = data.iloc[:,3]
values = data.iloc[:,0:3].values.tolist()     #  利用tolist函数转化成list

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(values, target)
y_pred=clf.predict(values)
print("高斯朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (data.shape[0],(target != y_pred).sum()))
#高斯和伯努利效果一样？   BernoulliNB 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(target,y_pred)

def cm_plot(yt,yp):
    cm = confusion_matrix(yt,yp)
    plt.matshow(cm,cmap = plt.cm.Set3)   #cmap设置颜色
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy = (y,x))    # xy = (x,y) 标注坐标
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    return plt
cm_plot(target,y_pred)

TP = cm[0,0]
FP = cm[1,0]
FN = cm[0,1]
TN = cm[1,1]

#from sklearn.metrics import accuracy_score
#accuracy_score(target, y_pred)

print("准确率是：{0:0.3f}".format(TP/(TP + FP)))




'''LM神经网络'''

import pandas as pd
#from random import shuffle   #随机函数，打乱数据
data = pd.read_csv(r'C:\Users\Administrator\Desktop\balanced1.csv')
data = data.values    #data = data.values
target = data[:,3]
values = data[:,0:3]   

from keras.models import Sequential  #神经网络初始化函数
from keras.layers import Dense,Activation   # 导入层函数和激活函数
#modelfile = 'C:/Users/Administrator/Desktop/learning/data/net.model'
model = Sequential()
model.add(Dense(input_dim = 3,output_dim = 10))   #  输入层为3  ，隐藏层为10
model.add(Activation('relu'))   # 隐藏层使用 relu 激活函数
model.add(Dense(input_dim = 10,output_dim = 1))     #隐藏层10，输出层1
model.add(Activation('sigmoid'))  # 输出层激活函数为sigmoid函数
model.compile(loss = 'binary_crossentropy',optimizer = 'adam')
# model.compile  自定义损失函数， 
#loss = 'binary_crossentropy'  损失函数为'binary_crossentropy'  
#optimizer = 'adam'  ，优化控制  使用adam求解
model.fit(values,target,nb_epoch = 100,batch_size = 1)

#model.save_weights(modelfile)
predict_result = model.predict_classes(values).reshape(len(target))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(target,predict_result)

import matplotlib.pyplot as plt

def cm_plot(y, yp):
  cm = confusion_matrix(y, yp) 
  plt.matshow(cm, cmap=plt.cm.Set3) 
  plt.colorbar() 
  
  for x in range(len(cm)): 
    for y in range(len(cm)):
        plt.annotate(cm[x,y], xy=(y,x), horizontalalignment='center', verticalalignment='center')
  
  plt.ylabel('True label') 
  plt.xlabel('Predicted label') 
  return plt
cm_plot(target,predict_result)

TP = cm[0,0]
FP = cm[1,0]
FN = cm[0,1]
TN = cm[1,1]
#print("LM神经网络准确率是：{0:0.3f}".format((TP + TN)/(TP + FP + FN + TN)))
print("LM神经网络准确率是：{0:0.3f}".format(TP/(TP + FP)))




# 输入
# 计算地图上两点经纬度间的距离

from math import radians, cos, sin, asin, sqrt  
# Haversine(lon1, lat1, lon2, lat2)的参数代表：经度1，纬度1，经度2，纬度2（十进制度数）    
def Haversine(lon1, lat1, lon2, lat2): 
    # 将十进制度数转化为弧度  radians() 方法将角度转换为弧度  
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
    # Haversine公式  
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371 # 地球平均半径，单位为公里  
    d = c * r
    print("该两点间距离={0:0.3f} km".format(d))

Haversine(113.707695,29.972898,113.717538,29.974165)


#计算距离
#s12  两点间距离
#a12  地球球面弧度
#azi1 在第一个点的方位角
#azi2 在第2个点的方位角
# Geodesic.WGS84.Inverse(纬度1,经度1,纬度2,经度2)
from geographiclib.geodesic import Geodesic   
a = Geodesic.WGS84.Inverse(29.972898,113.707695,29.974165,113.717538)
print("该两点间距离={0:0.3f} m".format(a['s12']))    
#"{0:0.3f}".format(a['s12'])    f: 输出浮点数的标准浮点形式, '0.3f'表示保留小数部分,3位
#print('该两点间的距离=',format(a['s12'],'0.3f'),'m')
