# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 20:22:15 2018

@author: Administrator
"""

'''朴素贝叶斯分类器'''

import pandas as pd
import matplotlib.pyplot as plt

file = r'C:\Users\Administrator\Desktop\testdata\test_30_8.csv'
data = pd.read_csv(file)
#from random import shuffle        # 打乱数据，要列表格式
#data = data.values.tolist()       # 转换成列表
#shuffle(data)
#data = pd.DataFrame(data,columns=["holiday","times","weather","if happen"])   #  列表转回成dataframe，其中名称需要改回
target = data.iloc[:,3]
values = data.iloc[:,0:3].values.tolist()     #  利用tolist函数转化成list
from sklearn.cross_validation import train_test_split
#随机划分训练集和测试集
values_train,values_test,target_train,target_test = train_test_split(values,target,test_size = 0.25,random_state = 0)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(values_train, target_train)
y_pred=clf.predict(values_test)
print("高斯朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (target_test.shape[0],(target_test != y_pred).sum()))
#高斯和伯努利效果一样？   BernoulliNB 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(target_test,y_pred)

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
cm_plot(target_test,y_pred)

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
data = pd.read_csv(r'C:\Users\Administrator\Desktop\testdata\test_30_8.csv')
#data = data.values    #data = data.values
from random import shuffle        # 打乱数据，要列表格式
data = data.values.tolist()       # 转换成列表
shuffle(data)
data = pd.DataFrame(data,columns=["holiday","times","weather","if happen"])   #  列表转回成dataframe，其中名称需要改回
data = data.values
target = data[:,3]
values = data[:,0:3] 
  
from sklearn.cross_validation import train_test_split
#随机划分训练集和测试集
values_train,values_test,target_train,target_test = train_test_split(values,target,test_size = 0.25,random_state = 0)

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
model.fit(values_train,target_train,nb_epoch = 10,batch_size = 1)

#model.save_weights(modelfile)
predict_result = model.predict_classes(values_test).reshape(len(target_test))
print("测试样本总数 ：%d ，预测错误样本数 : %d" % (target_test.shape[0],(target_test != predict_result).sum()))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(target_test,predict_result)

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
cm_plot(target_test,predict_result)

TP = cm[0,0]
FP = cm[1,0]
FN = cm[0,1]
TN = cm[1,1]
#print("LM神经网络准确率是：{0:0.3f}".format((TP + TN)/(TP + FP + FN + TN)))
print("LM神经网络准确率是：{0:0.3f}".format(TP/(TP + FP)))





