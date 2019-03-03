# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:56:50 2018

@author: Administrator
"""

'''拉格朗日插值'''

import pandas as pd
from scipy.interpolate import lagrange

inputfile = 'C:/Users/Administrator/Desktop/learning/data/missing_data.xls'
outputfile = 'C:/Users/Administrator/Desktop/learning/data/missing_data_processed.xls'

data = pd.read_excel(inputfile,header = None)

def ployinterp_column(s,n,k=5):
    y = s[list(range(n-k,n)) + list(range(n+1,n+1+k))]
    y = y[y.notnull()]
    return lagrange(y.index,list(y))(n)

for i in data.columns: 
    for j in range(len(data)):
        if (data[i].isnull())[j]:
            data[i][j] = ployinterp_column(data[i],j)
            
data.to_excel(outputfile,header = None,index=False)


'''2018-8-25
1.数据分为测试数据和训练数据。
2.LM神经网络构建。'''

import pandas as pd
from random import shuffle   #随机函数，打乱数据
data = pd.read_excel(r'C:\Users\Administrator\Desktop\learning\data\model.xls')
data = data.as_matrix()      #data = data.values
shuffle(data)
train = data[:int(len(data)*0.8),:]
test = data[int(len(data)*0.8):,:]

from keras.models import Sequential  #神经网络初始化函数
from keras.layers import Dense,Activation   # 导入层函数和激活函数
modelfile = 'C:/Users/Administrator/Desktop/learning/data/net.model'
model = Sequential()
model.add(Dense(input_dim = 3,output_dim = 10))   #  输入层为3  ，隐藏层为10
model.add(Activation('relu'))   # 隐藏层使用 relu 激活函数
model.add(Dense(input_dim = 10,output_dim = 1))     #隐藏层10，输出层1
model.add(Activation('sigmoid'))  # 输出层激活函数为sigmoid函数
model.compile(loss = 'binary_crossentropy',optimizer = 'adam')
# model.compile  自定义损失函数， 
#loss = 'binary_crossentropy'  损失函数为'binary_crossentropy'  
#optimizer = 'adam'  ，优化控制  使用adam求解
model.fit(train[:,:3],train[:,3],nb_epoch = 100,batch_size = 1)

model.save_weights(modelfile)
predict_result = model.predict_classes(train[:,:3]).reshape(len(train))

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(train[:,3],predict_result)
import matplotlib.pyplot as plt

def cm_plot(y, yp):
  cm = confusion_matrix(y, yp) 
  plt.matshow(cm, cmap=plt.cm.Greens) 
  plt.colorbar() 
  
  for x in range(len(cm)): 
    for y in range(len(cm)):
        plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
  
  plt.ylabel('True label') 
  plt.xlabel('Predicted label') 
  return plt
cm_plot(train[:,3],predict_result)

TP = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TN = cm[1,1]

print("准确率是：{0:0.3f}".format(TP/(TP + FP)))


'''2018-8-9    自动化测试'''


from selenium import webdriver
browser = webdriver.Chrome()
url = 'http://www.baidu.com'
browser.get(url)
browser.find_element_by_id("kw").send_keys("李伟")
browser.find_element_by_id("su").click()

#########百度输入框的定位方式##########
'''<input id="kw" class="s_ipt" type="text" maxlength="100" name="wd"
autocomplete="off">'''

#通过 id 方式定位
browser.find_element_by_id("kw").send_keys("李伟")
#通过 name 方式定位
browser.find_element_by_name("wd").send_keys("李伟")
#通过 tag name 方式定位
browser.find_element_by_tag_name("input").send_keys("李伟")
#通过 class name 方式定位
browser.find_element_by_class_name("s_ipt").send_keys("李伟")
#通过 CSS 方式定位
browser.find_element_by_css_selector("#kw").send_keys("李伟")
#通过 xphan 方式定位
browser.find_element_by_xpath("//input[@id='kw']").send_keys("李伟")
############################################

'''2018-8-10'''

from selenium import webdriver
browser = webdriver.Chrome()
url = 'http://www.baidu.com'
browser.get(url)

browser.maximize_window()    #   将浏览器最大化显示
browser.set_window_size(480, 800)   #    设置浏览器宽480、高800显示
browser.implicitly_wait(30)         #   智能等待30秒

browser.find_element_by_link_text('贴吧').click()
browser.find_element_by_partial_link_text("贴").click()
#通过 find_element_by_partial_link_text() 函数，我只用了“贴”字，脚本一样找到了"贴 吧" 的链接

browser.find_element_by_css_selector('a[name=\'tj_trnews\']').click()



'''
本章重点：
 键盘按键用法
 键盘组合键用法
 send_keys() 输入中文乱码问题
要想调用键盘按键操作需要引入 keys 包：
from selenium.webdriver.common.keys import Keys
通过 send_keys()调用按键：
send_keys(Keys.TAB) # TAB
send_keys(Keys.ENTER) # 回车
注意：这个操作和页面元素的遍历顺序有关，假如当前定位在账号输入框，按键
盘的 tab 键后遍历的不是密码框，那就不会输入密码。 假如输入密码后，还有
需要填写验证码，那么回车也起不到登陆的效果。'''
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
browser = webdriver.Chrome()
url = 'https://mail.qq.com/'
browser.get(url)
#browser.maximize_window()
browser.find_element_by_xpath('//*[@id="u"]').clear() 
browser.find_element_by_xpath('//*[@id="u"]').click()
#browser.find_element_by_xpath('//*[@id="u"]').clear()    #  清除以前历史信息
browser.find_element_by_xpath('//*[@id="u"]').send_keys('761402850')  #  输入账号
browser.find_element_by_xpath('//*[@id="u"]').send_keys(Keys.TAB)   #  tab换到下一个框
time.sleep(3)
browser.find_element_by_xpath('//*[@id="p"]').send_keys('***')   #  通过定位密码框
browser.find_element_by_xpath('//*[@id="p"]').send_keys(Keys.ENTER)   #  回车



'''2018-8-21'''

import numpy as np
import matplotlib.pyplot as plt

values = []
labels = []
with open(r'C:\Users\Administrator\Desktop\learning\data\svm.txt') as f:
    fr = f.readlines()
    for line in fr:
        data = line.strip().split('\t')
        values.append([float(data[0]),float(data[1])])
        labels.append(float(data[2]))
        
'''data = [line.strip().split('\t') for lin in f.readlines()]'''

def showdata(values,labels):
    data_plus = []
    data_minus = []
    for i in range(len(values)):
        if labels[i] > 0:
            data_plus.append(values[i])
        else:
            data_minus.append(values[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(data_plus_np.T[0],data_plus_np.T[1])
    plt.scatter(data_minus_np.T[0],data_minus_np.T[1])
#    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])   #正样本散点图
#    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) #负样本散点图
#    np.transpose在二维的时候相当于是矩阵转置   
    plt.show()
showdata(values,labels)    


'''2018-8-23  SVM   '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\Administrator\Desktop\learning\data\svm.csv')
x = dataset.iloc[:,[2,3]].values  #没有.values  返回的结果是个dataframe,有了之后  变成矩阵形式
y = dataset.iloc[:,4].values

from sklearn.cross_validation import train_test_split
#随机划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

from sklearn.preprocessing import StandardScaler  #数据标准化
#StandarScaler可以在训练数据集上做了标准转换操作之后，把相同的转换应用到测试训练集中。
#可以对训练数据，测试数据应用相同的转换，
#以后有新的数据进来也可以直接调用，不用再重新把数据放在一起再计算一次了。
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear',random_state = 0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)  #预测测试集

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#训练集结果
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#测试集结果
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



'''2018-8-25'''

from sklearn import datasets
iris = datasets.load_iris()
print(iris.data.shape)
from sklearn import svm
clf = svm.LinearSVC()
clf.fit(iris.data,iris.target)
clf.predict([[5.0,3.6,1.3,0.25]])
a = clf.coef_

from keras import Sequential
from keras.layers import Dense,Activation,Dropout
model = Sequential()
model.add(Dense(input_dim = 20,output_dim = 64))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(input_dim = 64,output_dim = 64))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(input_dim = 64,output_dim = 1))
model.add(Activation('sigmoid'))
from keras.optimizers import SGD
sgd = SGD(lr = 0.1,decay = 1e-6,momentum = 0.9,nesterov = True)
model.compile(loss = 'mean_squared_error',optimizer = sgd)
model.fit(x_train,y_train,epochs = 20,batch_size = 16)
score = model.evalutate(x_test,y_test,batch_size = 16)



'''2018-8-29   Simple Linear-Regression'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file = r'C:\Users\Administrator\Desktop\learning\data\studentscores.csv'
dataset = pd.read_csv(file)
#  dataset = pd.read_csv(file).values
#  x = dataset[:,0]
#  y = dataset[:,1]
x = dataset.iloc[:,:1].values
y = dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg = reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

plt.scatter(x_train,y_train,c = 'r')
plt.plot(x_train,reg.predict(x_train),c = 'b')

plt.scatter(x_test , y_test, color = 'red')
plt.plot(x_test , reg.predict(x_test), color ='blue')


'''Multiple Linear Regression'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file = r'C:\Users\Administrator\Desktop\learning\data\50_Startups.csv'
dataset = pd.read_csv(file)
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#对分类型特征值进行编码，对不连续的数值或文本进行编码, 即本例中的 city
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
x[: , 3] = labelencoder.fit_transform(x[:,3])
# labelencoder.classes_  查看标签种类  
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

x = x[: , 1:]

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
#random_state = 0,随机数种子控制每次划分训练集和测试集的模式，其取值不变时划分得到的结果一模一样，
#其值改变时，划分得到的结果不同。若不设置此参数，则函数会自动选择一种随机模式，得到的结果也就不同

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)



'''2018-8-30   Logistic Regression  '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = r'C:\Users\Administrator\Desktop\learning\data\Social_Network_Ads.csv'
dataset = pd.read_csv(file)
x = dataset.iloc[:,2:4].values
y = dataset.iloc[:,-1].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

from sklearn.preprocessing import StandardScaler
#标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
'''二者的功能都是对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等）
fit_transform(partData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），
然后对该partData进行转换transform，从而实现数据的标准化、归一化等等。。
根据对之前部分fit的整体指标，对剩余的数据（restData）使用同样的均值、方差、最大最小值等指标进行转换transform(restData)，
从而保证part、rest处理方式相同。
必须先用fit_transform(partData)，之后再transform(restData)
如果直接transform(partData)，程序会报错
如果fit_transfrom(partData)后，使用fit_transform(restData)而不用transform(restData)，
虽然也能归一化，但是两个结果不是在同一个“标准”下的，具有明显差异。'''

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

def cm_plot(yt, yp):
  cm = confusion_matrix(yt, yp) 
  plt.matshow(cm, cmap=plt.cm.Set3_r)   #  cmap  颜色图谱（colormap),设置颜色渐变,
  plt.colorbar()   #  colorbar 在绘图中显示颜色渐变条
  
  for x in range(len(cm)): 
    for y in range(len(cm)):
        plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
#annotate标注文字，annotate语法说明 ：annotate(s='str' ,xy=(x,y) ,xytext=(l1,l2) ,..)
#s 为注释文本内容 ，xy 为被注释的坐标点 ，xytext 为注释文字的坐标位置
#verticalalignment设置水平对齐方式 ，可选参数 ： 'center' , 'top' , 'bottom' ,'baseline' 
#horizontalalignment设置垂直对齐方式，可选参数：left,right,center 
  plt.ylabel('True label') 
  plt.xlabel('Predicted label') 
  return plt
cm_plot(y_test,y_pred)

'''最近邻   K-Nearest-Neighbours  '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = r'C:\Users\Administrator\Desktop\learning\data\Social_Network_Ads.csv'
dataset = pd.read_csv(file)
x = dataset.iloc[:,2:4].values
y = dataset.iloc[:,-1].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p = 2)
#  p:指定在’minkowski’度量上的指数   ’1’:曼哈顿距离  ‘2’:欧拉距离
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

import matplotlib.pyplot as plt

def cm_plot(yt,yp):
    cm = confusion_matrix(yt,yp)
    plt.matshow(cm,cmap = plt.cm.Set3)   #cmap设置颜色
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy = (x,y))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    return plt
cm_plot(y_test,y_pred)

TP = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TN = cm[1,1]
print("准确率是：{0:0.3f}".format((TP + TN)/(TP + FP + FN + TN)))






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file = r'F:\learning\data\houseprice.csv'
dataset = pd.read_csv(file)
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,8].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state = 0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

a  = reg.intercept_#截距
b = reg.coef_#回归系数
print("拟合参数:截距",a,",回归系数：",b)
y_pred = reg.predict(x_test)

plt.plot(range(len(y_pred)),y_pred,'red', linewidth=2.5,label="predict data")
plt.plot(range(len(y_test)),y_test,'y',label="test data")
plt.legend()
plt.show()

import seaborn as sns
sns.pairplot(dataset, x_vars=['TotRmsAbvGrd','Ce0tralAir','TotalBsmtSF','1stFlrSF','2ndFlrSF','YearInterval','Garage3ish','1terQual'],y_vars='SalePrice', size=3, aspect=0.8, kind='reg')  
plt.legend()
plt.show()


    
sns.regplot(dataset['TotRmsAbvGrd'],dataset['SalePrice'],color='red')
sns.regplot(dataset['Ce0tralAir'],dataset['SalePrice'],color='y')
sns.regplot(dataset['TotalBsmtSF'],dataset['SalePrice'],color='r')
sns.regplot(dataset['1stFlrSF'],dataset['SalePrice'],color='y')
sns.regplot(dataset['2ndFlrSF'],dataset['SalePrice'],color='r')
sns.regplot(dataset['YearInterval'],dataset['SalePrice'],color='y')
sns.regplot(dataset['Garage3ish'],dataset['SalePrice'],color='r')
sns.regplot(dataset['1terQual'],dataset['SalePrice'],color='y')
