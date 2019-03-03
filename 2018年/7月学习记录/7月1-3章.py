# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 08:47:06 2018

@author: Administrator
"""

'''2018-7-2    统计分析库   '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns         #  可视化

x = np.array([0.3,0.2,0.9,0.4,1.5,6.3,5.1,4.2,3.9,8.4])
np.mean(x)
x.mean()

np.var(x)
x.var()

np.median(x)
x.median()  #报错

np.max(x),  np.min(x)
x.max(),  x.min()

np.std(x)
x.std()

x.var(ddof=1)
# decrease degree of freedom，等于1表示计算公式里面是除以n-1（样本方差），而默认是除以n（方差）
x.std(ddof=1)

np.random.seed(123456789)

np.random.rand()    # 0-1之间的均匀分布
np.random.rand(5)

np.random.randn()   #  N(0,1) , 标准正态分布
np.random.randn(3,4)

np.random.randint(10,20,size = 10)    # 10-20之间的10个自然数，随机取(区间为[low,high）)
np.random.randint(low = 10,high = 20,size = (3,10))



plt.hist(np.random.rand(1000))

f,axes = plt.subplots(1,3,figsize = (12,3))
axes[0].hist(np.random.rand(1000))
axes[0].set(title = 'rand')

axes[1].hist(np.random.randn(1000))
axes[1].set_title('randn')

axes[2].hist(np.random.randint(low = 1,high = 10,size = 10000),bins = 9,align = 'left')
#bins = 9  表示有9个直方
axes[2].set_title('randint(low = 1,high = 10)')
fig.tight_layout()


a2 = np.random.choice(a=5, size=3, replace=False, p=[0.2, 0.1, 0.3, 0.4, 0.0])
# numpy.random.choice(a, size=3, replace=True, p=None)
# 参数意思分别 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布

from scipy import stats            # stats模块包含了多种概率分布的随机变量
from scipy import optimize         #最优化

x = stats.norm(1,0.5)
x.var()
x.std()
x.mean()
[x.moment(n) for n in range(5)]

#用moment查看分布的矩信息
#stats.norm.moment(n, loc=0, scale=1)
#标准正态分布的n阶原点矩







'''python数据分析与挖掘实战'''
'''1+2+3+...+100'''

s = 0
k = 0
while k < 100:
    k = k + 1
    s = s + k 
print(s)

s = 0
for k in range(101):
    s = s + k
print(s)

f = lambda x : x**2
f(6)
g = lambda x , y : y+x
g(2,3)

def add2(x,y):
     return [x+2,y+2]
print(add2(1,2))


def add2(x,y):
    return x+2,y+2
print(add2(1,2))

def add3(x,y):
    return x+2,y+3
a,b = add3(1,2)
print(a,b)

c = [1,'abc',[1,2],(1,2)]
c
a=c
a
a[0] = 2
a
c
b = c[:]
b
b[0] = 3
b
c

list('abcdefg')
tuple([1,2])
tuple(c)
sorted([1,3,5,4,2,9])
max((1,2,3))
len(b)
b.pop(1)
b
b.append(10)
b.remove(10)
b.insert(4,11)
b.count(10)
b.entend(['a','b'])


a = [1,2,3]
b = []
for i in a:
    b.append(i + 2)
b

b = [i + 2 for i in a]
b

d = {'today':10,'tomorrow':30}
d
d['today']
dict([['today',10],['tomorrow',20]])
dict.fromkeys(['today','tomorrow'],(10,20))

s = {1,2,2,3}
s
s1 = set([2,2,3,4])
s1
m = s|s1      #取并集
m
m2 = s & s1   # 取交集
m2
m3 = s - s1   # 取差集
m3
m4 = s ^ s1   #对称差集
m4

#  n！
a = [1,2,3]
b = map(lambda x : x+2,a)      # b = [i + 2 for i in a]
b = list(b)

from fuctools import reduce
reduce(lambda x,y : x*y,range(1,n+1))

s = 1
for i in range(2,n+1):
    s = s*i
print(s)

b = filter(lambda x : x>5 and x<8,range(10))  
b = list(b)

b = [x for x in range(10) if x>5 and x<8]
b = [x>5 and x<8 for x in range(10)]

import math
math.sin(1)
math.exp(2)
math.pi

from math import exp as e
e(2)
from math import sin as s
s(1)

import numpy as np
a = np.array([2,0,1,8])
sorted(a)
a
a.sort()

'''Scipy'''
#求解非线性方程组2x1-x2^2=1,x1^2-x2=2
from scipy.optimize import fsolve
def f(x):
    x1 = x[0]         #x1,x2 = x.tolist()
    x2 = x[1]
    return[2*x1 - x2**2 - 1 ,x1**2 - x2 - 2]   #  要通过移项将方程组右边变成0，否则报错
result = fsolve(f, [1,1])   
print(result) 


#数值积分
from scipy import integrate
def g(x):
    return (1-x**2)**0.5
pi_2, err = integrate.quad(g,-1,1)
print(pi_2)

'''matplotlib'''
import numpy as np
import matplotlib.pyplot as plt

x =np.linspace(0,10,1000)
y = np.sin(x) + 1
z = np.cos(x**2) + 1
plt.figure(figsize = (8,4))
plt.plot(x,y,'r',lw = 2,label = '$\sin x+1$')
plt.plot(x,z,'b--',label = '$\cos x^2+1$')
plt.xlabel('Time(s)')
plt.ylabel('Volt')
plt.title('A Simple Example')
plt.ylim(0,2.2)
plt.legend()
plt.show()


import pandas as pd
d = pd.DataFrame([[1,2,3],[4,5,6]])
d
'''StatsModel   用python进行ADF检验'''
from statsmodels.tsa.stattools import adfuller as ADF
import numpy as np
ADF(np.random.rand(100))

'''Sklearn    建立线性回归模型'''
from sklearn.linear_model import LinearRegression  #导入线性回归模型
model = LinearRegression()
print(model)

'''鸢尾花数据'''
from sklearn import datasets
iris = datasets.load_iris()   #加载数据集
print(iris.data.shape)
from sklearn import svm   #  导入SVM模型

clf =  svm.LinearSVC()    #3建立线性SVM分类器
clf.fit(iris.data,iris.target)
clf.predict([[5.0,3.6,1.3,0.25]])
clf.coef_




def CommDevisor(a,b):
  r = a % b
  while r != 0:
    a = b
    b = r
    r = a % b
  return b
print(CommDevisor(963,657))



def gcd(a,b):
    while b:
        r = a%b
        a = b
        b = r
    return a 
print(gcd(963,657))





'''2018-7-10 - 7-11'''

import pandas as pd
import xlrd         #  读取Excel文件
catering_sale = 'C:/Users/Administrator/Desktop/chapter3/demo/data/catering_sale.xls'
data = pd.read_excel(catering_sale,index_col = '日期')
data.describe()
len(data)

''' 箱线图检验异常值 '''
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   #  用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     #  用来正常显示负号
plt.figure() #建立图像
p = data.boxplot(return_type='dict')        #画箱线图，直接使用DataFrame的方法
x = p['fliers'][0].get_xdata()              # 'flies'即为异常值的标签
y = p['fliers'][0].get_ydata()
y.sort()                                   #从小到大排序，该方法直接改变原对象

#用annotate添加注释
#其中有些相近的点，注解会出现重叠，难以看清，需要一些技巧来控制。
#以下参数都是经过调试的，需要具体问题具体调试 
#xy 为被注释的坐标点
#xytext 为注释文字的坐标位置

for i in range(len(x)): 
  if i>0:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.05 -0.8/(y[i]-y[i-1]),y[i]))
  else:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.08,y[i]))
plt.show() #展示箱线图

data = data[(data['销量'] >400)&(data['销量'] <5000)]
sta = data.describe()
sta.loc['range'] = sta.loc['max'] - sta.loc['min']    #  极差
sta.loc['CV'] = sta.loc['std']/sta.loc['mean']        # 变异系数 
sta.loc['dis'] = sta.loc['75%'] - sta.loc['25%']      #四分位差
sta



import pandas as pd
import xlrd
dish_profit = 'C:/Users/Administrator/Desktop/chapter3/demo/data/catering_dish_profit.xls'
data = pd.read_excel(dish_profit,index_col = '菜品名')
data = data['盈利'].copy()
data.sort(ascending = False)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
data.plot(kind = 'bar')
plt.ylabel('盈利')
p = 1.0*data.cumsum()/data.sum()   # data.cumsum()  求累计次数
p.plot(color = 'r',secondary_y = True,style = '-o',lw = '2')
plt.annotate(format(p[6], '.4%'), xy = (6, p[6]), xytext=(6*0.9, p[6]*0.9), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")) #添加注释，即85%处的标记。这里包括了指定箭头样式。
plt.ylabel('盈利(比例)')
plt.show()



import pandas as pd
catering_sale = 'C:/Users/Administrator/Desktop/chapter3/demo/data/catering_sale_all.xls'
data = pd.read_excel(catering_sale,index_col = '日期')
data.corr()      #相关系数矩阵，即给出了任意两款菜式之间的相关系数
data.corr()[u'百合酱蒸凤爪']  #只显示“百合酱蒸凤爪”与其他菜式的相关系数
data[u'百合酱蒸凤爪'].corr(data[u'翡翠蒸香茜饺'])  #计算“百合酱蒸凤爪”与“翡翠蒸香茜饺”的相关系数

D = pd.DataFrame([range(1,8),range(2,9)])
D.corr(method = 'spearman')
s1 = D.loc[0]
s2 = D.loc[1]
s1.corr(s2,method = 'pearson')
s1.cov(s2)       # 计算协方差

import numpy as np
d = pd.DataFrame(np.random.randn(6,5))
d[0].corr(d[1])
d.cov()
d[0].cov(d[1])   #  第一列和第二列的协方差
d.skew()          # 偏度
d.kurt()          # 峰度
d.describe()

x = np.linspace(0,2*np.pi,50)
y = np.sin(x)
plt.plot(x,y,'bp--')

'''  画饼图
 explode——与x长度相等的list,饼切分距离 
label——标签,与x长度相等的list 
autopct——格式化百分比，如：“%1.1f%%”,保留一位小数显示百分比   如将0.5转成百分数a = '%1.1f%%' % (0.5*100)
shadow——阴影显示与否，False or True '''
import matplotlib.pyplot as plt
labels = 'Froge','Hogs','Dogs','Logs'
x = [15,30,45,10]         #  每一块比例
colors = ['yellowgreen','gold','lightskyblue','lightcoral']
explode = (0,0.1,0,0)     #  突出显示第二块
plt.pie(x,explode = explode,labels = labels,colors = colors,autopct = '%1.1f%%',shadow = True,startangle = 90)
plt.axis('equal')    #  显示为圆（避免压缩比例为椭圆）
plt.show()


'''直方图plt.hist()'''
import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(1000)   #  1000个服从正态分布的随机数
plt.hist(x,10)              #  分成10组绘制直方图

''' 箱线图   D.boxplot()  /   D.plot(kind = 'box') '''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x = np.random.randn(1000)
d = pd.DataFrame([x,x+1]).T
d.plot(kind = 'box')      # d.boxplot()
plt.show()


'''绘制对数数据图  D.plot(logx = True) / D.plot(logy = True) '''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

x = pd.Series(np.exp(np.arange(20)))
x.plot(label = '原始数据',legend = True)
x.plot(logy = True,label = '对数数据图',legend = True)
plt.show()


'''绘制误差条形图 D.plot(yerr = error) / D.plot(xerr = error) '''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

error = np.random.randn(10)
y = pd.Series(np.sin(np.arange(10)))   # 均值数据列
y.plot(yerr = error)
plt.show()


















