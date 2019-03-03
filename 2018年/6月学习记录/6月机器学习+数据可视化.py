# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 12:20:42 2018

@author: Administrator
"""

'''机器学习'''

"""2018.6.11    Numpy arrays基础"""

import numpy as np

"""和list很像"""
lst = [10,20,30,40]
print(lst)
arr = np.array([10,20,30,40])
print(arr)

"""1维array的索引类似list"""
lst[0]
arr[0]
arr[-1]
arr[2:]
arr[:-2]
arr[-2:]

"""arrays和lists的区别:  list可以混合类型   array单一类型"""
lst[-1] = 'hello world'
lst
#arr用这种会报错

arr.dtype   #访问元素数据类型
arr[-1] = 1.234
arr
#dtype一旦定下，不会变化，赋值之后也会强制转换

'''新建Arrays'''
np.zeros(5,dtype = float)  ##产生的值都是0
np.zeros(5,dtype = int)
np.zeros(5,dtype = complex)
np.ones(5)    ##产生的值都是1
b = np.empty(4)       #产生的值都是空值
b.fill(5.5)    #填充值

'''定义序列'''
np.arange(5)   #生成自然数构成的array，使用方法与内置函数range一样
c = [i for i in range(5)]
d = [i for i in np.arange(5)]

#linspace 和 logspace函数创建linearly and logarithmically间隔的格点
print('0和1之间的线性格点：')
print(np.linspace(0,1,4))   #生成包含0,1的4个数值点，点之间都是等分切
print('10**1和10**3之间的对数格点：')
print(np.logspace(1,3,4))

'''创建random arrays'''
np.random.randn(5)
norm10 = np.random.normal(10,3,5)   #生成5个样本，均值10， 标准差3
norm10[norm10 > 10]    #返回norm10所有的大于10的元素

'''用array作为索引'''
mask = norm10 > 9    ##判断norm10所有的元素是否都大于9，如果满足返回true，否则为false
mask
print('小于10的值：',norm10[norm10 < 10] )

norm10[norm10 > 9] = 0   #重设置大于9的值为0

'''大于1维的Arrays'''
lst2 = [[1,2],[3,4]]   #二维list 

lst2[0][1]             #返回第一行第二列的数值

lst2[1][1]             #返回第二行第二列的数值

arr2 = np.array([[1,2],[3,4]])

arr2[0,1]              #返回第一行第二列的数值

arr2[1,1]              #返回第二行第二列的数值

arr3 = np.zeros((2,3))    #返回2行3列的array，且值全部为0

arr3.shape                #查看行列数

arr4 = np.random.normal(10,3,(2,4))   #返回2行4列的array，且值为均值为10，标准差为3的正态分布的随机数

arr4.reshape(4,2)         ## 把2行4列转化为4行2列

arr5 = np.arange(8).reshape(2,4)

arr6 = np.arange(8)

arr6[0] = 1000

print(arr6)

arr7 = arr6.reshape(2,4)


'''2018.6.12'''

print('分片第二行：',arr7[1,2:3])    #返回第二行，第三列到第四列的值
print('所有行，第三列：',arr7[:,2])  #返回第三列的所有值
print('第一行：',arr7[0])           #返回第一行
print('第二行：',arr7[1])           #返回第二行

'''Array 的属性和方法'''

print('Data type                :',arr7.dtype)
print('Total number of elements :',arr7.size)
print('Number of dimensions     :',arr7.ndim)
print('Shape (Dimensionality)   :',arr7.shape)
print('Memory used(in bytes)    :',arr7.nbytes)

print('Minimum and maximum             :', arr.min(), arr.max())
print('Sum and product of all elements :', arr.sum(), arr.prod()) #prod相乘
print('Mean and standard deviation     :', arr.mean(), arr.std())

print(arr7)
print('沿着行求和:',arr7.sum(axis = 1))
print('沿着列求和:',arr7.sum(axis = 0))

print('原始array:\n',arr7)
print('转置array:\n',arr7.T)  #.T为转置


'''操作 arrays'''
a1 = np.arange(4)
a2 = np.arange(10,14)
print(a1, '+', a2, '=', a1 + a2)
print(a1, '*', a2, '=', a1 * a2)

10 * a1    #数乘

a1 + 5  #print(np.arange(4) + 5)

np.ones((3,3))

np.ones((3,3)) + np.arange(3)

np.arange(3).reshape(3,1) + np.ones(3)

np.arange(3).reshape(3,1) + np.arange(3)


'''numpy中的矩阵操作  np.dot()'''

v1 = np.array([2,3,4])
v2 = np.array([1,0,1])
v1 * v2

print(v1, '.', v2, '=', np.dot(v1,v2))

A = np.arange(6).reshape(2,3)

print(A, '.', v1, '=',np.dot(A,v1))

np.dot(A,v1.reshape(3,1))           #  np.dot 矩阵乘法

np.dot(A,A.T)

np.dot(A.T,A)



'''绘图与可视化matplotlib'''


import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


'''sep 是函数的形式参数，多数情况下， seq 参数用来指定字符的分隔符号。
不仅用在输出，也用在输入，也用在字符串的合并与拆分上。
csv 文件是用逗号分隔的，故而 sep = ","
tsv 文件是用制表符分隔的，故而 sep = "\t"
常用的分隔符还有空格 sep = " "   '''

anscombe = pd.read_csv('F:/clzl/wh/data/anscombe.tsv',sep = '\t')

anscombe.describe().loc[['mean','std'],:].round(2)

'''  .describe()是简单的描述统计，包括mean，std，min，max等，
    .loc意义：通过行标签索引行数据
    .round(2)表示保留两位小数  
    
loc[n]表示索引的是第n行（index 是整数）
loc[‘d’]表示索引的是第’d’行（index 是字符）
.iloc   ：通过行号获取行数据，不能是字符
ix——结合前两种的混合索引'''
    
x = np.linspace(-5,2,100)   #linspace函数可以生成元素为n的等间隔数列。而前两个参数分别是数列的开头与结尾。如果写入第三个参数，可以制定数列的元素个数
y1 = x**3 + 5*x**2 + 10
y2 = 3*x**2 + 10*x
y3 = 6*x + 10
plt.plot(x,y1)   
    
fig,ax = plt.subplots()    #figure 是画布, axis 是作图区域
ax.plot(x,y1) 
'''注意：如果fig和ax的定义语句，与画图的语句在两个cell中分别执行，
则不能画出具体图形，只能画出第一步。以上两行代码必须同时运行'''   
  
fig,ax = plt.subplots()
ax.plot(x,y1,color = 'blue',label = 'y(x)') 
ax.plot(x,y2,color = 'red',label = 'y’(x)')  
ax.plot(x,y3,color = 'green',label = 'y’’(x)') 
ax.set_xlabel('x')       #设置x轴名称
ax.set_ylabel('y')       #设置y轴名称
ax.legend()              #加图例
    
x = np.linspace(0,10,1000)  
y = np.sin(x)  
z = np.cos(x**2) 


  
plt.figure(figsize = (8,4))     #用figsize定义图的大小
plt.plot(x,y,label = '$sin(x)$',color = 'red',linewidth = 2)   
#linewidth = 2 : 指定曲线的宽度 ，等价于lw = 2
plt.plot(x,z,'b-',label = '$cos(x^2)$')    #这里的 'b-'是设置线条颜色，相当于color = 'blue'
#label : 给所绘制的曲线一个名字，此名字在图示(legend)中显示。
#只要在字符串前后添加"$"符号，matplotlib就会使用其内嵌的latex引擎绘制的数学公式。
plt.xlabel('Time(s)')   
plt.ylabel('Volt')
plt.title('A Pyplot Example')  
plt.ylim(-1.2,1.2)    # ylim : 设置Y轴的范围
plt.legend()
plt.show()    #plt.show()显示出我们创建的所有绘图对象
    

fig, ax = plt.subplots(figsize=(8,5))     

    

   
'''  2018.6.13   基本图形类型:
线图line
条形图bar （表现离散数据）
饼图pie
直方图 histogram（画连续数据）
散点图 scatter （两个因素的相关性)'''  
    
    
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
      
x = np.linspace(-3,3,25)  
y1 = x**3 + 3*x**2 + 10
y2 = -1.5*x**3 + 10*x**2 - 15
plt.figure(figsize = (4,3)) 
plt.plot(x,y1)
plt.plot(x,y2)    
    
plt.step(x,y1)    # step是阶梯图
plt.step(x,y2)    
    
width = 6/50.0
plt.bar(x-width/2,y1,color = 'blue',width = width)    
plt.bar(x+width/2,y2,color = 'green',width = width)

plt.fill_between(x,y1,y2)     # y1和 y2之间的面积填起来

plt.hist(y1,bins = 30)      #bins: 直方图的柱数，可选项，默认为10
plt.hist(y2,bins = 30)


#误差图
plt.errorbar(x, y2, yerr=y1, fmt='o-')

plt.errorbar(x,y2,yerr = y1,fmt = 'ok-',ecolor='red',alpha=1)

plt.errorbar(x,y1,yerr = y2,fmt = 'o-',ecolor='green',alpha=0.8)

''' 参数 yerr 中输入数据的误差序列
  fmt='ok-'，表示的是(x, y)这一坐标位置的显示形式，实心（o）圆点，
  k表示颜色是黑色，- 表示连接这些实心圆点，如果不设置颜色，默认颜色是蓝色
  ecolor='red'，表示错误线的颜色,如果不设置颜色，默认是蓝色
  alpha  表示透明度 '''

#茎叶图
plt.stem(x,y1,'b',markerfmt = 'bs-')
plt.stem(x,y2,'r',markerfmt = 'ro-')
# markerfmt = 'bs-' 中 s 表示实心方形标记，o 表示实心圈标记 , - 表示将标记用线连接起来 
# 参考  https://blog.csdn.net/Zach_z/article/details/78611935


#绘制散点图
x = np.linspace(0,5,50)
plt.scatter(x,-1 + x + 0.25*x**2 + 2*np.random.randn(len(x)))
plt.scatter(x,np.sqrt(x) + 2*np.random.randn(len(x)),color = 'green')


#Series
s = pd.Series([1,2,3,4,5,6])
s.index = ['a','b','c','d','e','f']
s.name = 'Scores'

s = pd.Series([1,2,3,4,5,6],index = ['a','b','c','d','e','f'],name = 'Scores')

s['d']
s.d
s[['a','b']]


s.median(), s.mean(), s.std(),s.min(), s.max()
s.quantile(q=0.25), s.quantile(q=0.5), s.quantile(q=0.75)
s.describe()


mpl.style.use('ggplot')
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
s.plot(ax=axes[0], kind='line', title="line") #线图
s.plot(ax=axes[1], kind='bar', title="bar") #柱状图
s.plot(ax=axes[2], kind='box', title="box") #箱线图
s.plot(ax=axes[3], kind='pie', title="pie") #饼图


'''2018.6.20'''
import pandas as pd
import numpy as np
df = pd.DataFrame([[909976,8615146,2872086,2273305],
                   ["Sweden","United kingdom","Italy","France"]])
    
df = pd.DataFrame([[909976,"Sweden"],
                   [8615146,"United kingdom"],
                   [2872086,"Italy"],
                   [2273305,"France"]])

df.index = ["Stockholm","London","Rome","Paris"]
df.columns = ["Population","State"]

df = pd.DataFrame([[909976,"Sweden"],
                   [8615146,"United kingdom"],
                   [2872086,"Italy"],
                   [2273305,"France"]],
                  index = ["Stockholm","London","Rome","Paris"],
                  columns = ["Population","State"])
df = pd.DataFrame({"Population":[909976,8615146,2872086,2273305],
                      "State": ["Sweden","United kingdom","Italy","France"]},
                      index = ["Stockholm","London","Rome","Paris"])

df.index
df.columns
df.values
df.Population
df['Population']
type(df.Population)
df.Population.Stockholm
df.loc[:,'State']
df.loc[:,'Population']
df.loc['Stockholm']
df.loc["Rome","Population"]
df.loc[["Paris","Rome"]]
df.loc[["Paris","Rome"],"Population"]
df.mean()
df.info()       #信息统计


df_pop = pd.read_csv('F:/clzl/wh/data/european_cities.csv')
df_pop.head()
df_pop.info()             ##查看以后，发现没有缺失值
df_pop["NumericPopulation"] = df_pop.Population.apply(lambda x: int(x.replace(",","")))

''' 对每一行用 apply 操作, 去掉population每个元素的逗号并转为integer
lambda的一般形式是关键字lambda后面跟一个或多个参数，紧跟一个冒号，以后是一个表达式
作为表达式，lambda返回一个值（即一个新的函数）'''

df_pop['State'].values[:3]
df_pop['State'].values[3]
df_pop["State"] = df_pop["State"].apply(lambda x : x.strip())   ## 去掉 string 的前后的空格
df_pop2 = df_pop.set_index('City')    # # set_index 将某个作为 index
df_pop2 = df_pop2.sort_index()
df_pop2.head()

df_pop3 = df_pop.set_index(['State','City']).sortlevel(0)
# sortlevel(0)根据第一个索引state排序，sortlevel(1)根据第二个索引city排序
df_pop3.loc["Spain"]

df_pop3 = df_pop[["State","City","NumericPopulation"]].set_index(["State","City"])
# 取三个变量
df_pop3.head(7)
df_pop3.loc[('Austria','Vienna')]

df_pop4 = df_pop3.sum(level = 'State').sort_values('NumericPopulation',ascending = False)
df_pop4.head()

df_pop5 = (df_pop3.drop('Rank',axis = 1).groupby('State').sum().sort_values('NumericPopulation',ascending = False))
df_pop5.head()

# drop 是删掉这一行     ,   groupby 汇总
df_pop5 = (df_pop3.drop('Rank',axis = 1).sum(level = 'State').sort_values('NumericPopulation',ascending = False))
#ascending = Falese  降序排列    ascending = True 升序排列
city_counts = df_pop.State.value_counts()

import matplotlib as mpl
import matplotlib.pyplot as plt

fig,(ax1,ax2) = plt.subplots(1,2,figsize = (12,4))   # 画布设置，ax1和ax2表示设置两个画布1,2
city_counts.plot(kind = 'barh',ax = ax1)   #  ax = ax1 表示图画在位置一上，barh 表示横着的 bar
ax1.set_xlabel('# cities in top 105')
df_pop5.NumericPopulation.plot(kind = 'barh',ax = ax2)
ax2.set_xlabel('Total pop. in top 105 citie')
fig.savefig("C:/Users/Administrator/Desktop/state-city-counts-sum.jpg")


import datetime
pd.date_range(datetime.datetime(2018,1,1),periods = 31)    # 用 datetime 函数创建

pd.date_range('2018/1/1',periods = 31)
pd.date_range('2018-1-1',periods = 31)

pd.date_range('2018-1-1 00:00','2018-1-1 12:00',freq = 'H')  #按小时

ts1 = pd.Series(np.arange(31),index = pd.date_range('2018-1-1',periods = 31))
ts1 = pd.Series(np.arange(31),index = pd.period_range('2018-1-1','2018-1-31',freq = 'D'))
ts1
ts1['2018-1-3']
ts1.index[2]
ts1.index[2].year,  ts1.index[2].month,   ts1.index[2].day

df1 = pd.read_csv('F:/clzl/wh/data/temperature_outdoor_2014.tsv',sep="\t", names=["time", "outdoor"])
df2 = pd.read_csv('F:/clzl/wh/data/temperature_indoor_2014.tsv',sep="\t", names=["time", "indoor"])
df1.head()     # 没有自动辨认成时间
df2.head()

df1.time = (pd.to_datetime(df1.time.values, unit="s").tz_localize('UTC').tz_convert('Europe/Stockholm'))
df1 = df1.set_index('time')
df1.index[0]
df2.time = (pd.to_datetime(df2.time.values,unit = 's').tz_localize('UTC').tz_convert('Europe/Stockholm'))
df2 = df2.set_index('time')

fig,ax = plt.subplots(1,1,figsize = (12,4))
df1.plot(ax = ax)
df2.plot(ax = ax)

df1.info()

df1.index < '2014-2-1'
df1_jan = df1[(df1.index > '2014-1-1') & (df1.index < '2014-2-1')]
df2_jan = df2['2014-1-1' : '2014-1-31']

fig,ax = plt.subplots(1,1,figsize = (12,4))
df1_jan.plot(ax = ax)
df2_jan.plot(ax = ax)
fig.tight_layout()  #自动调整subplot间的参数

df1.to_period('M').head()
df_month = pd.concat([df.to_period('M').groupby(level = 0).mean() for df in [df1,df2]],axis = 1)
df_month.head()

f,axes = plt.subplots(1,2,figsize = (12,4))
df_month.plot(kind = 'bar',ax = axes[0])
df_month.plot(kind = 'box',ax = axes[1])
f.tight_layout()
 
  
'''    2018.6.27-6.28     '''

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


names1880 = pd.read_csv('E:/clzl/wh/data/babynames/yob1880.txt',names = ['name','sex','births'])
names1880.head()
names1880.groupby('sex').births.sum()

'''考虑到文件的形式，每年放在一个文件中，我们需要做一些处理（现实中很多数据是这样的形式）
我们将year作为一个特征，合并所有的文件'''

years = range(1880,2011)
pieces = []
columns = ['name','sex','births']

for year in years:
    path = 'E:/clzl/wh/data/babynames/yob%d.txt' % year
    frame = pd.read_csv(path,names = columns)
    frame['year'] = year
    pieces.append(frame)
    names = pd.concat(pieces,ignore_index = True)
names.head()

'''  Pandas 透视表（pivot_table）
aggfunc可以包含很多函数,对该列元素进行计数或求和。
要添加这些功能，使用aggfunc和np.sum就很容易实现
http://python.jobbole.com/81212/   '''

total_births = pd.pivot_table(names,values = 'births',index = 'year',columns = 'sex',aggfunc = sum)
total_births.tail()

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']    #用来正常显示中文标签
total_births.plot(title='每年的男女总出生数量')

def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births/births.sum()
    return group
names_group = names.groupby(['year','sex']).apply(add_prop)


        



















