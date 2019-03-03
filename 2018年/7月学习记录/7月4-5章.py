# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:08:35 2018

@author: Administrator
"""

 '''规范化'''
import pandas as pd
import numpy as np
datafile = 'C:/Users/Administrator/Desktop/chapter4/demo/data/normalization_data.xls'
data = pd.read_excel(datafile,header = None)
(data - data.min())/(data.max() - data.min())   # 最大-最小规范化
(data - data.mean())/data.std()                 # 零-均值规范化
data/10**np.ceil(np.log10(data.abs().max()))    # 小数定标规范化


'''离散化'''
import pandas as pd
import matplotlib.pyplot as plt
datafile = 'C:/Users/Administrator/Desktop/chapter4/demo/data/discretization_data.xls'
data = pd.read_excel(datafile)
data = data['肝气郁结证型系数'].copy()
k = 4
d1 = pd.cut(data,k,labels = range(k))  #等宽离散化，各个类别依次命名为0，1，2，3
...


'''属性构造'''
import pandas as pd
inputfile = 'C:/Users/Administrator/Desktop/chapter4/demo/data/electricity_data.xls'
outputfile = 'C:/Users/Administrator/Desktop/Electricity_data.xls'
data = pd.read_excel(inputfile)
data['折损率'] = (data['供入电量'] - data['供出电量'])/data['供入电量']
data.to_excel(outputfile,index = False)


''' 主成分分析降维 '''
import pandas as pd
inputfile = 'C:/Users/Administrator/Desktop/chapter4/demo/data/principal_component.xls'
outputfile = 'C:/Users/Administrator/Desktop/dimention_reducted.xls'
data = pd.read_excel(inputfile,header = None)
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(data)    #  PCA.fit(data) 
pca.components_
pca.explained_variance_ratio_

pca = PCA(3)
pca.fit(data)
low_d = pca.transform(data)
pd.DataFrame(low_d).to_excel(outputfile)
pca.inverse_transform(low_d)


# unique 去除数据中重复值
import pandas as pd
import numpy as np
d = pd.Series([1,1,2,3,3,4,5])
d.unique()
np.unique(d)


z = np.random.rand(10,4)
z.sum(axis = 0)   #对列求和
z.sum(axis = 1)   #对行求和
from sklearn.decomposition import PCA
#pca = PCA()
pca.fit(z)
pca.components_
pca.explained_variance_ratio_


'''第五章'''

'''Logistic回归'''
import pandas as pd
filename = 'C:/Users/Administrator/Desktop/chapter5/demo/data/bankloan.xls'
data = pd.read_excel(filename)
x = data.iloc[:,:8].as_matrix()
y = data.iloc[:,8].as_matrix()
x.shape
y.shape

from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
rlr = RLR()
rlr.fit(x,y)
rlr.get_support()
#rlr.scores_   特征分数
print('通过随机逻辑回归模型筛选特征结束。')
print('有效特征为：%s' % ','.join(data.columns[rlr.get_support()]))
x = data[data.columns[rlr.get_support()]].as_matrix()
lr = LR()
lr.fit(x,y)
print('逻辑回归模型训练结束')
print('模型的平均正确率为：%s' % lr.score(x,y))


'''K-Means聚类'''

import pandas as pd
#参数初始化
inputfile = 'C:/Users/Administrator/Desktop/chapter5/demo/data/consumption_data.xls' #销量及其他属性数据
outputfile = 'C:/Users/Administrator/Desktop/data1_type.xls' #保存结果的文件名
k = 3 #聚类的类别
iteration = 500 #聚类最大循环次数
data = pd.read_excel(inputfile, index_col = 'Id') #读取数据
data_zs = 1.0*(data - data.mean())/data.std() #数据标准化

from sklearn.cluster import KMeans
model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration) #分为k类，并发数4
model.fit(data_zs) #开始聚类

#简单打印结果
r1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
r2 = pd.DataFrame(model.cluster_centers_) #找出聚类中心
r = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
r.columns = list(data.columns) + [u'类别数目'] #重命名表头
print(r)

#详细输出原始数据及其类别
r = pd.concat([data, pd.Series(model.labels_, index = data.index)], axis = 1)  #详细输出每个样本对应的类别
r.columns = list(data.columns) + [u'聚类类别'] #重命名表头
r.to_excel(outputfile) #保存结果


def density_plot(data): #自定义作图函数
  import matplotlib.pyplot as plt
  plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
  plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
  p = data.plot(kind='kde', linewidth = 2, subplots = True, sharex = False)
  [p[i].set_ylabel(u'密度') for i in range(k)]
  plt.legend()
  return plt

pic_output = '../tmp/pd_' #概率密度图文件名前缀
for i in range(k):
  density_plot(data[r[u'聚类类别']==i]).savefig(u'%s%s.png' %(pic_output, i))


from sklearn.manifold import TSNE

tsne = TSNE()
tsne.fit_transform(data_zs) #进行数据降维
tsne = pd.DataFrame(tsne.embedding_, index = data_zs.index) #转换数据格式

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

#不同类别用不同颜色和样式绘图
d = tsne[r[u'聚类类别'] == 0]
plt.plot(d[0], d[1], 'r.')
d = tsne[r[u'聚类类别'] == 1]
plt.plot(d[0], d[1], 'go')
d = tsne[r[u'聚类类别'] == 2]
plt.plot(d[0], d[1], 'b*')
plt.show()

















































