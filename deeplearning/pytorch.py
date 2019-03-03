# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 18:48:07 2018

@author: Administrator
"""


import numpy as np
a = [1,2,3]
a[1]
b = [[1,1,1],a,[3,4,5],[0,0,0]]

e = np.asmatrix(b)
c = np.array(b)


aa = np.ones((3,))
bb = np.zeros((3,))
ab = np.dot(aa,bb)
ccc = np.arange(2)

arr1 = np.array([[1,2,3]])
arr2 = np.array([1,2,3])
asc = np.dot(arr1,arr2)
avxd = np.empty((2,3,4))

on = np.eye(3)
np.identity(3)

v = np.arange(5)
vvv = np.array((3,6,5))
vv = np.array(np.arange(5))

arr = np.array([1,2,3,4,5])
arr.dtype
float_arr = arr.astype(np.float64)

a = np.ones(3)
aa = np.zeros((1,3))

aaa = a + aa

np.concatenate

a = np.arange(6).reshape(2,3)
b = np.argmax(a)
c = np.argmax(a, axis=0)

d = np.array([[2,2,2,9],[1,8,0,3],[4,4,5,5]])
np.argmax(d,axis=0)
np.argmax(d,axis=1)


a=np.array([[1,2,3],[4,5,6]])
b=np.array([[11,21,31],[7,8,9]])
d1 = np.concatenate((a,b),axis=0)
d2 = np.concatenate((a,b),axis=1)  




import torch
# 生成均匀间隔的点，输出为张量
d = torch.linspace(3,10,5)
print(d)


#以对数刻度均匀间隔的点
e = torch.logspace(-10,10,5)
print(e)

# 生成全是0或1的张量
torch.ones(2,3)
torch.ones(5)
torch.zeros(2,3)
torch.zeros(5)
#生成均匀分布的张量
torch.rand(4)
torch.rand(2,2)

# 生成服从正态分布的张量
torch.randn(5)
torch.randn(2,3)

#给定参数n，返回一个从0 到n-1 的随机整数排列
torch.randperm(6)

# 返回一组序列值
torch.arange(5)
torch.arange(0,5)
torch.arange(1,5)
torch.arange(0,6,2)

# torch.cat张量拼接
x = torch.randn(2,3)
torch.cat((x,x),0)    # 按列拼接
torch.cat((x,x,x),1)  # 按行拼接

y = torch.randn(2,2)
torch.cat((x,y),1)

z = torch.randn(1,3)
torch.cat((x,z),0)

# torch.chunk在给定维度(轴)上将输入张量进行分块儿
x = torch.randn(3,3)
torch.chunk(x,2,0)   #按行切
torch.chunk(x,2,1)   #按列切

# torch.gather沿给定轴dim，将输入索引张量index指定位置的值进行聚合
t = torch.Tensor([[1,2],[3,4]])
torch.gather(t,1,torch.LongTensor([[0,0],[1,0]]))

# torch 和 numpy转化
import numpy as np
np_data = np.array([[1,2,3],[4,5,6]])
torch_data = torch.from_numpy(np_data)  # numpy 转为 pytorch格式
print(torch_data)

torch2array = torch_data.numpy()        #  torch 转为numpy

# 转置torch.t
x = torch.Tensor([[1,2],[4,45]])
torch.t(x)


















