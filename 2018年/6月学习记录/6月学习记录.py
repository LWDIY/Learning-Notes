# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 21:15:26 2018

@author: Administrator
"""


'''  2018.6.4  处理缺失值  '''
#  在pandas中，使用np.nan来代替缺失值，这些值将默认不会包含在计算中
#  fillna	用指定值或插值方法（如ffill和bfill）填充缺失数据
import numpy as np
import pandas as pd
from numpy import nan as NA
from pandas import DataFrame, Series
df = Series([1,2,NA,3,NA])
                    
df
df.dropna()      #  DataFrame中dropna默认丢弃任何含有缺失值的行
df.dropna(how = 'all')  #  传入how=’all’来删除行全部为NaN的行
df.dropna(how = 'all',axis = 1)    #  删除所有包含空值的列，可以传入axis=1
df.fillna(0)     #  df.fillna(x)：用x替换DataFrame对象中所有的空值
df1 = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'h'],
                     columns=['one', 'two', 'three'])
df1['four'] = 'bar'
df1['five'] = df['one']>0
df1['six'] = 'NaN'     # 此处添加的一列  NaN 并不是空值，而是值的名称，所以下面的处理不了
df1
df2= df1.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
#   reindex会根据新索引进行重新排序，没有的索引会使用NaN来提点
df2
df2.dropna()      
df2.dropna(how = 'all')  
df2.dropna(how = 'all',axis = 1)
df2.fillna(0)




2018.6.5
import re
text = "JGood is a handsome boy, he is cool, clever, and so on..."
re.findall(r'\w*oo\w*', text)    #查找所有包含'oo'的单词

import re 
text = 'JGood is a handsome boy, he is cool, clever, and so on…'
regex = re.compile(r’\w*oo\w*’) 
print regex.findall(text) #查找所有包含’oo’的单词

import re
s = "adfad asdfasdf asdfas asdfawef asd adsfas " 
re.findall('\w+\s+\w',s)
reObj3 = re.compile('\w*s\w*') 
reObj3.findall(s)





"""2018.6.6
我们可以利用urlopen()方法可以实现最基本请求的发起，但这几个简单的参数并不足以 
构建一个完整的请求，如果请求中需要加入headers（请求头）等信息，我们就可以利用 
更强大的Request类来构建一个请求。也许这就是一下两个方式的区别吧"""

import urllib
url = 'https://www.baidu.com/'
response = urllib.request.urlopen(url)

header = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.89 Safari/537.36'
#  网页获取header的用户代理可以在地址栏输入about:version
req = urllib.request.Request(url =url,headers = header)
response = urllib.request.urlopen(req)   #用Request类构建了一个完整的请求，增加了headers等一些信息




"""爬取一个贴吧的图片"""

import requests
from bs4 import BeautifulSoup
import os

#在本地新建一个文件夹，命名为img，用以保存下载的图片
folder = 'imgs'
if not os.path.exists(folder):     # 检验给出的路径是否真地存:os.path.exists()
    os.makedirs(folder)            #创建多级目录：os.makedirs
    
#定义一个函数，用以下载图片  这一部分相当于把下面for循环的东西处理了
def download(url):
    response = requests.get(url)
    name = url.split('/')[-1]
    f = open(folder + '/' + name + '.jpg', 'wb')
    f.write(response.content)
    f.close()

#这里的网址比较简单，网页可以翻19次，相当于改变url，循环19次；然后对每一个页面都执行‘获取src’和‘下载图片’的操作    
for i in range(1,20):
    url_i = 'https://tieba.baidu.com/p/4064957036?pn=' + str(i)  
    response_i = requests.get(url_i).text
    
    #获取第i个页面的url、response类、html、soup，以及该页面所有图片对应的src
    soup_i = BeautifulSoup(response_i,'html.parser')       
    img_i = soup_i.find_all('img',{'class':"BDE_Image"})
    for img in img_i:
        img_src = img.get('src')
        print(img_src)
        download(img_src)   
#这里的download(img_src)  相当于把img_src 赋给了上面def download(url)中的url
#你也可以直接把上面的def download(url)写为def download(img_src)        
print('OK')


"""
你来试试爬取这个网址：http://www.netbian.com/index.htm
"""


import requests
from bs4 import BeautifulSoup
import os

#在本地新建一个文件夹，命名为img，用以保存下载的图片
folder = 'bizhi'
if not os.path.exists(folder):     # 检验给出的路径是否真地存:os.path.exists()
    os.makedirs(folder)            #创建多级目录：os.makedirs
    
#定义一个函数，用以下载图片  这一部分相当于把下面for循环的东西处理了
def download(url):
    response = requests.get(url)
    name = url.split('/')[-1]
    f = open(folder + '/' + name + '.jpg', 'wb')
    f.write(response.content)
    f.close()

#相当于改变url，循环然后对每一个页面都执行‘获取src’和‘下载图片’的操作    
for i in range(2,5):
    url_i = 'http://www.netbian.com/index_' + str(i) + '.htm' 
    response_i = requests.get(url_i).text
    
    #获取第i个页面的url、response类、html、soup，以及该页面所有图片对应的src
    soup_i = BeautifulSoup(response_i,'html.parser')       
    img_i = soup_i.find_all('img')
    for img in img_i:
        img_src = img.get('src')
        print(img_src)
        download(img_src)   
#这里的download(img_src)  相当于把img_src 赋给了上面def download(url)中的url
#你也可以直接把上面的def download(url)写为def download(img_src)        
print('OK')



"""2018.6.7  """

#画图的包
import matplotlib.pyplot as plt
import numpy as np








import pandas as pd
data_list = pd.DataFrame()
link_list = []
for i in range(1,150):
    url = 'http://bb.zhvac.com/AdminManage/CollectWeb/Search/' + str(i) +'?X-Requested-With=XMLHttpRequest'
    link_list.append(url)
for link in link_list:
    data_list = data_list.append(pd.read_html(link), ignore_index=True)
#pd.read_html能够读取带有table标签的网页中的表格。
data_list.to_csv('C:/Users/Administrator/Desktop/bb.csv',encoding = 'gbk')




import urllib
from bs4 import BeautifulSoup
import pandas as pd
url = 'http://bb.zhvac.com/AdminManage/CollectWeb/Search/1?X-Requested-With=XMLHttpRequest'
#def geturl(url):
rawhtml = urllib.request.urlopen(url).read().
soup = BeautifulSoup(rawhtml,'lxml')
tbody = soup.find_all('tbale')

for i in range(1,150):
    url_i = 'http://bb.zhvac.com/AdminManage/CollectWeb/Search/' + str(i) +'?X-Requested-With=XMLHttpRequest'
    response_i = requests.get(url_i).text






pip install selenium

import requests
from bs4 import BeautifulSoup
import re
import time
import os
#import time
from selenium import webdriver
import pandas as pd

driver=webdriver.Chrome() 
title=[]
zt=[]
local_main2='F:/train.csv'
if not os.path.exists(local_main2):
    data = pd.DataFrame(columns=['页数','名称','健康状态'],index=[0])
    data.to_csv(local_main2,index = None,encoding="utf_8_sig")
for j in range(1,150):
    url='http://bb.zhvac.com/AdminManage/CollectWeb/Search/'+str(j)+'?X-Requested-With=XMLHttpRequest'
    driver.get(url)
    a = driver.page_source
    soup = BeautifulSoup(a,'lxml')
    aaaa=soup.find_all('a',attrs={'data-target':'#myModal'})
    bb=[]
    for i in range(len(aaaa)):
        bb.append(aaaa[i].text)
    
    for o in range(1,20):
        try:
            title.append(bb[o*3])
            zt.append(bb[o*3+1])
            data1 = pd.DataFrame({'页数':j,'名称':bb[o*3],'健康状态':bb[o*3+1]},columns=['页数','名称','健康状态'],index=[0])
            data1.to_csv(local_main2,index = None,mode = 'a' ,header=  None,sep=',',encoding="utf_8_sig")
        except:
            break






























