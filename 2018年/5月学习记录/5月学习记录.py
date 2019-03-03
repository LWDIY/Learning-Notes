# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:44:13 2018

@author: Administrator
"""

2018.5.17   爬电影
import requests
import pandas as pd
from bs4 import BeautifulSoup
url='http://www.cbooo.cn/year?year=2018'
#用urlopen，来提取网页内容，效果一样
#from urllib.request import urlopen
#from bs4 import BeautifulSoup
#html = urlopen('http://www.cbooo.cn/year?year=2018')
#效果等价于requests.get()
#soup = BeautifulSoup(html,'lxml')
#movies_table = soup.find_all('table',{'id':"tbContent"})[0]
def getMovies(url):
    rawhtml = requests.get(url).content
    soup = BeautifulSoup(rawhtml,'lxml')
    movies_table = soup.find_all('table',{'id':"tbContent"})[0]
    movies = movies_table.find_all('tr')
    urls = [tr.find_all('td')[0].a.get('href') for tr in movies[1:]] 
    names = [tr.find_all('td')[0].a.get('title') for tr in movies[1:]]
    types = [tr.find_all('td')[1].string for tr in movies[1:]]
    box_offices = [int(tr.find_all('td')[2].string) for tr in movies[1:]]
    country = [tr.find_all('td')[5].string for tr in movies[1:]]
    time = [tr.find_all('td')[6].string for tr in movies[1:]]
    ms = pd.DataFrame({
        'names': names,
        'types': types,
        'box_offices': box_offices,
        'country': country,
        'time': time,
        'urls': urls})
    return ms 
Movies_data = pd.DataFrame({})
for i in range(8,19):
    if i<=9:
        url = 'http://www.cbooo.cn/year?year=200' + str(i)
    else :
        url = 'http://www.cbooo.cn/year?year=20' + str(i)
    data = getMovies(url) 
    Movies_data = pd.concat([Movies_data,data],ignore_index=True)
Movies_data
Movies_data.to_csv('C:/Users/Administrator/Desktop/movies.csv',encoding = 'gbk')


2018.5.18   爬基金
import requests
import pandas as pd
from bs4 import BeautifulSoup
url = 'http://fundact.eastmoney.com/banner/gp.html?from=groupmessage&isappinstalled=0'
rawhtml = requests.get(url).content
soup = BeautifulSoup(rawhtml,'lxml')
jijin_table = soup.find_all('table',{'class':"mainTb"})[0]
jijin_table
jijin = jijin_table.find_all('tr')
jijin
daima = [tr.find_all('td')[0].string for tr in jijin[1:]]
daima
name = [tr.find_all('td')[1].string for tr in jijin[1:]]
name
urls = [tr.find_all('td')[1].a.get('href') for tr in jijin[1:]]
urls
jingzhi = [tr.find_all('td')[2].text.split('05-')[0] for tr in jijin[1:]]
#加text就可以把全部的标签去掉
jingzhi
#v = [int(tr.find_all('td')[2].span.string) for tr in jijin[1:]]使用int会报错的
#
#提取净值另外做法
#jingzhi = [tr.find_all('td')[2].text[0:6] for tr in jijin[1:]]
#jingzhi
#jingzhi=[tr.find_all('td')[2].span.string for tr in jijin[1:]]
#从 span 标签提取
#jingzhi
rateD = [tr.find_all('td')[3].string for tr in jijin[1:]]
rateD
rateW = [tr.find_all('td')[4].string for tr in jijin[1:]]
rateW
fee = [tr.find_all('td')[13].a.string for tr in jijin[1:]]
fee

data = pd.DataFrame({
        'daima':daima,
        'names':name,
        'urls':urls,
        'jingzhi':jingzhi,
        'rateD':rateD,
        'rateW':rateW,
        'fee':fee})
data  
data.to_csv('C:/Users/Administrator/Desktop/jijin.csv',encoding = 'gbk')



2018.5.19学习
from urllib.request import urlopen
from bs4 import BeautifulSoup
html = urlopen("http://www.pythonscraping.com/pages/warandpeace.html")
bsObj = BeautifulSoup(html,'lxml')
namelist = bsObj.find_all('div',{'id':"text"})


import requests
import pandas as pd
from bs4 import BeautifulSoup
url = "http://www.pythonscraping.com/pages/warandpeace.html"
rawhtml = requests.get(url).content
soup = BeautifulSoup(rawhtml,'lxml')
namelist = soup.find_all('span',{"class":"green"})
for name in namelist:
    print(name.get_text())
#.get_text() 会把你正在处理的 HTML 文档中所有的标签都清除，然后返回
#一个只包含文字的字符串。假如你正在处理一个包含许多超链接、段落和标
#签的大段源代码，那么 .get_text() 会把这些超链接、段落和标签都清除掉，
#只剩下一串不带标签的文字。



2018.5.26
from urllib.request import urlopen
from bs4 import BeautifulSoup
html = urlopen('http://en.wikipedia.org/wiki/Kevin_Bacon')
baike = BeautifulSoup(html,'lxml')
url = baike.findAll('a')
url
for link in baike.findAll('a'):
    if 'href' in link.attrs:
        print(link.attrs['href'])      #注意 .attrs的使用方法



政府工作报告   2018.5.28
from urllib.request import urlopen
from bs4 import BeautifulSoup
url = 'http://www.hprc.org.cn/wxzl/wxysl/lczf/'
rawhtml = urlopen(url).read().decode('gb18030')
#要加后缀gb18030
soup = BeautifulSoup(rawhtml,'lxml')
namelist = soup.find_all('td',{'width':"82"})
nameslist = []
for names in namelist:
    nameslist.append(names.get_text())  
#将nameslist设置成空的，然后将names填进去
nameslist
# nameslist.append(names.string)
# print(names.get_text()) 效果一样的做法   print(names.string)
timelist = soup.find_all('td',{'width':"168"})  
timeslist = []
for time in timelist:
    timeslist.append(time.string)
timeslist


linkslist = soup.select('td.bl a')
linkslist = soup.find_all('td',{'class':"bl"})[0].find_all('a')
#select此时比find_all好用
linkslist[0]['href']
link = linkslist[0]['href'].split('./')[1]  
#如果不加[1]你可以试试结果
#学会用.split分割
#all_link = [i['href'].split('./')[1] for i in linkslist]

urls = [url + i['href'].split('./')[1] for i in linkslist]

import pandas as pd
data = pd.DataFrame({"names":nameslist,
                     "times":timeslist,
                     "urls":urls})
data.to_csv('C:/Users/Administrator/Desktop/zhengfubaogao.csv',encoding = 'gbk')





2018.5.29  爬取NBA球员年薪,带有table标签的网页中的表格
import pandas as pd
data = pd.DataFrame()
url_list = []
for i in range(1, 14):
    url = 'http://www.espn.com/nba/salaries/_/page/'+str(i)+'/seasontype/4' 
    url_list.append(url)
for url in url_list:
    data = data.append(pd.read_html(url), ignore_index=True)
#pd.read_html能够读取带有table标签的网页中的表格。
data = data[[x.startswith('$') for x in data[3]]]
#.startswith('$')是指以$开头的
data.to_csv('C:/Users/Administrator/Desktop/NAB_salary.csv',header=['RK','NAME','TEAM','SALARY'], index=False)

#源代码
#import pandas as pd
#
#data = pd.DataFrame()
#url_list = ['http://www.espn.com/nba/salaries/_/seasontype/4']
#for i in range(2, 13):
#    url = 'http://www.espn.com/nba/salaries/_/page/%s/seasontype/4' % i
#    url_list.append(url)
#for url in url_list:
#    data = data.append(pd.read_html(url), ignore_index=True)
#data = data[[x.startswith('$') for x in data[3]]]
#data.to_csv('NAB_salaries.csv',header=['RK','NAME','TEAM','SALARY'], index=False)


不行？？？
#import requests
#from bs4 import BeautifulSoup
#import pandas as pd
#url = 'http://www.espn.com/nba/salaries/_/seasontype/4'
#rawhtml = requests.get(url).content
#soup = BeautifulSoup(rawhtml,'lxml')
#salary = soup.find_all('table',{'class':"tablehead"})[0].find_all('tr')
#RK = [tr.find_all('td')[0] for tr in salary[1:]]
#Rk

用这种方法爬取电影
import pandas as pd
data_list = pd.DataFrame()
link_list = []
for i in range(8,19):
    if i <=9:
        url = 'http://www.cbooo.cn/year?year=200' + str(i)
    if i >9:
        url = 'http://www.cbooo.cn/year?year=20' + str(i)
    link_list.append(url)
for link in link_list:
    data_list = data_list.append(pd.read_html(link), ignore_index=True)
#pd.read_html能够读取带有table标签的网页中的表格。
data_list.to_csv('C:/Users/Administrator/Desktop/allmovies.csv',encoding = 'gbk')



2018.5.30
#pandas小模块学习 
import numpy as np
import pandas as pd
np.random.seed(1)
df = pd.DataFrame(np.random.randn(4, 4) * 4 + 3)     #生成随机数
df.T       # 对数据转置
df[:3]
df1 = pd.DataFrame(np.random.randn(4, 4) * 4 + 3,columns =['A','B','C','D'])
df1
df1[:3]
a = df.std()
df_norm = ((df - df.mean())/df.std())         # 归一化
df2 = pd.DataFrame({'A':['1','2','4','6'],
                   'B':['2','1','5','7'],
                   'C':np.random.randn(4),
                   'D':np.random.randn(4)})
df2['C']   #  选择一个单独的列，这将会返回一个Series，等同于df.A
df2.groupby('A').sum()












