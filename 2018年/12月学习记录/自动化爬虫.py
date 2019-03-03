# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:00:23 2018

@author: Administrator
"""


'''自动化爬虫'''

import requests
import pandas as pd
from bs4 import BeautifulSoup
from lxml import etree
from lxml.html import fromstring, tostring
url = 'https://list.jd.com/list.html?cat=670,671,672&page=1&sort=sort_totalsales15_desc&trans=1&JL=6_0_0#J_main'


def get_url(url):
#    html = requests.get(url)
    html = requests.get(url).text
    res=etree.HTML(html)
    price=res.xpath('//strong[@class="J_price"]/i')
    tostring(price[0])
    html.encoding = 'utf-8'
    soup = BeautifulSoup(html.text,'html.parser')
    li = soup.find_all('li',attrs = {'class':'gl-item'})
    names = [l.find('div',{'class':'p-name'}).text.replace(' ','') for l in li]
    price = [l.find('strong',{'class':'J_price'}).text for l in li]
    pingjia = [l.find('div',{'class':'p-commit p-commit-n'}).text for l in li]
#    all_pictures = [l.find('div',{'class':'p-img'}).find('img').get('src') for l in li]
  
    data = pd.DataFrame({'标题':names,'价格':price,'评价数':pingjia},
                        columns = ['标题','价格','评价数'],index = None)
#    js="var q=document.getElementById('id').scrollTop=10000"
#    driver.execute_script(js)
#    driver.find_element_by_xpath('//*[@id="J_bottomPage"]/span[1]/a[10]').click()
    return data

d = pd.DataFrame({})

urls = 'https://list.jd.com/list.html?cat=670,671,672&page={}&sort=sort_totalsales15_desc&trans=1&JL=6_0_0#J_main'
for i in range(1,51):
    url = urls.format(i)
    inf = get_url(url)
    datas = pd.concat([d,inf],ignore_index=True)
    
#def get_picture(p):           
#    response = requests.get(p).content
#    name = p.split('/')[-1]
#    f = open(name, 'wb')
#    f.write(response)
#    f.close()
#for picture in all_pictures:        
#    p = 'http:' + picture     

path = r'C:\Users\Administrator\Desktop\58.csv'
datas.to_csv(path,index = None,encoding = 'gb18030')


L1 = ['Hello','World',18,'Apple',None]
L2 = [s.lower() for s in L1 if isinstance(s,str)==True]
print(L2)


L1 = ['Hello','World',18,'Apple',None]
L2 = []
for s in L1:
    if isinstance(s,str)==True:
        L2.append(s.lower())
print(L2)
#    else:
#        print('不是str')
        
for i in range(1,10):
     for j in range(1,i+1):
        print("%d*%d=%2d" % (i,j,i*j),end=" ")
     print (" ")      























