# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:16:43 2018

@author: Administrator
"""


'''''爬取58同城租房信息'''
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
url = 'https://wh.58.com/chuzu/?PGTID=0d100000-0009-e796-2c7d-c840de5914dd&ClickID=2'

Type_of_housing = []
address = []
title = []
price = []
jjr = []

def get_url(url):
    time.sleep(2)
    rawhtml = requests.get(url)
    rawhtml.encoding = 'utf-8'
    soup = BeautifulSoup(rawhtml.text,'html.parser')
    li = soup.find_all('li',{'sortid':'1543680005000'})
  
    Type_of_housing = [l.find('p',{'class':'room strongbox'}).text.replace(' ','') for l in li]  
    address = [l.find('p',{'class':'add'}).text.replace(' ','') for l in li]
    title = [l.find('a',{'class':'strongbox'}).text.replace(' ','') for l in li]
    price = [l.find('div',{'class':'money'}).text for l in li]
    jjr = [l.find('span',{'class':'listjjr'}).text.replace(' ','') for l in li]
    
    data = pd.DataFrame({'标题': title, 
                         '租金':price, 
                         '经纪人':jjr,
                         '房屋类型':Type_of_housing, 
                         '地址': address}, 
                         columns = ['标题', '租金','经纪人','房屋类型','地址'],
                         index=None)
                      
    return data
    
house_data = pd.DataFrame({})
urli = 'https://wh.58.com/chuzu/pn{}/?PGTID=0d3090a7-0009-e99d-2872-212376ad474e&ClickID=2'
for i in range(1,50):
    if (i<2):
        url = 'https://wh.58.com/chuzu/?PGTID=0d3090a7-0009-e2a8-5e57-b941c556c01c&ClickID=2'
    else:
        url = urli.format(i)
        
    inf = get_url(url)
    house_data = pd.concat([house_data,inf],ignore_index=True)

path = r'C:\Users\Administrator\Desktop\58.csv'
house_data.to_csv(path,index = None,encoding = 'gb18030')


#1.strip()：把头和尾的空格去掉
#2.lstrip()：把左边的空格去掉
#3.rstrip()：把右边的空格去掉
#4.replace('c1','c2')：把字符串里的c1替换成c2。故可以用replace(' ','')来去掉字符串里的所有空格
#5.split()：通过指定分隔符对字符串进行切片，如果参数num 有指定值，则仅分隔 num 个子字符串  



###############################################################################################################

import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
import socket
timeout = 50
true_socket = socket.socket
socket.setdefaulttimeout(30) 

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}

Lease_method = [] #租赁方式
Type_of_housing = [] #房屋类型
Towards_the_floor = []#朝向楼层
In_the_community = []#所在小区
Region = []#所属区域
address = []#地址
title = []
price = []
name = []
phone = []
hrefs = []
pay_method = []


#获取详情页链接
def get_links(url):
    web_data_0 = requests.get(url,headers=headers)
    soup_0 = BeautifulSoup(web_data_0.text, 'lxml')
    links = soup_0.select('.des  h2  a')
    for link in links:
        href = 'https:' + link.get("href")
        hrefs.append(href)
    return (hrefs)


#获取详情页信息
def get_info(href):
    time.sleep(10)
    if(href is None):
        return
    else:                        
        web_data_1 = requests.get(href, headers=headers)
        soup_1 = BeautifulSoup(web_data_1.text, 'lxml')
        title = soup_1.select('body > div.main-wrap > div.house-title > h1')[0].get_text().strip()
        price = soup_1.select(
        'body > div.main-wrap > div.house-basic-info > div.house-basic-right.fr > div.house-basic-desc > div.house-desc-item.fl.c_333 > div > span.c_ff552e > b')[0].get_text().strip()
        pay_method = soup_1.select(
        'body > div.main-wrap > div.house-basic-info > div.house-basic-right.fr > div.house-basic-desc > div.house-desc-item.fl.c_333 > div > span.c_333')[0].get_text().strip()
        name = soup_1.select(
        'body > div.main-wrap > div.house-basic-info > div.house-basic-right.fr > div.house-basic-desc > div.house-agent-info.fr > p.agent-name.f16.pr')[0].get_text().strip()
        phone = soup_1.select(
        'body > div.main-wrap > div.house-basic-info > div.house-basic-right.fr > div.house-fraud-tip > div.house-chat-phone')[0].get_text().strip()
        Lease_method = soup_1.select(
        'body > div.main-wrap > div.house-basic-info > div.house-basic-right.fr > div.house-basic-desc > div.house-desc-item.fl.c_333 > ul > li:nth-of-type(1) > span:nth-of-type(2)')[0].get_text().strip()
        Type_of_housing = soup_1.select(
        'body > div.main-wrap > div.house-basic-info > div.house-basic-right.fr > div.house-basic-desc > div.house-desc-item.fl.c_333 > ul > li:nth-of-type(2) > span:nth-of-type(2)')[0].get_text().strip()
        Towards_the_floor = soup_1.select(
        'body > div.main-wrap > div.house-basic-info > div.house-basic-right.fr > div.house-basic-desc > div.house-desc-item.fl.c_333 > ul > li:nth-of-type(3) > span:nth-of-type(2)')[0].get_text().strip()
        In_the_community = soup_1.select(
        'body > div.main-wrap > div.house-basic-info > div.house-basic-right.fr > div.house-basic-desc > div.house-desc-item.fl.c_333 > ul > li:nth-of-type(4) > span:nth-of-type(2)')[0].get_text().strip()
        Region = soup_1.select(
        'body > div.main-wrap > div.house-basic-info > div.house-basic-right.fr > div.house-basic-desc > div.house-desc-item.fl.c_333 > ul > li:nth-of-type(5) > span:nth-of-type(2)')[0].get_text().strip()
        address = soup_1.select(
        'body > div.main-wrap > div.house-basic-info > div.house-basic-right.fr > div.house-basic-desc > div.house-desc-item.fl.c_333 > ul > li:nth-of-type(6) > span:nth-of-type(2)')[0].get_text().strip()
        return(title,price,pay_method,name,phone,Lease_method,Type_of_housing,Towards_the_floor,In_the_community,Region,address)
    

urli = 'https://wh.58.com/chuzu/pn{}/?utm_source=market&spm=b-31580022738699-pe-f-830.2345_101&PGTID=0d3090a7-0009-eb74-a8d2-b2ab53f02aed&ClickID=2'


for i in range(1,50):
    if (i<2):
        url = 'https://wh.58.com/chuzu/?utm_source=market&spm=b-31580022738699-pe-f-830.2345_101&PGTID=0d3090a7-0009-ef57-7292-68f909472421&ClickID=2'
    else:
        url = urli.format(i)
    hrefs = get_links(url)
    
    for href in hrefs:
        time.sleep(5)
        x = get_info(href)
        title.append(x[0])
        price.append(x[1])
        pay_method.append(x[2])
        name.append(x[3])
        phone.append(x[4])
        Lease_method.append(x[5])
        Type_of_housing.append(x[6])
        Towards_the_floor.append(x[7])
        In_the_community.append(x[8])
        Region.append(x[9])
        address.append(x[10])

path = r'C:\Users\Administrator\Desktop\58.csv'
#data1 = pd.DataFrame(columns=['标题','租金','支付方式', '经纪人', '联系电话','租赁方式', '房屋类型', '朝向楼层', '所在小区', '所属区域', '地址'])
#data1.to_csv(path, index = None,encoding = 'gbk')
excel = pd.DataFrame({'标题': title, '租金':price, '支付方式':pay_method, 
                      '经纪人':name, '联系电话':phone, '租赁方式':Lease_method, 
                      '房屋类型':Type_of_housing, '朝向楼层':Towards_the_floor, 
                      '所在小区':In_the_community, '所属区域':Region, '地址': address}, 
                      columns = ['标题', '租金','支付方式', '经纪人', '联系电话','租赁方式', '房屋类型', '朝向楼层', '所在小区', '所属区域', '地址'], index=None)
excel.to_csv(path,index = None,encoding = 'gb18030')



#####################################################################################################################

import time
import socket
import pandas as pd
import requests
from bs4 import BeautifulSoup


timeout = 50
true_socket = socket.socket
socket.setdefaulttimeout(30)  

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}

urli = 'http://wh.58.com/zpcaiwushenji/pn{}/?PGTID=0d303653-0009-edbc-b686-2823bf185240&ClickID=3'
url = 'https://wh.58.com/zpcaiwushenji/?PGTID=0d303653-0009-e912-00cb-ce55c12be39c&ClickID=3'

links = []  #每一个页面的链接
jobs = []    #职位名称
comnas = []  #公司名称
places = []  #地点
salary = []  #薪水
fbtime = []  #发布时间
xueli = []  #学历
jingy = []  #经验
jobnum = []  #招聘人数
jobdes = []  #职位描述

i = 1
while i <= 10:
    time.sleep(1)
    request_1 = requests.get(urli.format(i))
    request_1.encoding = 'utf-8'
    soup_1 = BeautifulSoup(request_1.text, 'html.parser')
    line_1 = soup_1.findAll('li', attrs = {'class':'job_item clearfix'})
    li_1 = [l.find('a').get('href') for l in line_1]
    place_1 = [l.find('span', {'class':'address'}).text for l in line_1]
    salary_1 = [l.find('p', {'class':'job_salary'}).text for l in line_1]
    comna_1 = [l.find('div', {'class':'comp_name'}).find('a').get('title') for l in line_1]
    job_1 = [l.find('span', {'class':'cate'}).text for l in line_1]
    xueli_1 = [l.find('span', {'class':'xueli'}).text for l in line_1]
    jingy_1 = [l.find('span', {'class':'jingyan'}).text for l in line_1]
    
    for k in range(len(li_1)):
        time.sleep(1)
        request_2 = requests.get(li_1[k])
        request_2.encoding = 'utf-8'
        soup_2 = BeautifulSoup(request_2.text, 'html.parser')
        fbtime_1 = soup_2.find('span', {'class':'pos_base_num pos_base_update'}).text
        fbtime.append(fbtime_1)
        jobnum_1 = soup_2.find('span', {'class':'item_condition pad_left_none'}).text
        jobnum.append(jobnum_1)
        jobdes_1 = soup_2.find('div', {'class':'des'}).text
        jobdes.append(jobdes_1)
        links.append(li_1[k])
        places.append(place_1[k])
        salary.append(salary_1[k])
        comnas.append(comna_1[k])
        jobs.append(job_1[k])
        xueli.append(xueli_1[k])
        jingy.append(jingy_1[k])
        request_2.close()
        
    request_1.close()
    print('第{}页'.format(i))
    i += 1
    
path = r'C:\Users\Administrator\Desktop\588.csv'
excel = pd.DataFrame({'链接':links, '职位名称':jobs, '公司名称':comnas, '任职地点':places, '月薪':salary, '最低学历': xueli, '经验限制': jingy, '招聘人数':jobnum, '职位描述':jobdes, '发布时间':fbtime}, columns = ['链接', '职位名称', '公司名称', '任职地点', '月薪', '最低学历', '经验限制', '招聘人数', '职位描述', '发布时间'], index=None)
excel.to_csv(path,index = None ,mode = 'a' ,header=  None,sep=',',encoding = 'gb18030')








