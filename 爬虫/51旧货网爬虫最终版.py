# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:10:02 2019

@author: Administrator
"""


# 提取出前十个网站的所有公司信息网址
#%%
import requests
from bs4 import BeautifulSoup
import pandas as pd

# url = 'http://www.51jiuhuo.com/company/search.html?page=1'

def get_all_url(url):
     rawhtml = requests.get(url).content     
     soup = BeautifulSoup(rawhtml,'lxml')     
     urls_list = soup.find_all('span',{'class':'m undline'})    
     one_page_company_all_urls = [urls_list[i].a.get('href') for i in range(0,len(urls_list))]     
     return one_page_company_all_urls

company_url = []

for n in range(1,6):
    url = 'http://www.51jiuhuo.com/company/search.html?page=' + str(n)
    company_url = company_url + get_all_url(url)    # 列表相加
     


#  公司信息提取
#  one_url = 'http://www.51jiuhuo.com/companyShows/intro/491361.html'

def get_one_url(one_url):
     rawhtml = requests.get(one_url).content
     soup = BeautifulSoup(rawhtml,'lxml')
     
     try:
         company_name = soup.find_all('span',{'class':'info'})[0].get_text()
     except:
         company_name = 'NULL'
     try:
         company_introduction = soup.find_all('span',{'class':'info'})[1].get_text()        
     except:
         company_introduction = 'NULL'
     try:         
         contacts = soup.find_all('div',{'class':'contact-div'})[0].get_text()
     except:
         contacts = 'NULL'
     try:        
         others_1 = soup.find_all('div',{'class':'companyname'})[0]
     except:
         others_1 = 'NULL'
     try:         
         telephone = others_1.find_all('dl')[0].get_text()
     except:
         telephone = 'NULL'
     try:         
         location = others_1.find_all('dl')[1].get_text()
     except:
         location = 'NULL'
     try:         
         company_url = others_1.find_all('dl')[2].a.get('href')
     except:
         company_url = 'NULL'
     try:     
         others_2 = soup.find_all('table',{'class':'comtb'})[0]
     except:
         others_2  = 'NULL'
     try:         
         main_products_or_services = others_2.find_all('td')[1].get_text()
     except:
         main_products_or_services = 'NULL'
     try:         
         main_industry = others_2.find_all('td')[3].get_text()
     except:
         main_industry = 'NULL'
     try:         
         position = others_2.find_all('td')[5].get_text()
     except:
         position = 'NULL'
     try:         
         department = others_2.find_all('td')[7].get_text()
     except:
         department = 'NULL'
     try:         
         scale_of_operation = others_2.find_all('td')[9].get_text()
     except:
         scale_of_operation = 'NULL'
                  
     finally:
          all_info = pd.DataFrame({'公司名称':company_name,
                                   '公司介绍':company_introduction,
                                   '联系人':contacts,
                                   '联系电话':telephone,
                                   '公司位置':location,
                                   '主营产品或服务':main_products_or_services,
                                   '主营行业':main_industry,
                                   '职位':position,
                                   '部门':department,
                                   '经营规模':scale_of_operation,
                                   '公司主页':company_url},index = [0])
     return all_info
 
             
all_information = pd.DataFrame({})
for one_url in company_url :
     all_infos = get_one_url(one_url)
     all_information = pd.concat([all_information,all_infos],ignore_index=True)

#保存公司信息数据
all_information.to_csv('C:/Users/Administrator/Desktop/company_infomation.csv',encoding = 'utf_8_sig')


#%%
#商品供应信息提取
## 首先获取供应信息的url
#观察公司介绍的url和供应信息的url，可以发现只需将公司介绍url的intro，替换为Supply

commodity_information_url = []
for i in range(len(company_url)):
     new_url = company_url[i].replace('intro','Supply')
     commodity_information_url.append(new_url)

#现在  供应信息的url存在   commodity_information_url 这个列表中
     
#commodity_url = 'http://www.51jiuhuo.com/companyShows/Supply/491361.html'
     
def get_commodity_information(commodity_url):
     rawhtml = requests.get(commodity_url).content
     soup = BeautifulSoup(rawhtml,'lxml')
     urls_list = soup.find_all('td',{'align':'left'})
     one_information_url = ['http://www.51jiuhuo.com' + urls_list[i].a.get('href') for i in range(len(urls_list))]
     return one_information_url

commodity_information = []

for commodity_url in commodity_information_url:
     commodity_information = commodity_information + get_commodity_information(commodity_url)
   
     
#%%
     
# 现在获取单个物品的详细信息
# one_information = 'http://www.51jiuhuo.com/buyer/detail/1143215.html'

#%%
def get_one(one_information):
     rawhtml = requests.get(one_information).content
     soup = BeautifulSoup(rawhtml,'lxml')
     try:
         title = soup.find_all('div',{'class':'b2_trust_e'})[0].get_text()
     except:
         title = 'NULL'
     try:         
         others = soup.find_all('div',{'class':'infobox'})[0].find_all('li')
     except:
         others = 'null'
     try:         
         dqj = others[0].get_text()
     except:
         dqj = 'NULL'
     try:         
         zxd = others[1].get_text()
     except:
         zxd = 'NULL'
     try:
         ghl = others[2].get_text()
     except:
         ghl = 'NULL'
     try:         
         jcx = others[3].get_text()
     except:
         jcx = 'NULL'
     try:         
         fhq = others[4].get_text()
     except:
         fhq = 'NULL'
     try:         
         didian = others[5].get_text()
     except:
         didian = 'NULL'
     try:         
         fbrq = others[6].get_text()
     except:
         fbrq = 'NULL'
     try:         
         yxq = others[7].get_text()
     except:
         yxq = 'NULL'
     try:         
         llcs = others[8].get_text()
     except:
         llcs = 'NULL'
     try:         
         description = soup.find_all('div',{'id':'divnous'})[0].find_all('p')[0].get_text()
     except:
         description = 'NULL'
     
     finally:
          one_info = pd.DataFrame({'标题':title,
                                   '当前价':dqj,
                                   '最小起订':zxd,
                                   '供货量':ghl,
                                   '几成新':jcx,
                                   '发货期':fhq,
                                   '所在地':didian ,
                                   '发布日期':fbrq,
                                   '有效期':yxq,
                                   '浏览次数':llcs,
                                   '描述':description},index = [0])
     return one_info
 

products_info = pd.DataFrame({})

for one_information in commodity_information:
     data = get_one(one_information)
     products_info = pd.concat([products_info,data],ignore_index=True)
     
#%% 
products_info.to_csv('C:/Users/Administrator/Desktop/products_infomation.csv',encoding = 'utf_8_sig')    











