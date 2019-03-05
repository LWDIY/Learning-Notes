# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:53:57 2019

@author: Administrator
"""

import re
import pandas as pd
import requests
from requests.exceptions import RequestException


def get_one_url(url):
     try:          
          header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}
          response = requests.get(url,headers = header)
          if response.status_code == 200:
               return response.text
          return None
     except RequestException:
          return None
     
     
#def parse_one_page(html):    
#     pattern = re.compile('<dd>.*?board-index.*?>(.*?)</i>.*?data-src="(.*?)".*?name.*?a.*?>(.*?)</a>.*?star.*?>(.*?)</p>.*?releasetime.*?>(.*?)</p>.*?integer.*?>(.*?)</i>.*?fraction.*?>(.*?)</i>.*?</dd>',re.S)
#     items = re.findall(pattern,html)
#     for item in items:
#          yield {'ranking':item[0],
#                 'image':item[1],
#                 'name':item[2],  
#                 'star':item[3].strip()[3:],  # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
#                 'releasetime':item[4].strip()[5:],
#                 'score':item[5] + item[6]
#                    }
          
def parse_one_page(html):    
     pattern = re.compile('<dd>.*?board-index.*?>(.*?)</i>.*?data-src="(.*?)".*?name.*?a.*?>(.*?)</a>.*?star.*?>(.*?)</p>.*?releasetime.*?>(.*?)</p>.*?integer.*?>(.*?)</i>.*?fraction.*?>(.*?)</i>.*?</dd>',re.S)
     items = re.findall(pattern,html)
     
     ranking = [items[i][0] for i in range(len(items))]
     image = [items[i][1] for i in range(len(items))]
     name = [items[i][2] for i in range(len(items))]
     actor = [items[i][3].strip()[3:] for i in range(len(items))]
     releasetime = [items[i][4].strip()[5:] for i in range(len(items))]
     score = [items[i][5] + items[i][6] for i in range(len(items))]
     data = pd.DataFrame({'ranking':ranking,
                          'image':image,
                          'name':name,
                          'actor':actor,
                          'releasetime':releasetime,
                          'score':score})
          
     return data

all_data = pd.DataFrame({})
for i in range(0,100,10):
     url = 'https://maoyan.com/board/4?offset=' + str(i)
     html = get_one_url(url)
     data = parse_one_page(html)     
     all_data = pd.concat([all_data,data],ignore_index=True)

all_data.to_csv(r'C:\Users\Administrator\Desktop\MaoYanTop100.csv',encoding = 'utf_8_sig')
          































