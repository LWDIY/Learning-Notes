# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:39:39 2019

@author: Administrator
"""
##    1打开页面超时异常   timeout

import socket
import urllib.request
import urllib.error

url = 'https://www.baidu.com/'
try:
     #为了试验，这里设置的时间非常短为0.01秒，所以程序最后肯定输出time out，如果设成0.1就不会
     response = urllib.request.urlopen(url,timeout=0.01)     
except urllib.error.URLError as e:
     print(type(e.reason))      # reason 返回的可能不是字符串而是一个对象
     if isinstance(e.reason,socket.timeout):
          print('Time Out')

#####   2 URLError
          
from urllib import request,error
url = 'https://cuiqingcai.com/index.htm'    #不存在的网址
try:
     response = request.urlopen(url)
except error.URLError as e:
     print(e.reason)

####   3 HTTPError

from urllib import request,error
url = 'https://cuiqingcai.com/index.htm'
try:
     response = request.urlopen(url)
except error.HTTPError as e:
     print(e.reason,e.code,e.headers,sep = '\n')


# URLError是HTTPError的父类，可以先捕捉子类错误，再捕捉父类错误
from urllib import request,error
url = 'https://cuiqingcai.com/index.htm'
try:
     response = request.urlopen(url)
except error.HTTPError as e:
     print(e.reason,e.code,e.headers,sep = '\n')
except error.URLError as e:
     print(e.reason)
else:
     print('Request Successfully')































     