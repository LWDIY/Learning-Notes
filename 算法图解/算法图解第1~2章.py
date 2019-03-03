# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:08:54 2019

@author: Administrator
"""

                         '''第一章二分法查找'''
##若一个有序列表包含n个元素，最多需要log2(n)步找到指定值
import math
math.log(128,2)   #包含128个元素的列表
math.log2(128)

#%%
def binary_search(my_list,item):
     first_loc = 0              #第一个位置
     last_loc = len(my_list)-1  #最后一个位置
     
     while first_loc <= last_loc:           #只要list没有缩减到只包含一个元素
          mid = (first_loc + last_loc)//2   #检查中间元素 自动向下取整    int((first_loc + last_loc)/2)
          guess = my_list[mid]
          
          if guess ==item:
               return mid
          if guess < item:
               first_loc = mid + 1
          else:
               last_loc = mid - 1
     return None
#%% 
my_list = [1,3,5,7,9]                     
print(binary_search(my_list,3))      #返回的是列表中元素的位置索引
print(binary_search(my_list,-1))      


                                '''第二章选择排序'''
#%%
#这个函数返回的是最小值的索引
def findSmallest(arr):
     smallest = arr[0]
     smallest_index = 0
     for i in range(1,len(arr)):
          if arr[i] < smallest:                       
               smallest = arr[i]
               smallest_index = i                    #所有代码等价于 arr.sort()
     return smallest_index

# 排序
def selectionSort(arr):
     newArr = []
     for i in range(len(arr)):
          smallest_index2 = findSmallest(arr)
          newArr.append(arr.pop(smallest_index2))
     return newArr
          
#%%                   
arr = [5,3,6,2,10]
print(selectionSort(arr))










 






 