# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:35:49 2018

@author: Administrator
"""

''' 11月--深度学习  '''

import numpy as np
a = np.array([[0,1],[2,3],[4,5]])
print(a.shape)
b = a.reshape(6,1)
c = b.reshape(2,3)
d = np.zeros((300,20))
e = np.transpose(d)    #  e = d.T  转置


from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
# keras 自带 one—hot 编码
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


''' 例一 ，电影评论分类：二分类问题'''

from keras.datasets import imdb
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words = 10000)
#  num_words = 10000 保留训练数据中前10000个最常出现的单词，舍弃低频单词
word_index = imdb.get_word_index()  # word_index 将单词映射为整数索引的字典
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()]) # 键值颠倒，整数索引映射为单词, items() 函数以列表返回可遍历的(键, 值) 元组数组。
decoded_review = ' '.join([reverse_word_index.get(i - 3,'') for i in train_data[0]])
#  join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串，上面是以空格连接 

# 将整数序列编码为二进制矩阵，才能输入神经网络
import numpy as np
def vectorize_sequences(sequences,dimension = 10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):   
        results[i,sequence] = 1.                  
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# enumerate() 函数用于将一个可遍历的数据对象(如列表 、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# 模型定义和编译
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16,activation = 'relu',input_shape = (10000,)))
model.add(layers.Dense(16,activation = 'relu'))
model.add(layers.Dense(1,activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics = ['accuracy'])

#留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 训练模型
history = model.fit(partial_x_train,partial_y_train,epochs = 20,batch_size = 521,validation_data = (x_val,y_val))
# batch_size = 521 这种方法把数据分为若干个批，按批来更新参数
# 对于validation_data来说，主要就是为了防止过拟合。比如说在训练过程中，查看模型在validation_data上的accuracy
history_dict = history.history
#  history_dict.keys()

#绘制训练损失和验证损失
import matplotlib.pyplot as plt
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values) + 1)
plt.plot(epochs,loss_values,'bo',label = 'Training loss')
plt.plot(epochs,val_loss_values,'b',label = 'Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#绘制训练精度和验证精度

plt.clf()   #清空图像
acc = history_dict['acc'] 
val_acc = history_dict['val_acc']
plt.plot(epochs,acc,'bo',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 训练所有数据
model = models.Sequential()
model.add(layers.Dense(16,activation = 'relu',input_shape = (10000,)))
model.add(layers.Dense(16,activation = 'relu'))
model.add(layers.Dense(1,activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics = ['accuracy'])
model.fit(x_train,y_train,epochs = 4,batch_size = 521)
results = model.evaluate(x_test,y_test)
#预测
model.predict(x_test)



''' 例二 ，新闻分类：多分类问题'''
from keras.datasets import reuters
#data = reuters.load_data()
(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words = 10000)

# 将索引解码为新闻文本
word_index = reuters.get_word_index()
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3,'') for i in train_data[0]])


''' 例三 ，预测房价：回归问题'''
from keras.datasets import boston_housing
(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()
# 数据标准化
mean = train_data.mean(axis= 0)  #  axis = 0 计算每一列均值 ，axis = 1 计算每一行均值
std = train_data.std(axis = 0)
train_data -= mean
train_data /=std

test_data -= mean   #  要用训练数据的均值和标准差
test_data /=std

from keras import models
from keras import layers
def build_model():    # 用函数来构建模型，便于模型多次实例化
    model = models.Sequential()
    model.add(layers.Dense(64,activation = 'relu',input_shape = (train_data.shape[1],)))
    model.add(layers.Dense(64,activation = 'relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer = 'rmsprop',loss = 'mse',metrics = ['mae'])
    return model

# K折交叉验证
import numpy as np
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print('交叉验证 #',i + 1)
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate(       
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
             axis = 0)

    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
             axis = 0)
# concatenate数组拼接,必须指定拼接方向 axis
#    a = np.array([[1, 2], [3, 4]])
#    b = np.array([[5, 6]])
#    c = np.concatenate((a,b),axis = 0)
#    d = np.concatenate([a,b.T],axis = 1)
    
    model = build_model()
    model.fit(partial_train_data,partial_train_targets,epochs = num_epochs,batch_size = 1,verbose = 0)
#    verbose：日志显示，默认为 1 
#    verbose = 0 不在控制台输出日志信息
#    verbose = 1 输出进度条记录
#    verbose = 2 每个epoch输出一行记录
    val_mse,val_mae = model.evaluate(val_data,val_targets,verbose = 0)
    all_scores.append(val_mae)
   
np.mean(all_scores)


# 500次 K折交叉验证结果
num_epochs = 500
all_mae_histories = [] 
for i in range(k):
    print('交叉验证 #',i + 1)
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate(       
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
             axis = 0)

    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
             axis = 0)
    
    model = build_model()
    history = model.fit(partial_train_data,partial_train_targets,
                        epochs = num_epochs,validation_data = (val_data,val_targets),
                        batch_size = 1,verbose = 0)

    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    
# 计算所有轮中K折交叉验证分数平均值
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# 绘制验证分数
import matplotlib.pyplot as plt
plt.plot(range(1,len(average_mae_history) + 1),average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.legend()
plt.show()

#删除前十个点，用指数移动平均值
def smooth_curve(points,factor = 0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1,len(smooth_mae_history) + 1),smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.legend()
plt.show()

# 根据图得到最优epochs = 80，训练最终模型
model = build_model()
model.fit(train_data,train_targets,epochs = 80,batch_size = 1,verbose = 1)
test_mse_score,test_mae_score = model.evaluate(test_data,test_targets)
print('预测房价与实际价格相差：{0:0.1f}'.format(test_mae_score * 1000),'美元')





'''  11月--python与mysql'''

import pandas as pd
import pymysql.cursors
connect = pymysql.Connect(
        host='localhost',  #host(str):      MySQL服务器地址
        port=3306,         #port(int):      MySQL服务器端口号
        user='root',       #user(str):      用户名
        passwd='123456',   #passwd(str):    密码
        db='baobiao',      #db(str):        数据库名称
        charset='utf8')    #charset(str):   连接编码

sql = 'select * from provences'
data = pd.read_sql(sql,connect)  # 将sql中table1的数据存到data里面


# 获取游标
cursor = connect.cursor()

#  创建数据表的sql 语句  并设置name_id 为主键自增长不为空
sql_createTb1 = '''CREATE TABLE data1(
                 本地网址 varchar(100) NOT NULL,
                 乙方单位 varchar(30),
                 甲方公司 varchar(30),
                 甲方电话 varchar(50),
                 PRIMARY KEY(本地网址))'''

sql_createTb2 = '''CREATE TABLE data2(
                 项目金额 int(50))'''

sql_createTb3 = '''CREATE TABLE data3(
                 代理机构 varchar(50),
                 代理电话 varchar(30))'''

# 插入一条数据到moneytb 里面。
sql_insert = "insert into data1(LAST_NAME,AGE,SEX) values('de2',18,'0')"

# 在 execute里面执行SQL语句
try:
    cursor.execute(sql_createTb1)
    cursor.execute(sql_insert1)
    print(cursor.rowcount)
    cursor.execute(sql_createTb2)
    cursor.execute(sql_insert2)
    print(cursor.rowcount)
    cursor.execute(sql_createTb3)
    cursor.execute(sql_insert3)
    print(cursor.rowcount)
    connect.commit()
    
except Exception as e:
    print(e)
    connect.rollback()



# 使用 execute()  方法执行 SQL 的增删改查 insert delete update select
sql_select = "select * from moneys"
sql_insert = "insert into moneys(name_id,LAST_NAME,SEX) values(8,'pythonN8','1')"
sql_update = "update moneys set LAST_NAME='helloPy' where name_id=2"
sql_delete = "delete from user where name_id=1"

# 执行 insert 增加的语句  如果出现异常对异常处理
try:
    cursor.execute(sql_insert)
    print(cursor.rowcount)
    cursor.execute(sql_update)
    print(cursor.rowcount)
    cursor.execute(sql_delete)
    print(cursor.rowcount)

    connect.commit()
except Exception as e:
    print(e)
    connect.rollback()

cursor.close()
connect.close()















