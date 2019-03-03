# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 11:07:45 2018

@author: Administrator
"""


#猫狗数据集实例化
from keras import models
from keras import layers
import matplotlib.pyplot as plt

train_dir = r'/cats_and_dogs_small/train'
validation_dir = r'/cats_and_dogs_small/validation'

# 添加一个dropout层
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape = (150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation = 'relu'))
model.add(layers.Dense(1,activation = 'sigmoid'))
# model.summary()

from keras import optimizers

model.compile(loss = 'binary_crossentropy',optimizer = optimizers.RMSprop(lr=1e-4),metrics = ['acc'])
from keras.preprocessing.image import ImageDataGenerator

#利用数据增强训练
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)  # 验证数据不能使用数据增强  

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150,150),
                                                    batch_size=32,
                                                    class_mode='binary')


validation_generator = train_datagen.flow_from_directory(validation_dir,
                                                    target_size=(150,150),
                                                    batch_size=32,
                                                    class_mode='binary')

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=50)

#  model.save('cats_and_dogs_small_2.h5')  #保存模型


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

plt.figure()

epochs = range(1,len(acc)+1)
plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title('Training and Validati on loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.show()
















