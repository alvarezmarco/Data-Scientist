# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vumOhmQ3-ohkDyxwd-b8AFmRWsFB9aC9
"""

import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import TensorBoard

batch_size=100
num_classes=10
epochs=10
filas,columnas= 28,28

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train=X_train.reshape(X_train.shape[0], filas, columnas,1)
X_test=X_test.reshape(X_test.shape[0], filas, columnas,1)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train=X_train/255
X_test=X_test/225
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

entradas=tf.keras.Input(shape=(28,28,1))
modelo=tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu')(entradas)
modelo=tf.keras.layers.Conv2D(128,kernel_size=(3,3),activation='relu')(modelo)
modelo=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(modelo)
modelo=tf.keras.layers.Flatten()(modelo)
modelo=tf.keras.layers.Dense(68,activation='relu')(modelo)
#modelo=tf.keras.layers.Dropout(0.25)(x)
modelo=tf.keras.layers.Dense(20,activation='relu')(modelo)
#modelo=tf.keras.layers.Dropout(0.25)(x)
modelo=tf.keras.layers.Dense(num_classes,activation='softmax')(modelo)

model =tf.keras.Model(inputs=entradas, outputs=modelo)
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['categorical_accuracy'])

model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test,y_test))

evaluacion=modelo.evaluate(X_test,y_test,verbose=1)

print(evaluacion)