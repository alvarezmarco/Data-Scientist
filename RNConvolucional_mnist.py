#Author: Marco Alvarez M.
#Este programa es de opensource.
#Reconocimiento-NoComercial-CompartirtIgual
#Se puede distribuir y/o modificar siempre y cuando no sean con fines comerciales.


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

#Cargar el set de datos
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#Las Redes Neuronales son un tipo de redes neuronales artificiales diseñadas para funcionar de forma muy similar a las neuronas de la corteza visual primaria de un cerebro
#humano. Estas han resultado ser ampliamente eficaces en tareas fundamentales de la visión artificial como la clasificación y la segmentación de imágenes.
#Estas redes usan capas de filtros convolucionales de una omás dimensiones, tras las cuales se insertan funciones no lineales de activación. 
#Fase inicial. Extracción de características: Esta es la fase inicial y esta compuesta principalmente por neuronas convolucionales 
#Clasificación. Se basan en el uso de capas as “Densas” formadas por neuronas convencionales, similares a las utilizadas por los modelos de tipo “perceptron”.


#Normalizacion de Imagenes grises
X_train=X_train.reshape(X_train.shape[0], filas, columnas,1)
X_test=X_test.reshape(X_test.shape[0], filas, columnas,1)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train=X_train/255
X_test=X_test/225
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


#Entradas de imágenes de 28*28 píxeles
entradas=tf.keras.Input(shape=(28,28,1))

#Primera capa de Convolución con la función de activación  Unidad Lineal Rectificada relu
modelo=tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu')(entradas)

#Segunda capa de Convolución y aplicación de MaxPoolin
#El filtro max-pooling es una forma de reducción del volumen de salida de las capas convolucionales que permite además incrementar el campo de percepción de la red. 
modelo=tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu')(modelo)
modelo=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(modelo)


#La instrucción Flatten convierte los elementos de la matriz de imagenes de entrada en un array plano. 
#Luego, con la instrucción Dense, añadimos una capa oculta (hidden layer) de la red neuronal. La primera tendrá 1000 nodos, la segunda 500 y la tercera (capa de salida) 100. Para la función de activación usaremos en las capas ocultas ReLu y para la capa de salida SoftMax.
modelo=tf.keras.layers.Flatten()(modelo)

#Capas Densas o Fully Connected: Este tipo de capas están representadas por las neuronas clásicas empleadas en los ya conocidos perceptrones
#Su función suele ser principalmente la de completar el clasificador o regresor final, que será el encargado de pasar de mapas de características 
#a valores concretos en función del objetivo de la red (clasificación o regresión).
modelo=tf.keras.layers.Dense(units=68,activation='relu')(modelo)
#modelo=tf.keras.layers.Dropout(0.25)(x)
#La capa de Dropout es una capa de regularización muy empleada para evitar el overfitting o sobreentrenamiento en las Redes Neuronales
modelo=tf.keras.layers.Dense(units=20,activation='relu')(modelo)
#modelo=tf.keras.layers.Dropout(0.25)(x)
#En esta Línea untis = num_clases = 10 porque en el dataset existen 10 tipos de prendas, por tanto existen 10 salidas para verificar que tipo es 
modelo=tf.keras.layers.Dense(units=num_classes,activation='softmax')(modelo)


#Se aplica el modelo
model =tf.keras.Model(inputs=entradas, outputs=modelo)
model.summary()

#Se compila el modelo
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['categorical_accuracy'])

#se entrena el modelo
model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test,y_test))

#Evaluación del modelo
evaluacion=modelo.evaluate(X_test,y_test,verbose=1)
print(evaluacion)

