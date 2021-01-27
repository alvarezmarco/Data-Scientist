#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10


# In[2]:


#Preprocesamiento de datos
#Nombres de clases a identificar.
class_name=['airplane', 'autonmobile', 'bird', 'cat', 'deer', 'dog', 'froge', 'horse', 'ship','truck']


# In[3]:


#cargar los datos
(X_train, y_train), (X_test, y_test)=cifar10.load_data()


# In[4]:


#Normalización de imágenes.
X_train=X_train/255.0
X_train.shape


# In[5]:


#Normalización de imágenes.
X_test=X_test/255.0
plt.imshow(X_test[10])


# In[6]:


#Creación de la Red Neuronal Convulcional
model=tf.keras.Sequential()


# In[7]:


#Primera Capa.
#Las capas convolucionales operan sobre los datos de entrada mediante el cálculo de convoluciones discretas con
#bancos de filtros finitos
#Aplicamos el filtro Conv2D y la función de activación lineal (unidad lineal rectificada relu), se usa generealmente para reconocimiento de voz y de imágenes
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[32,32,3]))


# In[8]:


#Segunda Capa
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3, padding='same', activation='relu'))
#Capa. Max Pool
#El filtro max-pooling es una forma de reducción del volumen de salida de las capas convolucionales,permite 
#incrementar el campo de percepción de la red. 
#pool_size: Tamaño del o enventanado del Max Pooling. strides: Factor de stride,padding: Padding 'valid' para permitir modificación de tamañon o 'same' para conservar tamaños
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# In[9]:


#Tercera Capa
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3, padding='same', activation='relu'))


# In[10]:


#Cuarta Capa
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# In[11]:


#Aplanado.
#La instrucción Flatten convierte los elementos de la matriz de imagenes de entrada en un array plano.
model.add(tf.keras.layers.Flatten())


# In[12]:


#Primera Capa dense.
#Con la instrucción Dense, se añade una capa oculta (hidden layer) de la red neuronal. En este ejemplo con 128 nodos
model.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[13]:


#Segunda Capa dense (dropoutput).
#Dropout:Es una capa de regularización muy empleada para evitar el overfitting o sobreentrenamiento en las redes neuronales
#Elimina las  contribuciones de ciertas neuronas junto a sus conexiones de entrada y salida.
#Esta eliminación se realiza de forma aleatoria con una probabilidad de eliminación definida previamente.
#La función de activación que se utiliza Sigmoidal Softmax que es una extensión de la clásica función logística,
#empleada principalmente para clasificación multiclase.
model.add(tf.keras.layers.Dense(units=10, activation='softmax' ))
model.summary()


# In[14]:


#Compilar  la red neuronal.
model.compile( optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])         
            


# In[ ]:


#Entrenar el modelo.
model.fit(X_train, y_train, epochs=5)


# In[ ]:


#Evaluar el modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print ('Test Acuraccy: {}'.format(test_accuracy))

