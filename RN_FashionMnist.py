{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import datetime\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.datasets import fashion_mnist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Se cargas los datos.\n",
    "(X_train, y_train), (X_test, y_test)=fashion_mnist.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Normalizar imágenes\n",
    "X_train=X_train/255.0\n",
    "X_test=X_test/255.0\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Reformar Datos\n",
    "X_train=X_train.reshape(-1,28*28)\n",
    "X_test=X_test.reshape(-1,28*28)\n",
    "X_train.shape\n",
    "X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Creación de la red neuronal.\n",
    "#Un modelo Sequential es apropiado para una pila simple de capas \n",
    "#donde cada capa tiene exactamente un tensor de entrada y un tensor de salida .\n",
    "model=tf.keras.models.Sequential()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Primera capa. Units es la cantidad de neuronas\n",
    "# Función de activación activation='relu'\n",
    "# input_shape=784 Se saca de reformar datos\n",
    "model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Segunda capa, una vez evaluado se ha decidido aumentar una capa para mejorar el modelo\n",
    "model.add(tf.keras.layers.Dense(units=64, activation='relu'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Capa Drop out Deserción. Se usa para que no exista sobreajuste.\n",
    "#La capa Dropout establece aleatoriamente las unidades de entrada en 0 con una frecuencia de velocidad \n",
    "#en cada paso durante el tiempo de entrenamiento, lo que ayuda a evitar el sobreajuste. \n",
    "#Las entradas que no están configuradas en 0 se escalan en 1 / (1 - tasa) de manera que la suma de todas las entradas no cambia.\n",
    "#Esto contrasta con la configuración de trainable = False para una capa de abandono. Entrenar no afecta el comportamiento de la capa, ya que el abandono no tiene ninguna variable / peso que pueda congelarse durante el entrenamiento)\n",
    "model.add(tf.keras.layers.Dropout(0.2))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Capa de salida. \n",
    "# En fashionmnist https://www.kaggle.com/zalando-research/fashionmnist podemos observar que existen 10 tipos de prendas\n",
    "#Por tanto units =10\n",
    "model.add(tf.keras.layers.Dense(units=10, activation='softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Compilar el modelo\n",
    "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "model.summery()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Entrenar el modelo.\n",
    "model.fit(X_train, y_train, epochs=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Evaluando el modelo\n",
    "test_loss, test_accuracy = model.evaluate(X_test, y_test)\n",
    "print ('Test Acuraccy: {}'.format(test_accuracy))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
