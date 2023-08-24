import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from keras.metrics import accuracy
# Tải dữ liệu Mnist
(train_data, train_label), (val_data, val_label) = mnist.load_data()

# Xử lý dữ liệu
train_data = train_data.astype('float32')
val_data = val_data.astype('float32')
train_data = train_data/255.0
val_data = val_data/255.0
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
val_data = val_data.reshape(val_data.shape[0], 28, 28, 1)

#Build model
model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])
#Fit model
model.fit(train_data, train_label, epochs=20, batch_size=128, validation_data=(val_data, val_label))
model.save('mnist.h5')
print('Saving the model as mnist.h5')

model.evaluate(val_data, val_label)