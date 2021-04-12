import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

training_images = np.array([])  # ( i, 120, 120, 3) pictures
training_labels = np.array([])  # (i, 4) ymin, xmin, ymax, xmax
training_class = np.array([])   # i object class

training_images, training_labels, training_class = shuffle(training_images, training_labels, training_class)


# 13 layer CNN for bounding box regression

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(120, 120, 3)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(.2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(.2))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(.2))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(.2))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(.5))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(.5))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(.001),
              loss='mean_squared_error',
              metrics=['accuracy'])


# 7 layer CNN for object classification

classify = tf.keras.models.Sequential()
classify.add(tf.keras.layers.InputLayer(input_shape=(120, 120, 3)))
classify.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
classify.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
classify.add(tf.keras.layers.Dropout(.2))
classify.add(tf.keras.layers.Conv2D(48, (3, 3), activation='relu', padding='same'))
classify.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
classify.add(tf.keras.layers.Dropout(.2))
classify.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
classify.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
classify.add(tf.keras.layers.Dropout(.2))
classify.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
classify.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
classify.add(tf.keras.layers.Dropout(.2))
classify.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
classify.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
classify.add(tf.keras.layers.Dropout(.2))
classify.add(tf.keras.layers.Flatten())
classify.add(tf.keras.layers.Dense(512, activation='relu'))
classify.add(tf.keras.layers.Dropout(.5))
classify.add(tf.keras.layers.Dense(256, activation='relu'))
classify.add(tf.keras.layers.Dropout(.5))
classify.add(tf.keras.layers.Dense(48, activation='relu'))
classify.add(tf.keras.layers.Dense(5, activation='softmax'))
classify.compile(optimizer=tf.keras.optimizers.Adam(.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
