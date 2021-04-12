import tensorflow as tf
import numpy as np
import cv2

path = 'D:\\folder\\folder' # Change to folder containing the h5 files

model = tf.keras.models.load_model(path + '\model.h5')
classify = tf.keras.models.load_model(path + '\classify.h5')

vid = cv2.VideoCapture(0)
while True:
    ret, img = vid.read()
    x = img.shape[0]
    y = (img.shape[1] - x) // 2
    img = img[0:x, y:x+y]
    img = cv2.resize(img, (120, 120))
    img = img / 255.0
    predict = model.predict(img.reshape(1, 120, 120, 3))
    y_min = (int(predict[0][0]*119))
    x_min = (int(predict[0][1]*119))
    y_max = (int(predict[0][2]*119))
    x_max = (int(predict[0][3]*119))
    frame1 = img[y_min:y_max, x_min:x_max]
    frame1 = np.pad(frame1, [(max(y_max-y_min, x_max-x_min)-y_max+y_min, 0), (max(y_max-y_min, x_max-x_min)-x_max+x_min, 0), (0, 0)], mode='constant')
    frame1 = cv2.resize(frame1, (120, 120))
    print(classify.predict(frame1.reshape((1, 120, 120, 3))).argmax()+1)
    for i in range(x_min, x_max):
        img[y_min][i], img[y_max][i] = np.array([255, 0, 0]), np.array([255, 0, 0])
    for j in range(y_min, y_max):
        img[j][x_min], img[j][x_max] = np.array([255, 0, 0]), np.array([255, 0, 0])
    cv2.imshow('image', cv2.resize(img, (720, 720)))
